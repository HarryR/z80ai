#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) with Overflow Simulation.

Trains neural networks that are aware of:
1. 2-bit weight quantization: weights in {-2, -1, 0, +1, +2}
2. 16-bit signed accumulator overflow during matmul
3. Fixed-point activation scaling

The model learns to avoid overflow naturally during training.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Z80 constraints
MAX_ACCUM = 32767      # 16-bit signed max
MIN_ACCUM = -32768     # 16-bit signed min
ACTIVATION_SCALE = 32  # Fixed-point scale factor


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for non-differentiable ops."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, x_quantized: torch.Tensor) -> torch.Tensor:
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Pass gradient straight through
        return grad_output, None


def quantize_weights_2bit(w: torch.Tensor, hard: bool = True,
                          temperature: float = 1.0) -> torch.Tensor:
    """Quantize weights to 2-bit: {-2, -1, 0, +1} (4 values for 2 bits)

    Args:
        w: Weights tensor
        hard: If True, use STE for gradients
        temperature: 0.0 = float weights, 1.0 = fully quantized
                     Use for progressive quantization during training
    """
    if temperature <= 0:
        return w  # Pure float

    # Scale based on 95th percentile to avoid outlier domination
    scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
    w_scaled = w / scale
    # Only 4 values: {-2, -1, 0, +1} to fit in 2 bits
    w_quant = torch.clamp(torch.round(w_scaled), -2, 1) * scale  # Scale back

    if temperature >= 1.0:
        # Fully quantized
        if hard:
            return StraightThroughEstimator.apply(w, w_quant)
        else:
            return w_quant
    else:
        # Blend: (1-temp)*float + temp*quantized
        w_blend = (1 - temperature) * w + temperature * w_quant
        if hard:
            return StraightThroughEstimator.apply(w, w_blend)
        else:
            return w_blend


def quantization_friendly_loss(w: torch.Tensor) -> torch.Tensor:
    """
    Loss that encourages weights to be close to quantization grid {-2,-1,0,1,2}.

    Instead of quantizing during forward pass (which breaks gradients),
    we add a loss term that pushes weights toward the quantization grid.
    """
    scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
    w_scaled = w / scale

    # Distance to nearest grid point in {-2,-1,0,+1}
    w_rounded = torch.clamp(torch.round(w_scaled), -2, 1)
    distance = (w_scaled - w_rounded).abs()

    return distance.mean()


def quantize_activations(x: torch.Tensor, scale: int = ACTIVATION_SCALE) -> torch.Tensor:
    """Quantize activations to simulated fixed-point."""
    x_scaled = x * scale
    x_quant = torch.round(x_scaled)
    return StraightThroughEstimator.apply(x_scaled, x_quant)


class OverflowAwareLinear(nn.Module):
    """
    Linear layer with 2-bit weight quantization and overflow-aware regularization.

    Uses efficient matmul but adds regularization to prevent overflow:
    1. Quantize weights to {-2,-1,0,+1,+2} using STE
    2. Compute worst-case accumulator magnitude
    3. Penalize if it would exceed 16-bit range
    """

    def __init__(self, in_features: int, out_features: int,
                 simulate_overflow: bool = True,
                 overflow_penalty: float = 0.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.simulate_overflow = simulate_overflow

        # Xavier initialization
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
            * np.sqrt(2.0 / (in_features + out_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Track overflow risk
        self.register_buffer('max_accum_seen', torch.tensor(0.0))

    def forward(self, x: torch.Tensor, quant_temp: float = 1.0) -> torch.Tensor:
        # Use quantized weights with STE (straight-through estimator)
        # quant_temp: 0 = float, 1 = fully quantized (progressive during training)
        w_quant = quantize_weights_2bit(self.weight, hard=True, temperature=quant_temp)
        out = F.linear(x, w_quant, self.bias)

        # Track worst-case accumulator for monitoring
        if self.training and self.simulate_overflow:
            with torch.no_grad():
                w_hard = quantize_weights_2bit(self.weight, hard=False, temperature=1.0)
                worst_case = (w_hard.abs() @ x.abs().T).max()
                self.max_accum_seen = max(self.max_accum_seen, worst_case)

        return out

    def get_quantization_loss(self) -> torch.Tensor:
        """Loss that encourages weights to be quantization-friendly."""
        return quantization_friendly_loss(self.weight)

    def get_overflow_risk(self) -> float:
        """Return ratio of max accumulator to overflow threshold."""
        return (self.max_accum_seen / MAX_ACCUM).item()

    def compute_overflow_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable overflow penalty based on accumulator estimates."""
        w_quant = quantize_weights_2bit(self.weight)

        # Estimate per-neuron accumulator magnitude
        # Sum of |weight| * |activation| gives worst case
        accum_estimate = (w_quant.abs() @ x.abs().T)  # [out_features, batch]

        # Soft penalty: how much we exceed the safe threshold
        # Use a softer threshold (e.g., 80% of max) to have safety margin
        safe_threshold = MAX_ACCUM * 0.8
        overflow = F.relu(accum_estimate - safe_threshold)

        return overflow.mean()

    def reset_overflow_stats(self) -> None:
        self.max_accum_seen.zero_()


class QATCommandClassifier(nn.Module):
    """
    Command classifier with Quantization-Aware Training.

    Uses OverflowAwareLinear layers to simulate Z80 constraints during training.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int,
                 simulate_overflow: bool = True) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.simulate_overflow = simulate_overflow

        # Build layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(OverflowAwareLinear(prev_size, hidden_size, simulate_overflow))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(OverflowAwareLinear(prev_size, num_classes, simulate_overflow))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_overflow_stats(self) -> dict:
        """Get overflow risk statistics for all layers."""
        stats = {}
        layer_idx = 0
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                layer_idx += 1
                stats[f'layer{layer_idx}'] = module.get_overflow_risk()
        return stats

    def compute_total_overflow_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total overflow penalty across all layers."""
        penalty = torch.tensor(0.0, device=x.device)
        current = x
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                penalty = penalty + module.compute_overflow_penalty(current)
                current = module(current)
            elif isinstance(module, nn.ReLU):
                current = module(current)
        return penalty

    def compute_quantization_loss(self) -> torch.Tensor:
        """Compute total quantization-friendly loss across all layers."""
        loss = torch.tensor(0.0)
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                loss = loss + module.get_quantization_loss()
        return loss

    def reset_overflow_stats(self) -> None:
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                module.reset_overflow_stats()

    def get_quantized_params(self) -> dict:
        """Extract 2-bit quantized weights and int16 biases for Z80."""
        params = {}
        layer_idx = 0

        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                layer_idx += 1
                name = f'fc{layer_idx}'

                with torch.no_grad():
                    w = module.weight
                    # Quantize weights
                    w_scale = torch.quantile(w.abs().flatten(), 0.95)
                    w_scaled = w / w_scale if w_scale > 0 else w
                    w_quant = (torch.clamp(torch.round(w_scaled), -2, 1)
                               .cpu().numpy().astype(np.int8))

                    # Quantize biases (scaled by 32)
                    b = module.bias
                    b_quant = torch.round(b * 32).cpu().numpy().astype(np.int16)

                    params[f'{name}_weight'] = w_quant
                    params[f'{name}_bias'] = b_quant

        return params


def train_qat_model(model: QATCommandClassifier,
                    X: torch.Tensor, y: torch.Tensor,
                    epochs: int = 500, lr: float = 0.01,
                    quant_loss_weight: float = 0.01,
                    overflow_penalty: float = 0.0001) -> list[float]:
    """
    Train with QAT-style regularization (not hard quantization).

    The model trains with float weights, but regularization encourages:
    1. Weights to be close to quantization grid {-2,-1,0,1,2}
    2. Accumulator values to stay within 16-bit signed range

    Args:
        model: QAT model
        X: Input features
        y: Labels
        epochs: Training epochs
        lr: Learning rate
        quant_loss_weight: Weight for quantization-friendly regularization
        overflow_penalty: Weight for overflow regularization loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        model.reset_overflow_stats()
        optimizer.zero_grad()

        outputs = model(X)
        ce_loss = criterion(outputs, y)

        # Quantization-friendly regularization (pushes weights toward grid)
        quant_loss = model.compute_quantization_loss() * quant_loss_weight

        # Overflow penalty
        overflow_loss = torch.tensor(0.0)
        if overflow_penalty > 0:
            overflow_loss = model.compute_total_overflow_penalty(X) * overflow_penalty

        loss = ce_loss + quant_loss + overflow_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == y).float().mean()
                overflow_stats = model.get_overflow_stats()
                max_risk = max(overflow_stats.values()) if overflow_stats else 0
                print(f"Epoch {epoch+1:3d}: CE={ce_loss.item():.4f}, "
                      f"Acc={acc:.1%}, QuantLoss={quant_loss.item():.3f}, "
                      f"OverflowRisk={max_risk:.2f}x")

    return losses


# =============================================================================
# Test with larger architecture
# =============================================================================

if __name__ == '__main__':
    import json

    from poc_inference import export_model_to_binary, integer_inference
    from train_commands import SimpleTokenizer

    print("=" * 60)
    print("QAT Training with Overflow-Aware Regularization")
    print("=" * 60)

    # Load training data
    examples = []
    with open('training_data.jsonl') as f:
        for line in f:
            obj = json.loads(line)
            examples.append((obj['text'], obj['command']))

    commands = sorted({ex[1] for ex in examples})
    command_to_idx = {cmd: i for i, cmd in enumerate(commands)}
    idx_to_command = {i: cmd for cmd, i in command_to_idx.items()}

    print(f"Loaded {len(examples)} examples, {len(commands)} classes")

    # Create tokenizer and encode
    tokenizer = SimpleTokenizer(vocab_size=128)
    tokenizer.fit([ex[0] for ex in examples])

    X = tokenizer.encode_batch([ex[0] for ex in examples])
    y_np = np.array([command_to_idx[ex[1]] for ex in examples])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    print(f"Dataset: {X.shape}")

    # Test with LARGER architecture
    print("\n" + "=" * 60)
    print("Training 128→128→64→16 with QAT")
    print("(First layer has 128 inputs - at the overflow limit!)")
    print("=" * 60)

    model = QATCommandClassifier(
        input_size=128,
        hidden_sizes=[128, 64],
        num_classes=len(commands),
        simulate_overflow=True
    )

    train_qat_model(model, X, y, epochs=500, lr=0.01, overflow_penalty=0.0001)

    # Test float accuracy
    print("\n" + "=" * 60)
    print("Float Inference Test:")
    print("=" * 60)

    model.eval()
    test_phrases = [
        "show me the files",
        "copy everything to drive b",
        "delete old backups",
        "edit my document",
        "how much disk space",
        "hello",
        "help me",
        "who are you",
    ]

    with torch.no_grad():
        for phrase in test_phrases:
            x = torch.tensor(tokenizer.encode(phrase), dtype=torch.float32).unsqueeze(0)
            pred = model(x).argmax(dim=1).item()
            cmd = idx_to_command[pred]
            print(f"  '{phrase}' → {cmd}")

    # Check overflow risk
    print(f"\nOverflow risk per layer: {model.get_overflow_stats()}")

    # Test integer inference accuracy
    print("\n" + "=" * 60)
    print("Integer Inference Accuracy (simulating Z80):")
    print("=" * 60)

    params = model.get_quantized_params()
    metadata = export_model_to_binary(params, '/tmp/qat_test.bin')
    with open('/tmp/qat_test.bin', 'rb') as f:
        packed_data = f.read()

    correct_float = 0
    correct_int = 0
    total = min(500, len(examples))

    with torch.no_grad():
        for i in range(total):
            phrase, true_cmd = examples[i]
            vec = tokenizer.encode(phrase)

            # Float inference
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            float_pred = idx_to_command[model(x).argmax(dim=1).item()]

            # Integer inference
            int_out = integer_inference(packed_data, metadata['layers'], vec)
            int_pred = idx_to_command[np.argmax(int_out)]

            if float_pred == true_cmd:
                correct_float += 1
            if int_pred == true_cmd:
                correct_int += 1

    print(f"Float accuracy:   {correct_float}/{total} ({100*correct_float/total:.1f}%)")
    print(f"Integer accuracy: {correct_int}/{total} ({100*correct_int/total:.1f}%)")

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'commands': commands,
        'command_to_idx': command_to_idx,
        'architecture': {
            'input_size': 128,
            'hidden_sizes': [128, 64],
            'num_classes': len(commands),
        }
    }, 'command_model_qat.pt')
    print("\nSaved QAT model to command_model_qat.pt")
