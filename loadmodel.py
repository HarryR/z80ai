"""
Load model parameters from either PyTorch (.pt) or NumPy (.npz) format.

This module allows build scripts to work with either format, enabling
CI environments to run without PyTorch installed.
"""

import json
import numpy as np


def load_model_params(model_path: str) -> tuple[dict, dict, str]:
    """
    Load model parameters from .pt or .npz file.

    Returns:
        params: dict of quantized weights/biases (fc1_weight, fc1_bias, etc.)
        arch: architecture dict with input_size, hidden_sizes
        charset: character set string
    """
    if model_path.endswith('.npz'):
        return _load_npz(model_path)
    elif model_path.endswith('.pt'):
        return _load_pt(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path} (expected .pt or .npz)")


def _load_npz(model_path: str) -> tuple[dict, dict, str]:
    """Load from NumPy npz format."""
    data = np.load(model_path)

    # Extract metadata
    arch = json.loads(bytes(data['_architecture']).decode('utf-8'))
    charset = bytes(data['_charset']).decode('utf-8')

    # Extract params (everything except metadata)
    params = {k: data[k] for k in data.files if not k.startswith('_')}

    return params, arch, charset


def _load_pt(model_path: str) -> tuple[dict, dict, str]:
    """Load from PyTorch checkpoint format."""
    import torch
    from feedme import AutoregressiveModel

    checkpoint = torch.load(model_path, weights_only=True)
    arch = checkpoint['architecture']
    charset = checkpoint['charset']
    num_chars = len(charset)

    model = AutoregressiveModel(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        num_chars=num_chars
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    params = model.get_quantized_params()

    return params, arch, charset
