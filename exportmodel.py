#!/usr/bin/env python3
"""
Export PyTorch model checkpoint to NumPy .npz format.

This allows the build scripts to run without PyTorch installed,
which is useful for CI environments where PyTorch is too heavy.

Usage:
    python exportmodel.py --model model.pt --output model.npz
"""

import argparse
import json
import numpy as np
import torch
from feedme import AutoregressiveModel


def export_model(model_path: str, output_path: str):
    """Export PyTorch checkpoint to NumPy npz format."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=True)
    arch = checkpoint['architecture']
    charset = checkpoint['charset']
    num_chars = len(charset)

    print(f"Architecture: input={arch['input_size']}, hidden={arch['hidden_sizes']}, output={num_chars}")
    print(f"Charset ({num_chars} chars): {repr(charset[:-1])} + EOS")

    # Create and load model
    model = AutoregressiveModel(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        num_chars=num_chars
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Get quantized parameters
    params = model.get_quantized_params()

    # Build export dict
    export_data = {}

    # Add all weight/bias arrays
    for key, value in params.items():
        export_data[key] = value

    # Add metadata as encoded strings
    export_data['_architecture'] = np.array(json.dumps(arch).encode('utf-8'))
    export_data['_charset'] = np.array(charset.encode('utf-8'))

    # Save to npz
    np.savez(output_path, **export_data)
    print(f"Exported to {output_path}")

    # Print summary
    layer_names = sorted(set(k.replace('_weight', '').replace('_bias', '')
                            for k in params.keys()))
    for name in layer_names:
        w = params[f'{name}_weight']
        b = params[f'{name}_bias']
        print(f"  {name}: weight {w.shape}, bias {b.shape}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to NumPy format')
    parser.add_argument('--model', '-m', default='command_model_autoreg.pt',
                        help='Input PyTorch model checkpoint (.pt)')
    parser.add_argument('--output', '-o', default='model.npz',
                        help='Output NumPy archive (.npz)')
    args = parser.parse_args()

    export_model(args.model, args.output)


if __name__ == '__main__':
    main()
