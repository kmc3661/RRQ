"""Shared public model, quantization and scale configuration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads(Path(__file__).with_name('model_registry.json').read_text())

def model_config(name):
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}; choose from {tuple(MODELS)}')
    return MODELS[name]

def add_quant_args(parser, activation_choices=(4, 6, 8)):
    parser.add_argument('--model', choices=tuple(MODELS), default='qwen3_vl_8b')
    parser.add_argument('--load-device', choices=('cpu','cuda'), default=None,
                        help='Checkpoint staging device; inference defaults to CPU for 32B, CUDA for 8B')
    parser.add_argument('--w-bits', type=int, choices=(4,), default=4,
                        help='Weight precision; released packed kernels implement W4')
    parser.add_argument('--a-bits', type=int, choices=activation_choices, default=4,
                        help='Activation precision, independent of weight precision')

def scale_path(name, w_bits, a_bits, explicit=None):
    path = Path(explicit) if explicit else ROOT / 'artifacts/scales' / f"{model_config(name)['artifact']}_W{w_bits}A{a_bits}.npz"
    if not path.is_file():
        raise FileNotFoundError(f'No scales for {name} W{w_bits}A{a_bits}: {path}. Run main_quant.py for this exact model/precision and pass --scales.')
    return path

def validate_decoder(model, name):
    layers = model.model.language_model.layers
    expected = model_config(name)['layers']
    if len(layers) != expected:
        raise ValueError(f'{name} expects {expected} decoder layers; checkpoint has {len(layers)}')
    return len(layers)
