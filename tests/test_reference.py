import torch
from rrq.quantization import hadamard, encode_weight, encode_activation, integer_linear
from rrq.scales import save, load
from rrq.linear import QuantizedLinear
from rrq.analysis import newly_zeroed

def test_rotation_preserves_dot_before_bf16_rounding():
    torch.manual_seed(0)
    x = torch.randn(2, 128, dtype=torch.float64)
    w = torch.randn(16, 128, dtype=torch.float64)
    assert torch.allclose(hadamard(x) @ hadamard(w).T, x @ w.T, atol=3e-6, rtol=3e-6)

def test_refit_never_increases_normalized_weight_error():
    torch.manual_seed(1); w = torch.randn(32, 256).bfloat16()
    a, b = encode_weight(w, 0), encode_weight(w, 3)
    target = a.group_value.float() / a.row_scale.float()[:, None]
    for enc in (a, b):
        integer = enc.codes.int().reshape(32, 2, 128) * enc.multiplier.int()[:, :, None] + enc.offset.int()[:, :, None]
        assert integer.min() >= -128 and integer.max() <= 127
    ae = (a.value.float() / a.row_scale.float()[:, None] - target).square().sum()
    be = (b.value.float() / b.row_scale.float()[:, None] - target).square().sum()
    assert be <= ae
    packed = b.packed(); assert torch.equal(packed & 15, b.codes[:, ::2])
    assert torch.equal(packed >> 4, b.codes[:, 1::2])

def test_nonzero_denominator():
    x = torch.tensor([[0., 1., 2., 0.], [1., 1., 0., 0.]])
    q = torch.tensor([[0., 0., 2., 0.], [1., 0., 0., 0.]])
    assert newly_zeroed(x, q) == 50.0

def test_scale_serialization_is_exact_and_nonpickle(tmp_path):
    values = [('a', ('b', 'c'), torch.randn(64).abs().bfloat16())]
    save(tmp_path/'scales.npz', values); restored = load(tmp_path/'scales.npz')
    assert values[0][:2] == restored[0][:2]
    assert torch.equal(values[0][2], restored[0][2])

def test_reference_handles_bits_and_zero_input():
    torch.manual_seed(3); fc = torch.nn.Linear(128, 32, bias=False).bfloat16()
    for bits in (4, 6, 8):
        m = QuantizedLinear(fc, bits, 'reference')
        x = torch.zeros(2, 128, dtype=torch.bfloat16)
        q, a, mu = encode_activation(x, bits)
        assert q.count_nonzero() == 0
        assert a.min() > 0 and mu.min() >= 1 and mu.max() <= 8
        assert torch.equal(m(x), torch.zeros(2, 32, dtype=torch.bfloat16))
