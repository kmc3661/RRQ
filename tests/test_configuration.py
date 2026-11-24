import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from rrq.config import add_quant_args, model_config, scale_path, validate_decoder
from rrq.linear import QuantizedLinear, install
from rrq.scales import validate_scales


def test_separate_weight_activation_and_size_explicit_models():
    p=argparse.ArgumentParser();add_quant_args(p)
    a=p.parse_args(['--model','qwen3_vl_32b','--w-bits','4','--a-bits','8'])
    assert (a.model,a.w_bits,a.a_bits)==('qwen3_vl_32b',4,8)
    assert model_config(a.model)['layers']==64
    assert model_config(a.model)['checkpoint']=='Qwen/Qwen3-VL-32B-Instruct'


def test_invalid_weight_precision_and_old_backend_fail():
    fc=torch.nn.Linear(128,8,bias=False).bfloat16()
    for kwargs in ({'w_bits':8},{'backend':'native'}):
        try:QuantizedLinear(fc,**kwargs)
        except ValueError:pass
        else:raise AssertionError('Unsupported precision/backend was accepted')


def test_model_conversion_counts_follow_64_layer_decoder():
    layers=[object() for _ in range(64)]
    model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    with patch('rrq.linear.install_layer', return_value=7) as convert:
        assert install(model,a_bits=8,w_bits=4,name='qwen3_vl_32b')==448
        assert convert.call_count==64
        assert convert.call_args.kwargs['w_bits']==4
    try:validate_decoder(model,'qwen3_vl_8b')
    except ValueError:pass
    else:raise AssertionError('8B/32B mismatch accepted')


def test_size_specific_scales_cannot_fall_back_to_8b():
    for name in ('qwen3_vl_8b','llava_ov15_8b','internvl35_8b'):
        for a in (4,6,8):assert scale_path(name,4,a).is_file()
    try:scale_path('qwen3_vl_32b',4,4)
    except FileNotFoundError as e:assert 'qwen3_vl_32b' in str(e)
    else:raise AssertionError('32B silently reused 8B scales')


def test_scale_validation_rejects_partial_layers():
    model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[object()]*64)))
    try:validate_scales(model,[])
    except ValueError:pass
    else:raise AssertionError('Partial scale file accepted')


def test_installed_model_registry_matches_visible_configuration():
    root=Path(__file__).resolve().parents[1]
    assert json.loads((root/'configs/models.json').read_text())==json.loads((root/'rrq/model_registry.json').read_text())


def test_decode_graph_counts_follow_64_layer_decoder_without_cuda():
    from rrq.runtime.decode_graph import install as install_graphs
    from torch import nn
    # Only graph registration is tested; no GPU stream or graph capture is run.
    class ToyRMSNorm(nn.Module):
        def forward(self,x):return x
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('q_proj','k_proj','v_proj','o_proj'):setattr(self,name,nn.Linear(32,32,bias=False))
            self.q_norm=ToyRMSNorm();self.k_norm=ToyRMSNorm();self.head_dim=32
        def forward(self,hidden_states,**kwargs):return (hidden_states,)
    # The adapter patches the framework's RoPE function in the forward namespace.
    Attention.forward.__globals__['apply_rotary_pos_emb']=lambda q,k,*args:(q,k)
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.self_attn=Attention();self.mlp=nn.Linear(32,32,bias=False)
            self.input_layernorm=ToyRMSNorm();self.post_attention_layernorm=ToyRMSNorm()
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.ModuleList(Layer() for _ in range(64));self.norm=ToyRMSNorm()
    model=SimpleNamespace(model=SimpleNamespace(language_model=Decoder()))
    with patch('torch.cuda.Stream', return_value=object()):result=install_graphs(model)
    assert result['post_attention_coalesced_layers']==64
    assert result['qkv_groups']==64
    assert len(model._ptq_decode_graph_context.graphs)==129
