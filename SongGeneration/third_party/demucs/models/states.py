#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : states.py
@Time    : 2023/8/8 下午7:01
@Author  : waytan
@Contact : waytan@tencent.com
@License : (C)Copyright 2023, Tencent
@Desc    : Utilities to save and load models.
"""
import functools
import inspect
import warnings
from pathlib import Path
from fractions import Fraction

import torch


def load_state_dict(net, pth_path):
    kwargs = {'sources': ['drums', 'bass', 'other', 'vocal'], 'audio_channels': 2, 'samplerate': 44100,
              'segment': Fraction(39, 5), 'channels': 48, 'channels_time': None, 'growth': 2, 'nfft': 4096,
              'wiener_iters': 0, 'end_iters': 0, 'wiener_residual': False, 'cac': True, 'depth': 4, 'rewrite': True,
              'multi_freqs': [], 'multi_freqs_depth': 3, 'freq_emb': 0.2, 'emb_scale': 10, 'emb_smooth': True,
              'kernel_size': 8, 'stride': 4, 'time_stride': 2, 'context': 1, 'context_enc': 0, 'norm_starts': 4,
              'norm_groups': 4, 'dconv_mode': 3, 'dconv_depth': 2, 'dconv_comp': 8, 'dconv_init': 0.001,
              'bottom_channels': 512, 't_layers': 5, 't_hidden_scale': 4.0, 't_heads': 8, 't_dropout': 0.02,
              't_layer_scale': True, 't_gelu': True, 't_emb': 'sin', 't_max_positions': 10000, 't_max_period': 10000.0,
              't_weight_pos_embed': 1.0, 't_cape_mean_normalize': True, 't_cape_augment': True,
              't_cape_glob_loc_scale': [5000.0, 1.0, 1.4], 't_sin_random_shift': 0, 't_norm_in': True,
              't_norm_in_group': False, 't_group_norm': False, 't_norm_first': True, 't_norm_out': True,
              't_weight_decay': 0.0, 't_lr': None, 't_sparse_self_attn': False, 't_sparse_cross_attn': False,
              't_mask_type': 'diag', 't_mask_random_seed': 42, 't_sparse_attn_window': 400, 't_global_window': 100,
              't_sparsity': 0.95, 't_auto_sparsity': False, 't_cross_first': False, 'rescale': 0.1}
    model = net(**kwargs)
    state_dict = torch.load(pth_path)
    model.load_state_dict(state_dict)
    return model


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, 'cpu')
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def get_state(model, quantizer, half=False):
    """Get the state from a model, potentially with quantization applied.
    If `half` is True, model are stored as half precision, which shouldn't impact performance
    but half the state size."""
    if quantizer is None:
        dtype = torch.half if half else None
        state = {k: p.data.to(device='cpu', dtype=dtype) for k, p in model.state_dict().items()}
    else:
        state = quantizer.get_quantized_state()
        state['__quantized'] = True
    return state


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get('__quantized'):
        quantizer.restore_quantized_state(model, state['quantized'])
    else:
        model.load_state_dict(state)
    return state


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
