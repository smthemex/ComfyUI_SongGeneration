#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : pretrained.py
@Time    : 2023/8/8 下午7:22
@Author  : waytan
@Contact : waytan@tencent.com
@License : (C)Copyright 2023, Tencent
@Desc    : Loading pretrained models.
"""
from pathlib import Path

import yaml

from .apply import BagOfModels
from .htdemucs import HTDemucs
from .states import load_state_dict


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument("-n", "--name", default=None,
                       help="Pretrained model name or signature. Default is htdemucs.")
    parser.add_argument("--repo", type=Path,
                        help="Folder containing all pre-trained models for use with -n.")


def get_model_from_yaml(yaml_file, model_file):
    bag = yaml.safe_load(open(yaml_file))
    model = load_state_dict(HTDemucs, model_file)
    weights = bag.get('weights')
    segment = bag.get('segment')
    return BagOfModels([model], weights, segment)
