# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
import torch
from yacs.config import CfgNode as CN

_C = CN()
_C.seed = 0
# Base config files
_C.BASE = ['']


_C.data.T_cache = true
_C.data.dataset = "News"
_C.data.type = "mcar_"
_C.data.missingrate = 0.5
_C.data.ipsnum = 40
_C.data.path = "data/News/mcar_0.5"
_C.data.T.normalization = "quantile"
_C.model.num_embedding_arch = [
    "linear",
    "relu",
]
_C.d_num_embedding = 328
_C.model.transformer.residual_dropout = 0.0
_C.model.transformer.n_blocks = 1
_C.model.transformer.attention_dropout = 0.30138168803582194
_C.model.transformer.ffn_d_hidden_factor = 1.7564330326604605
_C.model.transformer.ffn_dropout = 0.21182739966945235
_C.training.batch_size = 256
_C.training.lr = 0.00019578897201212982
_C.training.weight_decay = 7.501954443620125e-06
_C.bins.count = 229


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    _update_config_from_file(config, "/home/liangqiong/Research/Deep_Learning/Pytorch/VIT/Swin-Transformer-main/configs/swin_tiny_patch4_window7_224.yaml")
    # _update_config_from_file(config, args.cfg)

    config.defrost()

    # config.freeze()


def get_config_tabular(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
