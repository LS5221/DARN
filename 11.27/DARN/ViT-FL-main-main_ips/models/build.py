# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
from .pretrainmodel import SAINT

def build_model(args, swin_args=None):
    model_type = "saint"
    if model_type == 'swin':
        model = SwinTransformer(img_size=args.img_size,
                                patch_size=swin_args.MODEL.SWIN.PATCH_SIZE,
                                in_chans=swin_args.MODEL.SWIN.IN_CHANS,
                                num_classes=swin_args.MODEL.NUM_CLASSES,
                                embed_dim=swin_args.MODEL.SWIN.EMBED_DIM,
                                depths=swin_args.MODEL.SWIN.DEPTHS,
                                num_heads=swin_args.MODEL.SWIN.NUM_HEADS,
                                window_size=swin_args.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=swin_args.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=swin_args.MODEL.SWIN.QKV_BIAS,
                                qk_scale=swin_args.MODEL.SWIN.QK_SCALE,
                                drop_rate=swin_args.MODEL.DROP_RATE,
                                drop_path_rate=swin_args.MODEL.DROP_PATH_RATE,
                                ape=swin_args.MODEL.SWIN.APE,
                                patch_norm=swin_args.MODEL.SWIN.PATCH_NORM)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=args.img_size,
                        patch_size=swin_args.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=swin_args.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=swin_args.MODEL.NUM_CLASSES,
                        embed_dim=swin_args.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=swin_args.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=swin_args.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=swin_args.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=swin_args.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=swin_args.MODEL.DROP_RATE,
                        drop_path_rate=swin_args.MODEL.DROP_PATH_RATE,
                        ape=swin_args.MODEL.SWIN_MLP.APE,
                        patch_norm=swin_args.MODEL.SWIN_MLP.PATCH_NORM)
    elif model_type == 'saint':
        model = SAINT(
            categories=tuple(args.cat_dims),
            num_continuous=len(args.con_idxs),
            dim=args.embedding_size,
            dim_out=1,
            depth=args.transformer_depth,
            heads=args.attention_heads,
            attn_dropout=args.attention_dropout,
            ff_dropout=args.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings=args.cont_embeddings,
            attentiontype=args.attentiontype,
            final_mlp_style=args.final_mlp_style,
            y_dim=args.y_dim
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
