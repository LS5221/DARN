#
# from copy import deepcopy
# from typing import Any, List, Literal, Optional, Tuple, Union, cast
# import numpy as np
# import rtdl_our
# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch.nn import Parameter  # type: ignore[code]
# from tqdm import trange
# import lib
# import json
# from IPS import sampling, compute_ips
# class BaseModel(nn.Module):
#     category_sizes: List[int]  # torch.jit does not support list[int]
#
#     def __init__(self, config: Config, dataset: lib.Dataset, n_bins: Optional[int]):
#         super().__init__()
#         assert dataset.X_num is not None
#         lower_bounds = dataset.X_num['train'].min().tolist()
#         upper_bounds = dataset.X_num['train'].max().tolist()
#         self.num_embeddings = (
#             NumEmbeddings(
#                 D.n_num_features,
#                 config.model.d_num_embedding,
#                 config.model.num_embedding_arch,
#                 config.model.periodic,
#                 config.model.autodis,
#                 n_bins,
#                 config.model.memory_efficient,
#             )
#             if config.model.num_embedding_arch
#             else DICEEmbeddings(
#                 cast(int, config.model.d_num_embedding), lower_bounds, upper_bounds
#             )
#             if config.model.dice
#             else None
#         )
#         self.category_sizes = dataset.get_category_sizes('train')
#         d_cat_embedding = (
#             config.model.d_num_embedding
#             if self.category_sizes
#             and (
#                 config.is_transformer
#                 or config.model.d_cat_embedding == 'd_num_embedding'
#             )
#             else config.model.d_cat_embedding
#         )
#         assert d_cat_embedding is None or isinstance(d_cat_embedding, int)
#         self.cat_embeddings = (
#             None
#             if d_cat_embedding is None
#             else rtdl_our.CategoricalFeatureTokenizer(
#                 self.category_sizes, d_cat_embedding, True, 'uniform'
#             )
#         )
#
#     def _encode_input(
#         self, x_num: Optional[Tensor], x_cat: Optional[Tensor]
#     ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
#         if self.num_embeddings is not None:
#             assert x_num is not None
#             t = torch.isnan(x_num)
#             if(torch.any(t)):
#                 print("youqdhqo")
#             x_num = self.num_embeddings(x_num)
#         if self.cat_embeddings is not None:
#             assert x_cat is not None
#             t = torch.isnan(x_cat)
#             if (torch.any(t)):
#                 print("youqdhqo")
#             x_cat = self.cat_embeddings(x_cat)
#         elif x_cat is not None:
#             x_cat = torch.concat(
#                 [
#                     nn.functional.one_hot(x_cat[:, i], category_size)  # type: ignore[code]
#                     for i, category_size in enumerate(self.category_sizes)
#                 ],
#                 1,
#             )
#         return x_num, x_cat
#
#
# class NonFlatModel(BaseModel):
#     def __init__(self, config: Config, dataset, n_bins: Optional[int]):
#         super().__init__(config, dataset, n_bins)
#         assert config.model.transformer is not None
#         assert self.num_embeddings is not None
#         transformer_options = deepcopy(config.model.transformer)
#         if config.model.transformer_default:
#             transformer_options = (
#                 rtdl_our.FTTransformer.get_default_transformer_config(
#                     n_blocks=transformer_options.get('n_blocks', 3)
#                 )
#                 | transformer_options
#             )
#         elif config.model.transformer_baseline:
#             transformer_options = (
#                 rtdl_our.FTTransformer.get_baseline_transformer_subconfig()
#                 | transformer_options
#             )
#         d_cat_embedding = (
#             None if self.cat_embeddings is None else self.cat_embeddings.d_token
#         )
#         d_embedding = config.model.d_num_embedding or d_cat_embedding
#         assert d_embedding is not None
#         self.cls_embedding = rtdl_our.CLSToken(d_embedding, 'uniform')
#         self.main_module = rtdl_our.Transformer(
#             d_token=d_embedding,
#             **transformer_options,
#             d_out=dataset.nn_output_dim,
#         )
#         self.num_continuous = dataset.n_num_features
#         self.num_categories = dataset.n_cat_features
#         self.pt_mlp = simple_MLP(
#             [d_embedding * (self.num_continuous + self.num_categories),
#              6 * d_embedding * (self.num_continuous + self.num_categories) // 5,
#              d_embedding * (self.num_continuous + self.num_categories) // 2])
#         if self.num_categories != 0:
#             self.mlp1 = sep_MLP(d_embedding, self.num_categories, dataset.categories)
#         self.mlp2 = sep_MLP(d_embedding, self.num_continuous, np.ones(self.num_continuous).astype(int))
#     def forward(self, x_num, x_cat, num_mask, cat_mask, num_ips, cat_ips):
#         assert x_num is not None or x_cat is not None
#         x = self._encode_input(x_num, x_cat)
#         for x_ in x:
#             if x_ is not None:
#                 assert x_.ndim == 3
#         ips = torch.concat([num_ips, cat_ips], 1)
#         mask = torch.concat([num_mask, cat_mask], 1)
#         x = torch.concat([x_ for x_ in x if x_ is not None], 1)
#         x = self.cls_embedding(x)
#         return self.main_module(x, ips, mask)