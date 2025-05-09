import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention_withoutmask(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, ips=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # print(x.shape, ips.shape)
        # print("old", q.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # print("new", q.shape)
        # Compute attention scores
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        old_sim = sim

        # print("neww", sim.shape)
        # Apply key padding mask (if provided)
        if key_padding_mask is not None:
            array = key_padding_mask.cpu().numpy()
            # mask = ~mask

            # 检查每一行是否都为True
            all_true = np.all(array, axis=1)
            all_true_index = torch.tensor(all_true)
            key_padding_mask = key_padding_mask.bool()
            # print("newww", key_padding_mask.shape)
            # print(key_padding_mask)
            key_padding_mask = ~key_padding_mask
            # print("sim", sim.shape, ips.shape)
            # print(key_padding_mask)
            ips = ips.unsqueeze(1).unsqueeze(2)
            # print(ips.shape, ips)

            sim = sim * ips
            sim = sim.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            # print("newww", key_padding_mask.shape)
            # print("s", sim.shape)
            # print("k", key_padding_mask.shape, key_padding_mask)
            sim = sim.float()
            sim[all_true_index] = old_sim[all_true_index]
        # Compute attention weights
        attn = sim.softmax(dim=-1)
        attn = attn.float()
        # Apply attention weights to values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Concatenate heads and apply output layer
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out


class mixAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.0,
            n_feat=0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.out_mlp = simple_MLP([n_feat * 2 * inner_dim, 1000, n_feat * inner_dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ips_enc):
        h = self.heads
        # print(x.shape)
        x = torch.cat((x, ips_enc.to(x.device)), dim=1)
        # print(x.shape)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        # input_flattened = out.view(out.shape[0], -1)
        #
        # # 将展平后的张量输入到 MLP 模型中
        # output_flattened = self.out_mlp(input_flattened)
        #
        # # 将输出张量重塑为目标形状 (256, 29, 32)
        # out = rearrange(output_flattened, 'b (n h)-> b n h ', h=out.shape[2])
        out = self.to_out(out)
        out, _ = torch.chunk(out, 2, dim=1)

        return out


# 行列注意力transformer
class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),

                    # PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, cont_mask=None, cat_mask=None, cat_ips=None, con_ips=None, row_ips=None):
        if cont_mask is not None:
            cont_mask = torch.cat((cat_mask, cont_mask), dim=1).bool()
            ips = torch.cat((cat_ips, con_ips), dim=1)
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        # cont_mask = ~cont_mask
        # print(cont_mask)
        # print(x)
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                if (cont_mask == None):
                    x = attn1(x)
                else:
                    x = attn1(x, key_padding_mask=cont_mask, ips=ips)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x, ips=row_ips)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        return x


# 列注意力transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn = nn.MultiheadAttention(dim, heads, attn_dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x, x_cont=None, cont_mask=None, cat_mask=None, cat_ips=None, con_ips=None):
        if cont_mask is not None:
            cont_mask = torch.cat((cat_mask, cont_mask), dim=1).bool()
            ips = torch.cat((cat_ips, con_ips), dim=1)
            # cont_mask = ~cont_mask
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape

        for attn1, ff1 in self.layers:
            # print("Aaaaaaaaaa", x, x.shape, mask, mask.shape)
            if (cont_mask is None):
                x = attn1(x)
            else:
                # x = x.transpose(0, 1)
                # print("123", co, cont_mask.shape,x.shape)
                # co += 1
                # x, _ = self.attn(x, x, x, key_padding_mask = cont_mask, need_weights = False) #torch.Size([59, 256, 32])
                x = attn1(x, key_padding_mask=cont_mask, ips=ips)
                # x = x.transpose(0, 1)
                # print(x.shape)
            x = ff1(x)
        # print("xshape", x.shape)
        return x


class missTransformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, n_feat):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn = nn.MultiheadAttention(dim, heads, attn_dropout)

        for n_layer in range(depth):
            if n_layer == -1:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(
                        mixAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, n_feat=n_feat))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, cont_mask=None, cat_mask=None, cat_ips=None, con_ips=None, ips_enc=None):
        if cont_mask is not None:
            cont_mask = torch.cat((cat_mask, cont_mask), dim=1).bool()
            ips = torch.cat((cat_ips, con_ips), dim=1)
            # cont_mask = ~cont_mask
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        n_layer = 0
        for attn1, ff1 in self.layers:
            # print("Aaaaaaaaaa", x, x.shape, mask, mask.shape)
            if n_layer == -1:
                if (cont_mask is None):
                    x = attn1(x)
                else:
                    x = attn1(x, key_padding_mask=cont_mask, ips=ips)
                x = ff1(x)
            else:
                if (cont_mask is None):
                    x = attn1(x)
                else:
                    x = attn1(x, ips_enc=ips_enc)
                x = ff1(x)
            n_layer += 1
        return x


# mlp模块
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class TabAttention(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=1,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.,
            lastmlp_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        # 为每个连续数据构建映射mlp
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)

    def forward(self, x_categ, x_cont, x_categ_enc, x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim=-1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc, x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else:
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim=-1)
        flat_x = x.flatten(1)
        return self.mlp(flat_x)

