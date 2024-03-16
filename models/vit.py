import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.use_adapter = False
    def forward(self, x):
        x = self.net(x)
        if self.use_adapter:
            x = self.adapter1(x)
        return x
    def add_adapters(self, dropout=0.0):
        if not self.use_adapter:
            self.use_adapter = True
            self.adapter1 = AdapterBlock(self.dim, self.dim, dropout)

class AdapterBlock(nn.Module):
    def __init__(self, input_dim, adapter_hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, adapter_hidden_dim)
        self.linear2 = nn.Linear(adapter_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        # initialize weights to a small constant
        for module in [self.linear1, self.linear2]:
            nn.init.normal_(module.weight, 0, .01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x): # x: (seq_len, B, d)
        # down-project
        u = F.relu(self.linear1(self.dropout(x)))  # (seq_len, B, h)
        # up-project
        u = self.linear2(u)  # (seq_len, B, d)
        # skip connection
        u = x + u
        return u

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()

        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.dim = dim
        project_out = not (heads == 1 and dim_head == dim)
        self.use_adapter = False
        self.project_out =project_out
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        
        # print("out: ", out)
        # print("QKV output dimension: ",out.shape)
        # for i in range(8):
        #     print("Head "+str(i+1)+": ")
        #     print("Total mean: ",torch.mean(out[0][i]))
        #     print("QKV output: ",out[0][i])
            # print("Isinstance mean: ",torch.mean(out[0][i],dim=0))
        # print("Whole activation: ",out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.use_adapter:
            out = self.adapter1(out)  # (seq_len, B, d)
        return out
    
    def add_adapters(self, dropout=0.0):
        in_dim = self.dim if self.project_out else self.inner_dim
        if not self.use_adapter:
            self.use_adapter = True
            self.adapter1 = AdapterBlock(in_dim, in_dim, dropout)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.transformers = nn.ModuleList([])
        for _ in range(depth):
            self.transformers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        ll = 0
        for attn, ff in self.transformers:
            ll += 1
            x = attn(x) + x
            x = ff(x) + x
        return x



# def get_b16_config():
#     """Returns the ViT-B/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (16, 16)})
#     config.hidden_size = 768
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 3072
#     config.transformer.num_heads = 12
#     config.transformer.num_layers = 12
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.classifier = 'token'
#     config.representation_size = None
#     return config


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size = 16 , num_classes = 10 , dim=768, depth=12, heads=12, mlp_dim=3072, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

    def produce_feature(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x

    def cal_feature(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return x

    def Show_detail(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return x 