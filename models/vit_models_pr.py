# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from functools import reduce
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm,Parameter
from torch.nn.modules.utils import _pair
from scipy import ndimage
import models.configs as configs
import torch.nn.functional as F
from .modeling_resnet import ResNetV2
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
import copy


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def mul(a, b):
    "Same as a * b."
    return a * b

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis,g_prompts):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.g_prompts = g_prompts
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        if self.g_prompts is not None:
            B = hidden_states.shape[0]
            ## prefix for  key and value respectively
            prompts =  self.g_prompts.expand(B, -1,2, -1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(torch.cat([hidden_states,prompts[:,:,0,:]],dim=1))
            mixed_value_layer = self.value(torch.cat([hidden_states,prompts[:,:,1,:]],dim=1))
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis,g_prompts= None):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis,g_prompts)
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis,prompt_dict):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i in prompt_dict:
                layer = Block(config, vis,prompt_dict[i])
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis,prompt_dict = {}):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis,prompt_dict)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
        
class VisionTransformer_m(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, vis=False, args = None):
        super(VisionTransformer_m, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        self.args =args
        self.topk = 1
        self.num_group = args.key_prompt
        self.share_blocks = args.share_blocks
        self.share_blocks_g = args.share_blocks_g
        self.hidden_size = config.hidden_size
        self.prompt_common_g = None
        self.selection = False
        self.cluster_size = {i:0 for i in range(self.args.key_prompt)}
        self.cluster_size_g = {i:0 for i in range(self.args.key_prompt)}
        self.cluster_size_l = {i:0 for i in range(self.args.key_prompt)}
        
        val = math.sqrt(6. / float(3 * reduce(mul, config.patches["size"], 1) + config.hidden_size))  # noqa
        if len(self.share_blocks)>0:
            self.prompt_common = nn.ParameterList([nn.Parameter(torch.zeros(
                1,args.n_prompt, config.hidden_size))]+ [nn.Parameter(torch.zeros(
                1,1, config.hidden_size)) for _ in range(len(self.share_blocks)-1)])
        if self.args.key_prompt > 0:
            self.prompt_embeddings = nn.ModuleDict({str(j):nn.ParameterList([nn.Parameter(torch.zeros(
                1,1, config.hidden_size)) for _ in range(self.num_group)]) for j in range(len(self.share_blocks_g))})   
            self.ortho_keys(self.num_group,config)
            for key in self.prompt_embeddings.keys():
                for prompt in self.prompt_embeddings[key]:
                    nn.init.uniform_(prompt.data.data, -val, val)
        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u
    def clear_cluster(self):
        self.cluster_size = {k:v/10 for k,v in self.cluster_size.items()}
        self.cluster_size_c = {k: {c: j/10 for c,j in v.items()} for k,v in self.cluster_size_c.items()}
    def ortho_keys(self,num_prompt,config):
        initial_cluster_centers = torch.zeros(
            num_prompt, config.hidden_size, dtype=torch.float
        )
        nn.init.xavier_uniform_(initial_cluster_centers)

        orthogonal_cluster_centers = torch.zeros(
            num_prompt, config.hidden_size, dtype=torch.float
        )
        orthogonal_cluster_centers[0] = initial_cluster_centers[0]
        for i in range(1, num_prompt):
            project = 0
            for j in range(i):
                project += self.project(
                    initial_cluster_centers[j], initial_cluster_centers[i])
            initial_cluster_centers[i] -= project
            orthogonal_cluster_centers[i] = initial_cluster_centers[i] / \
                torch.norm(initial_cluster_centers[i], p=2)

        initial_cluster_centers = orthogonal_cluster_centers
        self.prompt_keys = Parameter(
            initial_cluster_centers, requires_grad=True)
    def forward(self, x,indexes=None,embedding_dict=None):
        B = x.shape[0]
        x = self.transformer.embeddings(x)
        output_dict ={}
        need_pro = True
        if self.selection:
             # use preprocessed embeddings if exsiting
            if indexes is not None:
                need_pro = False
                emb_list = []
                for id in indexes:
                    if int(id) in embedding_dict.keys():
                        emb_list.append(embedding_dict[int(id)])
                    else:
                        need_pro  = True
                        break   
            if need_pro: 
                output_f = self.forward_g(x)     
                out_x = output_f['out_x']
                if indexes is not None:
                    for num,id in enumerate(indexes):
                        if int(id) not in embedding_dict.keys():
                            embedding_dict[int(id)] = out_x[num].detach().cpu()
            else:
                out_x = torch.stack(emb_list,dim=0)
                out_x = out_x.to(self.args.device)
            if indexes is not None:    
                output_dict['embedding_dict'] = embedding_dict
            topk,reduced_sim,_ = self.clusters_g(out_x)
            output_dict['reduced_sim'] = reduced_sim
        for i,layer_block in enumerate(self.transformer.encoder.layer):
            if i in self.share_blocks:
                prompts =  self.prompt_common[self.share_blocks.index(i)].expand(B, -1, -1)
                x=  torch.cat((
                            x[:, :1, :],
                            prompts,
                            x[:, 1:, :]
                        ), dim=1)
            if i in self.share_blocks_g and self.selection:
                prompt_list = self.prompt_embeddings[str(self.share_blocks_g.index(i))]
                lis_prompt = []
                for param in prompt_list:
                    lis_prompt.append(param)
                ensemble_prompts = torch.cat(lis_prompt,dim=1)[0][topk[:,:]]
                x=  torch.cat((
                            x[:, :1, :],
                            ensemble_prompts,
                            x[:, 1:, :]
                        ), dim=1)
            x, weights = layer_block(x)
        x = self.transformer.encoder.encoder_norm(x)
        if self.selection:
            cls_token = x[:,:len(self.share_blocks_g)+len(self.share_blocks)+1].mean(1)
        else:
            cls_token = x[:,:len(self.share_blocks)+1].mean(1)
        logits = self.head(cls_token)
        output_dict['logits']  = logits
        return output_dict
    def forward_g(self, x):
        B = x.shape[0]
        output_f = {}
        with torch.no_grad():
            for i,layer_block in enumerate(self.transformer.encoder.layer):
                ### for shared layers 
                if i in self.share_blocks and self.args.domain_query:  
                    prompts =  self.prompt_common[self.share_blocks.index(i)].expand(B, -1, -1)
                    x=  torch.cat((
                                x[:, :1, :],
                                prompts.detach(),
                                x[:, 1:, :]
                            ), dim=1)
                x, weights = layer_block(x)
                #### datasets with domain feature shift
                if self.args.dataset in ['office'] and i == 5:
                    break
            x = self.transformer.encoder.encoder_norm(x)
            out_x = x[:,0]
        output_f['out_x'] = out_x 
        return output_f
    def freeze(self):
        for k, p in self.transformer.named_parameters():
            if "prompt" not in k :
                p.requires_grad = False
    def train(self,mode=True):
        self.training = mode
        if mode:
            self.transformer.encoder.eval()
            self.transformer.embeddings.eval()
        else:
            for module in self.children():
                module.train(mode)
    def clusters_g(self,x):
        query_norm = F.normalize(F.relu(x),dim=-1)
        prompt_key_norm = F.normalize(self.prompt_keys,dim=-1)
        prompt_sim  = torch.matmul(query_norm,prompt_key_norm.T)
        if self.training: 
            size_vector = torch.Tensor([list(self.cluster_size_g.values())])
            if size_vector.sum() != 0:
                prob = size_vector/size_vector.sum()
                ### command out this code ofr soft clustering if your cluster data is quit imbalanced 
                # th_l = 1.0/self.num_group * 2
                # th_s = 1.0/self.num_group * 0.5
                # prob = torch.where((prob > th_s) & (prob < th_l), torch.tensor(1.0/self.num_group), prob)
                prompt_sim = (1- prompt_sim)*((prob).to(self.args.device)+5e-4)
            else:
                prompt_sim = (1- prompt_sim)
            scores, topk = prompt_sim.topk(1, 1, False, True)
            for group in torch.flatten(topk[:,0:1]).cpu().detach().numpy():
                self.cluster_size[int(group)] +=1
            batched_key_norm = prompt_key_norm[topk[:,0:1]]        
            matched_sim = batched_key_norm[:,0,:] * query_norm[:,:]
            matched_sim = -matched_sim.sum(1)
            reduced_sim = matched_sim.sum()/ matched_sim.shape[0]
            return topk,reduced_sim,scores
        else:
            prompt_sim = (1- prompt_sim)
            scores, topk = prompt_sim.topk(self.topk, 1, False, True)
            batched_key_norm = prompt_key_norm[topk[:,:]]
            matched_sim = batched_key_norm[:,0,:] * query_norm[:,:]
            matched_sim = (1-matched_sim)
            reduced_sim = matched_sim.sum()/ matched_sim.shape[0]
            return topk,reduced_sim,scores
    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
