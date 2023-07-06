import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from hkey.nn import HKRPQLinear
from rpq.nn import RPQEmbedding
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from typing import Optional


class HKRPQParallelBlock(nn.Module):
    """ Parallel HKRPQ transformer block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442

    Adapted from TIMM implementation.
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4,
            codebooks_ratio=1,
            sigma=10,
            threshold=0.98,
            shared_codebooks=False,
            codebooks=None
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.sigma = sigma
        self.threshold = threshold
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim
        out_proj_in_dim = mlp_hidden_dim + dim
        out_proj_out_dim = 2 * dim
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        self.out_split = [dim] * 2

        self.in_norm = nn.LayerNorm(dim, elementwise_affine=False)
        in_ratio = max(1, in_proj_out_dim//dim)
        out_ratio = max(1, out_proj_out_dim//out_proj_in_dim)
        self.in_proj = HKRPQLinear(dim, in_proj_out_dim, 
                                   num_heads*codebooks_ratio, 
                                   n_clusters=num_heads*in_ratio, 
                                   bias=False, shared_codebooks=shared_codebooks,
                                   sigma=sigma, threshold=threshold)
        self.out_proj = HKRPQLinear(out_proj_in_dim, out_proj_out_dim, 
                                   num_heads*codebooks_ratio, 
                                   n_clusters=num_heads*out_ratio, 
                                   bias=False, shared_codebooks=shared_codebooks,
                                   sigma=sigma, threshold=threshold)
        if shared_codebooks:
            self.in_proj.rpqweight.codebooks = codebooks[0]
            self.out_proj.rpqweight.codebooks = codebooks[1]

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        res = x

        # Combined MLP fc1 & qkv projections
        x, q, k, v = self.in_proj(self.in_norm(x)).split(self.in_split, dim=-1)

        # Dot product attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x_attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
        ).transpose(1, 2).reshape(B, N, C)

        # Combined MLP fc2 & attn_output projection 
        x_mlp, x_attn = self.out_proj(torch.cat([x, x_attn], dim=-1)).split(self.out_split, dim=-1)

        # Residual connections
        x = x_mlp + x_attn + res
        return x


class HKRPQTransformerConfig(PretrainedConfig):

    def __init__(self,
                 vocab_size=32768,
                 vocab_cluster_size=256,
                 dim=768,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 codebooks_ratio=1,
                 sigma=10,
                 threshold=0.98,
                 mask_token_id=119,
                 pad_token_id=120,
                 shared_codebooks=True,
                 gradient_checkpointing=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.vocab_cluster_size = vocab_cluster_size
        self.dim = dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.codebooks_ratio = codebooks_ratio
        self.sigma = sigma
        self.threshold = threshold
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.shared_codebooks = shared_codebooks  
        self.gradient_checkpointing = gradient_checkpointing


class HKRPQTransformerEncoder(nn.Module):
    def __init__(self, config: HKRPQTransformerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.codebooks_ratio = config.codebooks_ratio
        self.sigma = config.sigma
        self.threshold = config.threshold
        self.shared_codebooks = config.shared_codebooks
        self.gradient_checkpointing = config.gradient_checkpointing

        if (self.dim % (self.num_heads*self.codebooks_ratio) != 0) or ((self.dim*self.mlp_ratio + self.dim) % (self.num_heads*self.codebooks_ratio)) != 0:
            raise ValueError(f"""dim({self.dim}) must be divisible by num_heads*codebooks_ratio({self.num_heads*self.codebooks_ratio})
            and dim*mlp_ratio+dim({self.dim*self.mlp_ratio + self.dim}) must be divisible by num_heads*codebooks_ratio({self.num_heads*self.codebooks_ratio})""")

        self.embed_tokens = RPQEmbedding(config.vocab_size, config.dim, self.num_heads)

        if self.shared_codebooks:
            self.codebooks = nn.ParameterList(                    
                [torch.empty(self.num_heads*self.codebooks_ratio, 256, self.dim // (self.num_heads*self.codebooks_ratio)),
                torch.empty(self.num_heads*self.codebooks_ratio, 256, (self.dim*self.mlp_ratio +self.dim) // (self.num_heads*self.codebooks_ratio))]
            )
            for cb in self.codebooks:
                nn.init.kaiming_uniform_(cb, a=math.sqrt(5))
        else:
            self.codebooks=None

        self.blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.blocks.append(HKRPQParallelBlock(self.dim, self.num_heads, self.mlp_ratio, 
                                                self.codebooks_ratio, self.sigma, self.threshold, 
                                                self.shared_codebooks, self.codebooks))


    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, 
                     device: torch.device, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min).to(device)
    

    def forward(self, input_ids, attention_mask=None):
        input_embeds = self.embed_tokens(input_ids)
        attention_mask = self._expand_mask(attention_mask, input_embeds.dtype, input_embeds.device) if attention_mask is not None else None

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                input_embeds = checkpoint(
                    create_custom_forward(blk),
                    input_embeds,
                    attention_mask,
                )
            else:
                input_embeds = blk(input_embeds, attention_mask)
        return input_embeds


class HKRPQTransformerPreTrainedModel(PreTrainedModel):
    config_class = HKRPQTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HKRPQParallelBlock"]

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (HKRPQTransformerEncoder)):
            module.gradient_checkpointing = value


class HKRPQTransformerModel(HKRPQTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = HKRPQTransformerEncoder(config)

    def forward(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)
    
class HKRPQTransformerForMaskedLM(HKRPQTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = HKRPQTransformerEncoder(config)
        expansion_ratio = config.vocab_size//config.dim
        self.lm_head = HKRPQLinear(config.dim, config.vocab_size, 
                                   config.num_heads*config.codebooks_ratio, 
                                   config.num_heads*expansion_ratio, 
                                   bias=False, sigma=expansion_ratio*config.sigma, 
                                   threshold=config.threshold/expansion_ratio)
        self.lm_head.codebooks = self.encoder.embed_tokens.rpqweight.codebooks
        self.lm_head.codes = self.encoder.embed_tokens.rpqweight.codes
        self.lm_head.init_hk()

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return (loss, logits)
    
class HKRPQTransformerForCausalLM(HKRPQTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = HKRPQTransformerEncoder(config)
        expansion_ratio = config.vocab_size//config.dim
        self.lm_head = HKRPQLinear(config.dim, config.vocab_size, 
                                   config.num_heads*config.codebooks_ratio, 
                                   n_clusters=config.vocab_cluster_size, 
                                   bias=False, sigma=expansion_ratio*config.sigma, 
                                   method='max', threshold=config.threshold/expansion_ratio)
        self.lm_head.codebooks = self.encoder.embed_tokens.rpqweight.codebooks
        self.lm_head.codes = self.encoder.embed_tokens.rpqweight.codes

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return (loss, logits)