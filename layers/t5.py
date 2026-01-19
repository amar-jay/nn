# https://github.com/Wan-Video/Wan2.2/commit/fa96d0963f1aef3b0f9dd312296d54b13e2da538
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L51
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
import logging
from typing import Literal
import torch
import torch.nn.functional as F
import math
from .tokenizers.huggingface_tokenizer import HuggingfaceTokenizer


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        torch.nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        torch.nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        assert torch.is_tensor(m.gate[0].weight)
        torch.nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        torch.nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        torch.nn.init.normal_(m.fc2.weight, std=m.dim_ff**-0.5)
    elif isinstance(m, T5Attention):
        torch.nn.init.normal_(m.q_proj.weight, std=(m.dim * m.dim_attn)**-0.5)
        torch.nn.init.normal_(m.k_proj.weight, std=m.dim**-0.5)
        torch.nn.init.normal_(m.v_proj.weight, std=m.dim**-0.5)
        torch.nn.init.normal_(m.o_proj.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        torch.nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)



# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L51
class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)

        return self.weight * x

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate='tanh')

class T5FeedForward(torch.nn.Module):
    def __init__(self, dim, dim_ff, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_ff = dim_ff
        self.gate = torch.nn.Sequential(torch.nn.Linear(dim, dim_ff, bias=False), GELU())
        self.fc1 = torch.nn.Linear(dim, dim_ff, bias=False)
        self.fc2 = torch.nn.Linear(dim_ff, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x 

# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class T5Attention(torch.nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1, use_flash_attention=False):
        assert dim_attn % num_heads == 0, "Equal division per head is required."
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.dim = dim
        self.dim_attn = dim_attn

        self.q_proj = torch.nn.Linear(dim, dim_attn, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim_attn, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim_attn, bias=False)
        self.o_proj = torch.nn.Linear(dim_attn, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, context=None, mask=None, pos_bias=None):
        B, T, nh, hs = x.size(0), x.size(1), self.num_heads, self.dim_head

        context = x if context is None else context


        # calculate query, key, values for all heads individually
        q = self.q_proj(x).view(B, -1, nh, hs)
        k = self.k_proj(context).view(B, -1, nh, hs)
        v = self.v_proj(context).view(B, -1, nh, hs)

        # attention bias
        attn_bias = x.new_zeros(B, nh, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            if mask.ndim == 2:
                mask = mask.view(B, 1, 1, -1)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            x = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_bias, 
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0
            )
            x = x.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            # compute attention (T5 does not use scaling), it uses attention bias
            attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
            attn = F.softmax(attn.float(), dim=-1).type_as(attn)
            x = torch.einsum('bnij,bjnc->binc', attn, v)
            x = x.reshape(B, T, nh * hs)

        # output projection
        x = self.dropout(self.o_proj(x))
        return x

class T5RelativeEmbedding(torch.nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = torch.nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

class T5SelfAttention(torch.nn.Module):
    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        if self.shared_pos:
            e = pos_bias
        else:
            assert self.pos_embedding is not None
            e = self.pos_embedding(x.size(1), x.size(1))

        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x

class T5Encoder(torch.nn.Module):
    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, torch.nn.Embedding) \
            else torch.nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = torch.nn.Dropout(dropout)
        self.blocks = torch.nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        if self.shared_pos:
            assert self.pos_embedding is not None
            e = self.pos_embedding(x.size(1), x.size(1)) 
        else: e = None

        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class T5CrossAttention(torch.nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        if self.shared_pos:
            e = pos_bias
        else:
            assert self.pos_embedding is not None
            e = self.pos_embedding(
                x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5Decoder(torch.nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, torch.nn.Embedding) \
            else torch.nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = torch.nn.Dropout(dropout)
        self.blocks = torch.nn.ModuleList([
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        _, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        if self.shared_pos:
            assert self.pos_embedding is not None
            e = self.pos_embedding(x.size(1), x.size(1))
        else:
            e = None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(torch.nn.Module):

    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = torch.nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(
        cfg,
        _type: Literal['encoder','decoder','seq2seq']='seq2seq',
        dtype=torch.float32,
        device:str|int='cpu',
        ):
        assert _type in ['encoder','decoder','seq2seq']

        match _type:
            case 'encoder':
                cfg['vocab'] = cfg.pop('vocab_size')
                cfg['num_layers'] = cfg.pop('encoder_layers')
                _ = cfg.pop('decoder_layers')
                model_cls = T5Encoder
            case 'decoder':
                cfg['vocab'] = cfg.pop('vocab_size')
                cfg['num_layers'] = cfg.pop('decoder_layers')
                _ = cfg.pop('encoder_layers')
                model_cls = T5Decoder
            case 'seq2seq':
                model_cls = T5Model

        with torch.device(device):
            model = model_cls(**cfg)

        model = model.to(dtype=dtype, device=device)

        return model

class T5EncoderModel:
    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path='models_t5_umt5-xxl-enc-bf16.pth',
        tokenizer_path='google/umt5-xxl',
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        cfg = dict(
            vocab_size=256384,
            dim=4096,
            dim_attn=4096,
            dim_ffn=10240,
            num_heads=64,
            encoder_layers=24,
            decoder_layers=24,
            num_buckets=32,
            shared_pos=False,
            dropout=0.1)

        # init model
        model = _t5(
            cfg=cfg,
            _type='encoder',
            dtype=dtype,
            device=device).eval().requires_grad_(False)

        logging.info(f'loading {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
