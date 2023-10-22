# %%
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from IPython.display import display
from dataclasses import dataclass
from typing import List, Optional, Union
import torch as t
import transformers
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn
from torch.nn import functional as F
import utils

# %%
MAIN = __name__ == "__main__"


@dataclass(frozen=True)
class BertConfig:
    """Constants used throughout the Bert model. Most are self-explanatory.

    intermediate_size is the number of hidden neurons in the MLP (see schematic)
    type_vocab_size is only used for pretraining on "next sentence prediction", which we aren't doing.
    """

    vocab_size: int = 28996
    intermediate_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 512
    dropout: float = 0.1
    type_vocab_size: int = 2
    layer_norm_epsilon: float = 1e-12


@dataclass
class BertOutput:
    """The output of your Bert model.

    logits is used for w2d1 and is the prediction for each token in the vocabulary
    The other fields are used on w2d2 for the sentiment task.
    """

    logits: Optional[t.Tensor] = None
    is_positive: Optional[t.Tensor] = None
    star_rating: Optional[t.Tensor] = None


config = BertConfig()


# %%
class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads

        self.wq = nn.Linear(config.hidden_size, config.hidden_size)
        self.wk = nn.Linear(config.hidden_size, config.hidden_size)
        self.wv = nn.Linear(config.hidden_size, config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """
        Q = rearrange(self.wq(x), "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(self.wk(x), "b s (h d) -> b h s d", h=self.num_heads)

        return einsum(
            "batch head seq_q head_size, batch head seq_k head_size -> batch head seq_q seq_k",
            Q,
            K,
        ) / (self.head_size**0.5)

    def forward(self, x: t.Tensor) -> t.Tensor:
        A = self.attention_pattern_pre_softmax(x)
        A = t.softmax(A, dim=-1)
        V = rearrange(self.wv(x), "b s (h d) -> b h s d", h=self.num_heads)
        AV = einsum(
            "batch head seq_k head_size, batch head seq_q seq_k -> batch head seq_q head_size",
            V,
            A,
        )
        AV = rearrange(AV, "b h s d -> b s (h d)")
        AV = self.wo(AV)

        return AV


# %%
class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, tuple, t.Size],
        eps=1e-05,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        dim = (
            1 if isinstance(self.normalized_shape, int) else len(self.normalized_shape)
        )
        self.dims = list(i for i in range(-dim, 0))
        self.device = device
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(self.normalized_shape).to(self.device))
            self.bias = nn.Parameter(t.zeros(self.normalized_shape).to(self.device))

    def forward(self, x: t.Tensor):
        """x and the output should both have the shape (N, *)"""
        mean = x.mean(self.dims, keepdim=True)
        var = x.var(self.dims, unbiased=False, keepdim=True)
        norm = (x - mean) / t.sqrt(var + self.eps)
        if self.elementwise_affine:
            return norm * self.weight + self.bias
        return norm


# %%
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(t.randn(num_embeddings, embedding_dim))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        return self.weight[x]


# %%
class BertMLP(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_res = self.w1(x)
        x_res = F.gelu(x_res)
        x_res = self.w2(x_res)
        x_res = self.dropout(x_res)
        return self.ln(x + x_res)


# %%
class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attn = BertSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_res = self.attn(x)
        x_res = self.dropout(x_res)
        return self.ln(x + x_res)


class BertBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attn = BertAttention(config)
        self.mlp = BertMLP(config)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.attn(x)
        return self.mlp(x)


# %%
class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.tok_embed = Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_embed = Embedding(config.type_vocab_size, config.hidden_size)

        self.ln1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *(BertBlock(config) for _ in range(config.num_layers))
        )
        self.bias = nn.Parameter(t.zeros(config.vocab_size))  # why not use normal_?
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, token_type_ids=None) -> BertOutput:
        if token_type_ids is None:
            token_type_ids = t.zeros_like(input_ids, dtype=t.int64)

        pos = t.arange(input_ids.shape[1]).to(input_ids.device)
        pos = repeat(pos, "n -> b n", b=input_ids.shape[0])

        x = self.tok_embed(input_ids)
        x = x + self.type_embed(token_type_ids)
        x = x + self.pos_embed(pos)
        x = self.dropout(self.ln1(x))
        x = self.blocks(x)
        x = self.fc(x)
        x = F.gelu(x)
        x = self.ln2(x)
        x = einsum(
            "vocab hidden, batch seq hidden -> batch seq vocab",
            self.tok_embed.weight,
            x,
        )
        out = x + self.bias

        return BertOutput(logits=out)


# %%
def load_pretrained_weights(config: BertConfig) -> Bert:
    hf_bert = utils.load_pretrained_bert()
    my_bert = Bert(config)

    hf_bert_state = hf_bert.state_dict().items()
    my_bert_state = my_bert.state_dict().items()

    import pandas as pd

    df = pd.DataFrame.from_records(
        [
            (hk, hv.shape, mk, mv.shape)
            for ((hk, hv), (mk, mv)) in zip(hf_bert_state, my_bert_state)
        ],
        columns=["hf_name", "hf_shape", "model_name", "model_shape"],
    )
    with pd.option_context("display.max_rows", None):
        display(df)

    def _copy(x, y):
        x.detach().copy_(y)

    def _copy_weight_bias(x, y):
        _copy(x.weight, y.weight)
        if getattr(x, "bias", None) is not None:
            _copy(x.bias, y.bias)

    # Set every parameter to NaN so we know if we missed one
    for p in my_bert.parameters():
        p.requires_grad = False
        p.fill_(t.nan)

    # Copy the embeddings
    _copy_weight_bias(my_bert.tok_embed, hf_bert.bert.embeddings.word_embeddings)
    _copy_weight_bias(my_bert.pos_embed, hf_bert.bert.embeddings.position_embeddings)
    _copy_weight_bias(my_bert.type_embed, hf_bert.bert.embeddings.token_type_embeddings)
    _copy_weight_bias(my_bert.ln1, hf_bert.bert.embeddings.LayerNorm)

    from transformers.models.bert.modeling_bert import BertLayer

    my_block: BertBlock
    hf_block: BertLayer

    for my_block, hf_block in zip(my_bert.blocks, hf_bert.bert.encoder.layer):  # type: ignore
        _copy_weight_bias(my_block.attn.attn.wq, hf_block.attention.self.query)
        _copy_weight_bias(my_block.attn.attn.wk, hf_block.attention.self.key)
        _copy_weight_bias(my_block.attn.attn.wv, hf_block.attention.self.value)
        _copy_weight_bias(my_block.attn.attn.wo, hf_block.attention.output.dense)
        _copy_weight_bias(my_block.attn.ln, hf_block.attention.output.LayerNorm)
        _copy_weight_bias(my_block.mlp.w1, hf_block.intermediate.dense)
        _copy_weight_bias(my_block.mlp.w2, hf_block.output.dense)
        _copy_weight_bias(my_block.mlp.ln, hf_block.output.LayerNorm)

    _copy_weight_bias(my_bert.fc, hf_bert.cls.predictions.transform.dense)
    _copy_weight_bias(my_bert.ln2, hf_bert.cls.predictions.transform.LayerNorm)
    my_bert.bias.detach().copy_(hf_bert.cls.predictions.bias)

    return my_bert


if MAIN:
    my_bert = load_pretrained_weights(config)
    for name, p in my_bert.named_parameters():
        assert (
            p.is_leaf
        ), "Parameter {name} is not a leaf node, which will cause problems in training. Try adding a detach somewhere"


# %%
def predict(model: Bert, tokenizer, text: str, k=15) -> List[List[str]]:
    model.eval()
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    output = model(tokens).logits
    preds = output[tokenizer.mask_token_id == tokens]
    top_preds = preds.topk(k, dim=-1).indices
    return [tokenizer.decode(mask) for mask in top_preds]
