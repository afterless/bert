# %%
import hashlib
import os
import sys
import time
import zipfile
import torch as t
import transformers
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from finetune_data_prep import maybe_download

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
DATA_FOLDER = "./data/"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"tokens_{DATASET}.pt")
# %%
if MAIN:
    path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
    maybe_download(BASE_URL + DATASETS[DATASET], path)
    expected_hexdigest = {
        "103": "0ca3512bd7a238be4a63ce7b434f8935",
        "2": "f407a2d53283fc4a49bcff21bc5f3770",
    }
    with open(path, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()  # out of date?
        assert actual_hexdigest == expected_hexdigest[DATASET]
    print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    z = zipfile.ZipFile(path)

    def decompress(split: str) -> str:
        return z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8")

    train_text = decompress("train").splitlines()
    val_text = decompress("valid").splitlines()
    test_text = decompress("test").splitlines()


# %%
def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    """
    Tokenizer text and rearange into chunks of max_seq length.
    Return (batch, seq) and an integer dtype
    """
    tokens = tokenizer(
        lines,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,
    )["input_ids"]
    out = t.cat([t.tensor(tok, dtype=t.int64) for tok in tokens])
    n = len(out) // max_seq * max_seq
    return rearrange(out[:n], "(b s) -> b s", s=max_seq)


if MAIN:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Toeknizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)


# %%
def flat(x: t.Tensor) -> t.Tensor:
    """Helper function for combining the batch and seq dims"""
    return rearrange(x, "b s ... -> (b s) ...")


def unflat(x: t.Tensor, max_seq: int) -> t.Tensor:
    """Helper function for uncombining the batch and seq dims"""
    return rearrange(x, "(b s) ... -> b s ...", s=max_seq)


def random_mask(
    input_ids: t.Tensor,
    mask_token_id: int,
    vocab_size: int,
    select_frac=0.15,
    max_frac=0.8,
    random_frac=0.1,
) -> tuple[t.Tensor, t.Tensor]:
    """
    Given a batch of tokens, return a copy with tokens replaced according to paper
    input_ids: (batch, seq)
    Return: (model_input, was_masked)
    model_input: (batch, seq) - a new Tensor with replacements made, suitable for passing to BertLanguageModel.
    was_selected: (batch, seq) - 1 if the token at this index will contribute to MLM loss, 0 otherwise
    """

    B, S = input_ids.shape
    out_flat = flat(input_ids.clone())
    n = len(out_flat)
    n_select = int(n * select_frac)
    idx = t.randperm(n, device=input_ids.device)
    start = 0

    n_mask = int(n_select * max_frac)
    mask_idx = idx[start : start + n_mask]
    out_flat[mask_idx] = mask_token_id
    start += n_mask

    n_rand = int(n_select * random_frac)
    rand_idx = idx[start : start + n_rand]
    out_flat[rand_idx] = t.randint(
        0, vocab_size, rand_idx.shape, device=input_ids.device
    )
    start += n_rand

    was_selected_flat = t.zeros_like(out_flat)
    was_selected_flat[idx[:n_select]] = 1
    return unflat(out_flat, S), unflat(was_selected_flat, S)

    # _, S = input_ids.shape
    # masked_tokens = flat(input_ids.detach()).to(device)
    # to_mask = t.rand(masked_tokens.shape) < select_frac

    # mask = t.rand(masked_tokens.shape)
    # mask[mask >= max_frac + random_frac] = masked_tokens[mask >= max_frac + random_frac].to(t.float32)
    # mask[mask <= max_frac] = mask_token_id
    # mask[(max_frac < mask) & (mask < (max_frac + random_frac))] = t.randint(
    #     0, vocab_size, mask[(max_frac <= mask) & (mask < max_frac + random_frac)].shape
    # ).to(t.float32)

    # print(masked_tokens.dtype, mask.dtype)
    # mask = mask.to(t.long)
    # masked_tokens[to_mask] = mask[to_mask]

    # return unflat(masked_tokens, max_seq=S), unflat(to_mask, max_seq=S)


# %%
if MAIN:
    # Implement unigram loss
    sample = train_data[:50]
    B, S = sample.shape
    counts = t.bincount(train_data.flatten(), minlength=tokenizer.vocab_size)
    freqs = counts.float() / t.tensor(train_data.shape).prod()
    top_tokens = t.argsort(counts, descending=True)[:10]
    print("Top tokens: ")
    for tok in top_tokens:
        print(tokenizer.decode(tok), f"{freqs[tok]:.04f}")
    logprobs = repeat(t.log(freqs), "v -> bs v", bs=B * S)
    freq_loss = F.cross_entropy(logprobs, flat(sample))
    print(f"Sample cross-entropy loss of unigram freq: {freq_loss:.2f}")


# %%
def cross_entropy_selected(
    pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor
) -> t.Tensor:
    """
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to MLM loss, 0 otherwise

    Return the mean loss per predicted token.
    """
    if was_selected.sum() == 0:
        return t.tensor(float("nan"), device=was_selected.device)

    pred = rearrange(pred, "b s v -> (b s) v")
    target = flat(target)
    was_selected = flat(was_selected)
    was_selected_idx = t.arange(len(was_selected))[was_selected.bool()]
    pred_t = pred[was_selected_idx]  # (batch, n_selected, vocab_size)
    target_t = target[was_selected_idx]  # (batch, n_selected)
    return F.cross_entropy(pred_t, target_t)


if MAIN:
    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(
        batch, tokenizer.mask_token_id, tokenizer.vocab_size
    )
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")
