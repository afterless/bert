# %%
import os
import sys
import time
from dataclasses import dataclass
import torch as t
import transformers
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from bert import BertCommon, BertConfig, load_pretrained_weights
from data_prep import DATA_FOLDER, SAVED_TOKENS_PATH

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if MAIN:
    (train_data, test_data) = t.load(SAVED_TOKENS_PATH)
    bert_config = BertConfig()


# %%
@dataclass(frozen=True)
class BertClassifierOutput:
    """The output of BertClassifier"""

    is_positive: t.Tensor
    star_rating: t.Tensor


class BertClassifier(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.base = BertCommon(bert_config)
        self.fc1 = nn.Linear(bert_config.hidden_size, 2)
        self.fc2 = nn.Linear(bert_config.hidden_size, 1)
        self.dropout = nn.Dropout(bert_config.dropout)
        pass

    def forward(
        self, input_ids: t.Tensor, one_zero_attention_mask: t.Tensor
    ) -> BertClassifierOutput:
        output = self.base(input_ids, one_zero_attention_mask)
        logits = output[:, 0]
        out = self.dropout(logits)
        sent = self.fc1(out)
        star = rearrange(self.fc2(out), "b 1 -> b") * 5 + 5
        return BertClassifierOutput(sent, star)


# %%
def train(tokenizer, config_dict: dict) -> BertClassifier:
    wandb.init(project="bert_finetune", config=config_dict)
    config = wandb.config
    model = BertClassifier(bert_config)
    bert_lm = load_pretrained_weights(bert_config)
    model.base = bert_lm.common
    del bert_lm
    # model.base.load_state_dict(load_pretrained_weights(BertConfig()).common.state_dict())
    for p in model.parameters():
        p.requires_grad_(True)
    model.to(device).train()

    perm = t.randperm(len(train_data))[: config.num_steps]
    small_train_data = TensorDataset(*train_data[perm])
    trainloader = DataLoader(
        small_train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True
    )

    optimizer = t.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    optimizer.zero_grad()
    class_loss_fn = t.nn.CrossEntropyLoss()
    examples_seen = 0
    for _ in range(config.epochs):
        for i, (inp, mask, pos, star) in enumerate(tqdm(trainloader)):
            inp = inp.to(device)
            mask = mask.to(device)
            pos = pos.long().to(device)
            star = star.long().to(device)
            out = model(inp, mask)

            class_loss = class_loss_fn(out.is_positive, pos)
            star_loss = config.star_loss_weight * F.l1_loss(out.star_rating, star)
            (class_loss + star_loss).backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if i % config.step_every == 0:
                wandb.log(
                    {
                        "class_loss": class_loss.item(),
                        "star_loss": star_loss.item(),
                        "exampels_seen": examples_seen,
                    }
                )
                examples_seen += len(inp)
                optimizer.step()
                optimizer.zero_grad()
            if i % 100 == 0:
                table = wandb.Table(
                    columns=[
                        "text",
                        "pred_sent",
                        "true_sent",
                        "pred_star",
                        "true_star",
                    ]
                )
                texts = tokenizer.batch_decode(inp[:5], skip_special_tokens=True)
                for i, text in enumerate(texts):
                    table.add_data(
                        text,
                        "positive"
                        if out.is_positive[i].argmax().item() == 1
                        else "negative",
                        "positive" if pos[i] == 1 else "negative",
                        out.star_rating[i].item(),
                        star[i].item(),
                    )
                wandb.log({"examples": table})
        t.save(model.state_dict(), config.filename)
    return model


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    config_dict = dict(
        lr=1e-5,
        batch_size=8,
        step_every=1,
        epochs=1,
        weight_decay=0.01,
        num_steps=9600,
        star_loss_weight=0.2,
        filename="./data/bert_classifier.pt",
    )
    classifier = train(tokenizer, config_dict)


# %%
def test_set_predictions(
    model: BertClassifier, test_data: TensorDataset, batch_size=256
) -> tuple[t.Tensor, t.Tensor]:
    """
    Return (predicted sentiment, predicted star rating) for each test set example.

    predicted sentiment: shape (len(test_data),) - 0 or 1 for positive/negative
    star: shape (len(test_data), ) - star rating
    """
    testloader = DataLoader(test_data, batch_size=256, shuffle=True, pin_memory=True)
    pred_sentiments = []
    pred_stars = []
    with t.inference_mode():
        for inp, mask, _, _ in tqdm(testloader):
            inp = inp.to(device)
            mask = mask.to(device)
            out = model(inp, mask)
            pred_sentiments.append(out.is_positive.argmax(dim=-1))
            pred_stars.append(out.star_rating)

    return t.cat(pred_sentiments, dim=0), t.cat(pred_stars, dim=0)


if MAIN:
    n = len(test_data)
    perm = t.randperm(n)[:1000]
    test_subset = TensorDataset(*test_data[perm])
    (pred_sentiments, pred_stars) = test_set_predictions(classifier, test_subset)
    correct = pred_sentiments.cpu() == test_subset.tensors[2]
    sentiment_acc = correct.float().mean()
    star_diff = pred_stars.cpu() - test_subset.tensors[3]
    star_error = star_diff.abs().mean()
    print(f"Test accuracy: {sentiment_acc:.2f}")
    print(f"Star MAE: {star_error:.2f}")
