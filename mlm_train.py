# %%
import os
import sys
import torch as t
import transformers
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm.auto import tqdm
import wandb

from bert import BertLanguageModel, BertConfig, predict
from train_data_prep import cross_entropy_selected, random_mask

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    hidden_size = 512
    assert hidden_size % 64 == 0
    bert_config_tiny = BertConfig(
        max_position_embeddings=512,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_layers=8,
        num_heads=hidden_size // 64,
    )
    config_dict = dict(
        filename="./data/bert_lm.pt",
        lr=0.0002,
        epochs=40,
        batch_size=128,
        weight_decay=0.01,
        mask_token_id=tokenizer.mask_token_id,
        warmup_step_frac=0.01,
        eps=1e-06,
        max_grad_norm=None,
    )
    (train_data, val_data, test_data) = t.load("./data/tokens_2.pt")
    print("Training data size: ", train_data.shape)
    train_loader = DataLoader(
        TensorDataset(train_data),
        shuffle=True,
        batch_size=config_dict["batch_size"],
        drop_last=True,
    )


# %%
def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    """Return the learning rate for use at this step of training"""
    min_lr = max_lr / 10
    if step < max_step * warmup_step_frac:
        return min_lr + (max_lr - min_lr) * (step / (max_step * warmup_step_frac))

    step -= int(max_step * warmup_step_frac)
    max_step -= int(max_step * warmup_step_frac)
    return max_lr + (min_lr - max_lr) * (step / max_step)


if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"])
    lrs = [
        lr_for_step(
            step,
            max_step,
            max_lr=config_dict["lr"],
            warmup_step_frac=config_dict["warmup_step_frac"],
        )
        for step in range(max_step)
    ]
    (fig, ax) = plt.subplots(figsize=(12, 4))
    ax.plot(lrs)
    ax.set(xlabel="Step", ylabel="Learning Rate", title="Learning Rate Schedule")


# %%
def make_optimizer(model: BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    """
    Loop over model parameters and form two parameter groups

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    """
    decay_params = []
    no_decay_params = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            decay_params.append(m.weight)
            if m.bias is not None:
                no_decay_params.append(m.bias)
        else:
            no_decay_params.extend(m.parameters(recurse=False))
    return t.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config_dict["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0},
        ],
        lr=config_dict["lr"],
        eps=config_dict["eps"],
    )


if MAIN:
    test_config = BertConfig(
        max_position_embeddings=4,
        hidden_size=1,
        intermediate_size=4,
        num_layers=3,
        num_heads=1,
        head_size=1,
    )
    optimizer_test_model = BertLanguageModel(test_config)
    opt = make_optimizer(
        optimizer_test_model, dict(weight_decay=0.1, lr=0.0001, eps=1e-06)
    )
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected {expected_num_with_weight_decay} params with weight decay, got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(
        optimizer_test_model.parameters()
    ), "Not all parameters were included in the optimizer"


# %%
def bert_mlm_pretrain(
    model: BertLanguageModel, config_dict: dict, train_loader: DataLoader
) -> None:
    """Train using masked language modeling"""
    model.train().to(device)
    wandb.init(project="bert_mlm", config=config_dict)
    config = wandb.config
    optimizer = make_optimizer(model, config_dict)
    optimizer.zero_grad()
    scaler = t.cuda.amp.GradScaler()
    tokens_seen = 0
    step = 0
    for epoch in tqdm(range(config.epochs)):
        for (batch,) in train_loader:
            B, S = batch.shape
            batch = batch.to(device)
            masked_batch, mask = random_mask(
                batch, config.mask_token_id, model.config.vocab_size
            )
            with t.cuda.amp.autocast():
                pred = model(masked_batch)
                loss = cross_entropy_selected(pred, batch, mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            step_lr = lr_for_step(step, max_step, config.lr, config.warmup_step_frac)
            for g in optimizer.param_groups:
                g["lr"] = step_lr

            if config.max_grad_norm is not None:
                total_params = 0
                clipped_grads = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_params += p.nelement()
                        clipped_grads += (p.grad > config.max_grad_norm).sum().item()
                t.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            # log model input and output
            text = tokenizer.decode(masked_batch[0])
            preds = pred[0][masked_batch[0] == tokenizer.mask_token_id]
            correct = batch[0][masked_batch[0] == tokenizer.mask_token_id]
            tops = preds.topk(3, dim=-1).indices
            predictions = [[tokenizer.decode(t) for t in mask] for mask in tops]
            correct = [tokenizer.decode(t) for t in correct]
            table = wandb.Table(
                columns=["Tokens Seen", "Masked Sentence", "Predictions", "Correct"]
            )
            table.add_data(tokens_seen, text, predictions, correct)

            log_dict = dict(
                train_loss=loss,
                tokens_seen=tokens_seen,
                learning_rate=step_lr,
                prediction_table=table,
            )
            if config.max_grad_norm is not None:
                log_dict["clipped_grad_norm"] = clipped_grads / total_params  # type: ignore

            wandb.log(log_dict, step=tokens_seen)
            tokens_seen += B * S
            step += 1

        if epoch == config.epochs - 1 or (epoch % 10 == 0):
            with open(config.filename, "wb") as f:
                t.save(model.state_dict(), f)


if MAIN:
    model = BertLanguageModel(bert_config_tiny)
    num_parms = sum((p.nelement() for p in model.parameters()))
    print(f"Model has {num_parms} parameters")
    bert_mlm_pretrain(model, config_dict, train_loader)
