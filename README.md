# BERT Implementation

## Purpose

The project is an implementation of the BERT language model from the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" and more specifically the process was first taking the weights from the "bert-base-cased" HuggingFace model and then later training it on a dataset. Much of this implementation consisted of understanding how attention worked, finetuning it on the IMDB dataset, and training a masked language model on a small GPU configuration using the WikiText-2 dataset.

When training this model, I achieved about 40% accuracy on a test hold of the WikiText-2 dataset, warranting possible improvements in accuacy.

## Setup

If you wish to run this project, ensure (Miniconda)[] is installed on your machine and if you are on macOS or Linux you can run the following:

```bash
ENV_PATH=./bert/.env/
conda create -p $ENV_PATH python=3.9 -y
conda install -p $ENV_PATH pytorch=2.0.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

If you are on Windows, you can run this:

```bash
$env:ENV_PATH='c:\users\<user_name>\bert\.env'
conda create -p $env:ENV_PATH python=3.9 -y
conda install -p $env:ENV_PATH pytorch=2.0.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

## Acknowledgements

Much of this implementation was guided by a program created by Redwood Research. Many thanks to Redwood for creating this program and serving as a stepping stone for this implenentation.
