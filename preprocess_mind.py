#!/usr/bin/env python3
"""Preprocess raw MIND-small files into the format expected by
run_all_experiments.py.

Produces, per split:
  behaviors_parsed.tsv  (train only: 1 positive + 4 sampled negatives per row)
  news_parsed.tsv       (BERT-tokenised titles, 20 tokens)
  user2int.tsv          (train only)

Usage:
  python preprocess_mind.py --train_dir <MINDsmall_train> --dev_dir <MINDsmall_dev>
"""
import argparse
import random

import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

NUM_WORDS_TITLE = 20
NEGATIVE_SAMPLING_RATIO = 4


def parse_behaviors(source, target, user2int_path, seed=42):
    print(f"Parse {source}")
    random.seed(seed)
    behaviors = pd.read_table(
        source, header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = {}
    for user in behaviors['user']:
        if user not in user2int:
            user2int[user] = len(user2int) + 1
    pd.DataFrame(user2int.items(), columns=['user', 'int']).to_csv(
        user2int_path, sep='\t', index=False)
    print(f'Num users: {len(user2int)}')

    behaviors['user'] = behaviors['user'].map(user2int)
    for row in tqdm(behaviors.itertuples(), total=behaviors.shape[0],
                    desc="Negative sampling"):
        positive = iter([x for x in row.impressions if x.endswith('1')])
        negative = [x for x in row.impressions if x.endswith('0')]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(NEGATIVE_SAMPLING_RATIO):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        behaviors.at[row.Index, 'impressions'] = pairs

    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]).reset_index(drop=True)
    tqdm.pandas(desc="Splitting impressions")
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.progress_map(
            lambda x: (' '.join(e.split('-')[0] for e in x),
                       ' '.join(e.split('-')[1] for e in x))).tolist())
    behaviors.to_csv(target, sep='\t', index=False,
                     columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_news(source, target, tokenizer):
    print(f"Parse {source}")
    news = pd.read_table(source, header=None, usecols=[0, 3], quoting=3,
                         names=['id', 'title']).fillna(' ')

    def tokenize(row):
        tokens = tokenizer(row.title.lower(), max_length=NUM_WORDS_TITLE,
                           padding='max_length', truncation=True)
        return pd.Series([row.id, str(dict(tokens))], index=['id', 'title'])

    tqdm.pandas(desc="Tokenising titles")
    news.progress_apply(tokenize, axis=1).to_csv(target, sep='\t', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--dev_dir', required=True)
    ap.add_argument('--pretrained_model_name', default='bert-base-uncased')
    args = ap.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    parse_behaviors(f"{args.train_dir}/behaviors.tsv",
                    f"{args.train_dir}/behaviors_parsed.tsv",
                    f"{args.train_dir}/user2int.tsv")
    parse_news(f"{args.train_dir}/news.tsv",
               f"{args.train_dir}/news_parsed.tsv", tokenizer)
    parse_news(f"{args.dev_dir}/news.tsv",
               f"{args.dev_dir}/news_parsed.tsv", tokenizer)


if __name__ == '__main__':
    main()
