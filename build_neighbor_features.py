"""Precompute a co-click *collaborative context* vector per article:
the co-click-weighted mean of the (frozen BERT) title embeddings of its
co-click neighbours. This injects item-based collaborative-filtering
signal -- what an article's co-read neighbours actually are, semantically
-- which pure-content NRMS cannot see. Computed once, offline.

Output: topology_data/neighbor_context.pkl  {news_id -> float32[768]}
Isolated articles receive the zero vector.
"""
import os, json, pickle
import numpy as np
import pandas as pd
import torch
from ast import literal_eval
from scipy import sparse
from transformers import AutoModel
from tqdm import tqdm

TOPO = './topology_data'
TRAIN = '../mind_dataset/MINDsmall_train'
DEV = '../mind_dataset/MINDsmall_dev'
device = torch.device('cpu')  # offline, keep GPUs free for training


def title_embeddings():
    """Frozen BERT mean-pooled title embedding for every news id."""
    bert = AutoModel.from_pretrained('bert-base-uncased').to(device).eval()
    emb = {}
    for split in [TRAIN, DEV]:
        df = pd.read_table(os.path.join(split, 'news_parsed.tsv'), usecols=['id', 'title'])
        rows = df.to_dict('records')
        for i in tqdm(range(0, len(rows), 128), desc=f'BERT {os.path.basename(split)}'):
            batch = rows[i:i+128]
            ids = [r['id'] for r in batch]
            toks = [literal_eval(r['title']) for r in batch]
            input_ids = torch.tensor([t['input_ids'] for t in toks])
            attn = torch.tensor([t['attention_mask'] for t in toks])
            with torch.no_grad():
                h = bert(input_ids=input_ids, attention_mask=attn)[0]  # B,T,768
            m = attn.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            for nid, v in zip(ids, pooled):
                if nid not in emb:
                    emb[nid] = v.numpy().astype(np.float32)
    return emb


def main():
    adj = sparse.load_npz(os.path.join(TOPO, 'coclick_adj.npz')).tocsr()
    news_ids = json.load(open(os.path.join(TOPO, 'news_ids.json')))
    emb = title_embeddings()
    d = 768
    default = np.zeros(d, dtype=np.float32)
    title_mat = np.stack([emb.get(nid, default) for nid in news_ids])  # N,768

    ctx = {}
    n = adj.shape[0]
    for i in tqdm(range(n), desc='neighbour context'):
        s, e = adj.indptr[i], adj.indptr[i+1]
        neigh, w = adj.indices[s:e], adj.data[s:e].astype(np.float32)
        if len(neigh) == 0:
            ctx[news_ids[i]] = default
            continue
        w = w / w.sum()
        ctx[news_ids[i]] = (title_mat[neigh] * w[:, None]).sum(0).astype(np.float32)

    with open(os.path.join(TOPO, 'neighbor_context.pkl'), 'wb') as f:
        pickle.dump(ctx, f)
    nz = sum(1 for v in ctx.values() if np.any(v))
    print(f'neighbour context saved for {nz}/{n} articles')


if __name__ == '__main__':
    main()
