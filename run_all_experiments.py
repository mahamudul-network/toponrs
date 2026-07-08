#!/usr/bin/env python3
"""
TopoNRS Experiment Runner for MDPI Topology Paper.
Uses Hyperbolic Geometry (Poincaré Ball) + Persistent Homology + Topology Regularization.
"""
import os, sys, json, pickle, time, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, BertTokenizer
from ast import literal_eval
from scipy import sparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIND_TRAIN = '../mind_dataset/MINDsmall_train'
MIND_DEV = '../mind_dataset/MINDsmall_dev'
TOPO_DIR = './topology_data'
RESULTS_DIR = './results'

# ===================== HYPERBOLIC GEOMETRY =====================
class HyperbolicOps:
    """Poincaré ball operations for hyperbolic geometry."""

    @staticmethod
    def clip_norm(x, max_norm=2.5, eps=1e-5):
        """Clip the norm of a tangent vector to keep the exp-map stable."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / torch.clamp(norm / max_norm, min=1.0)

    @staticmethod
    def expmap0(v, c=1.0, eps=1e-5):
        """Exponential map at origin (0)."""
        sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    
    @staticmethod
    def logmap0(y, c=1.0, eps=1e-5):
        """Logarithmic map at origin (0)."""
        sqrt_c = torch.sqrt(torch.tensor(c, device=y.device))
        y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=eps)
        return torch.arctanh(torch.clamp(sqrt_c * y_norm, max=1.0 - eps)) * y / (sqrt_c * y_norm)
    
    @staticmethod
    def hyperbolic_distance_poincare(x, y, c=1.0, eps=1e-5):
        """Alternative formulation using the standard Poincaré distance formula."""
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        
        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
        
        x_norm_sq = torch.clamp(x_norm_sq, max=1.0 - eps)
        y_norm_sq = torch.clamp(y_norm_sq, max=1.0 - eps)
        
        numerator = 2.0 * diff_norm_sq
        denominator = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)
        
        arg = 1.0 + numerator / torch.clamp(denominator, min=eps)
        dist = torch.acosh(torch.clamp(arg, min=1.0 + eps))
        return dist.squeeze(-1)

# Collaborative co-click context features (id -> float32[768]); populated
# in main() only when the active variant uses them. Read by both datasets.
COLLAB_FEATURES = {}
COLLAB_DIM = 768

def _collab_feat(nid):
    if not COLLAB_FEATURES:
        return None
    v = COLLAB_FEATURES.get(nid)
    if v is None:
        return torch.zeros(COLLAB_DIM, dtype=torch.float32)
    return torch.as_tensor(v, dtype=torch.float32)

# ===================== DATASET =====================
class NewsDataset(Dataset):
    def __init__(self, news_path, ph_features_path=None):
        self.news_parsed = pd.read_table(news_path, usecols=['id','title'])
        self.news2dict = self.news_parsed.to_dict('index')
        
        self.ph_features = {}
        if ph_features_path and os.path.exists(ph_features_path):
            with open(ph_features_path, 'rb') as f:
                self.ph_features = pickle.load(f)
                
        for k1 in self.news2dict:
            nid = self.news2dict[k1]['id']
            v = literal_eval(self.news2dict[k1]['title'])
            self.news2dict[k1]['title'] = torch.cat([
                torch.tensor(v['input_ids']).unsqueeze(0),
                torch.tensor(v['attention_mask']).unsqueeze(0)
            ], dim=0)
            if self.ph_features:
                raw_feat = torch.tensor(self.ph_features.get(nid, np.zeros(6, dtype=np.float32)))
                self.news2dict[k1]['ph_feat'] = torch.log1p(torch.clamp(raw_feat, min=0))
            else:
                self.news2dict[k1]['ph_feat'] = torch.zeros(6, dtype=torch.float32)
            cf = _collab_feat(nid)
            if cf is not None:
                self.news2dict[k1]['collab_feat'] = cf

    def __len__(self): return len(self.news2dict)
    def __getitem__(self, idx): return self.news2dict[list(self.news2dict.keys())[idx]]

class TopoNegativeSampler:
    """Topology-guided hard negative sampling.

    mode='1hop': draw from the positive's direct co-click neighbourhood
        weighted by co-click strength (the original, flawed choice: these
        are near-positives / likely false negatives).
    mode='2hop': draw from the positive's 2-hop neighbourhood --
        structurally similar articles that are NOT directly co-clicked
        (ambiguous negatives, far less likely to be false negatives).
    """
    def __init__(self, adj_path, news2idx_path, mode='2hop', cand_path=None):
        self.mode = mode
        if mode == '2hop':
            with open(cand_path, 'rb') as f:
                self.cand = pickle.load(f)
            n = sum(1 for v in self.cand.values() if v)
            print(f"Negative sampler: 2-hop ambiguous negatives for {n} articles")
        else:
            self.adj = sparse.load_npz(adj_path).tocsr()
            with open(news2idx_path) as f:
                self.news2idx = json.load(f)
            self.idx2news = {v: k for k, v in self.news2idx.items()}
            print(f"Negative sampler: 1-hop co-click, {self.adj.shape[0]} nodes")

    def sample(self, pos_id, exclude):
        if self.mode == '2hop':
            cands = self.cand.get(pos_id)
            if not cands:
                return None
            order = np.random.permutation(len(cands))
            for j in order[:6]:
                if cands[j] not in exclude:
                    return cands[j]
            return None
        idx = self.news2idx.get(pos_id)
        if idx is None:
            return None
        row = self.adj.getrow(idx)
        neigh, w = row.indices, row.data.astype(np.float64)
        if len(neigh) == 0:
            return None
        for _ in range(4):
            j = np.random.choice(len(neigh), p=w / w.sum())
            cand = self.idx2news[neigh[j]]
            if cand not in exclude:
                return cand
        return None

class TrainDataset(Dataset):
    def __init__(self, behaviors_path, news_path, num_clicked=50, ph_features_path=None,
                 neg_sampler=None, neg_gamma=0.0):
        self.num_clicked = num_clicked
        self.neg_sampler = neg_sampler
        self.neg_gamma = neg_gamma
        self.behaviors = pd.read_table(behaviors_path)
        news_parsed = pd.read_table(news_path, index_col='id', usecols=['id','title'])
        self.news2dict = news_parsed.to_dict('index')
        
        self.ph_features = {}
        if ph_features_path and os.path.exists(ph_features_path):
            with open(ph_features_path, 'rb') as f:
                self.ph_features = pickle.load(f)
                
        for k1 in tqdm(self.news2dict, desc="Processing news"):
            v = literal_eval(self.news2dict[k1]['title'])
            self.news2dict[k1]['id'] = k1
            self.news2dict[k1]['title'] = torch.cat([
                torch.tensor(v['input_ids']).unsqueeze(0),
                torch.tensor(v['attention_mask']).unsqueeze(0)
            ], dim=0)
            if self.ph_features:
                raw_feat = torch.tensor(self.ph_features.get(k1, np.zeros(6, dtype=np.float32)))
                self.news2dict[k1]['ph_feat'] = torch.log1p(torch.clamp(raw_feat, min=0))
            else:
                self.news2dict[k1]['ph_feat'] = torch.zeros(6, dtype=torch.float32)
            cf = _collab_feat(k1)
            if cf is not None:
                self.news2dict[k1]['collab_feat'] = cf

        self.padding = {
            'id': 'PADDED_NEWS',
            'title': torch.cat([
                torch.tensor([101,102]+[0]*18).unsqueeze(0),
                torch.tensor([1,1]+[0]*18).unsqueeze(0)
            ], dim=0),
            'ph_feat': torch.zeros(6, dtype=torch.float32)
        }
        if COLLAB_FEATURES:
            self.padding['collab_feat'] = torch.zeros(COLLAB_DIM, dtype=torch.float32)
        self.all_news_ids = list(self.news2dict.keys())

    def __len__(self): return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        clicked = list(map(int, row.clicked.split()))
        cand_ids = row.candidate_news.split()
        if self.neg_sampler is not None and self.neg_gamma > 0:
            # candidate 0 is the positive; replace a fraction gamma of the
            # negatives with co-click neighbours of the positive
            exclude = set(cand_ids) | set(row.clicked_news.split())
            for c in range(1, len(cand_ids)):
                if np.random.rand() < self.neg_gamma:
                    hard = self.neg_sampler.sample(cand_ids[0], exclude)
                    if hard is not None and hard in self.news2dict:
                        cand_ids[c] = hard
                        exclude.add(hard)
        candidate_news = [self.news2dict[x] for x in cand_ids]
        history_ids = row.clicked_news.split()[:self.num_clicked]
        clicked_news = [self.news2dict[x] for x in history_ids]
        ct = len(clicked_news)
        pad = self.num_clicked - ct
        clicked_news = [self.padding]*pad + clicked_news
        mask = [0]*pad + [1]*ct
        return {"candidate_news": candidate_news, "clicked_news": clicked_news,
                "clicked_news_mask": mask, "clicked": clicked}

class UserDataset(Dataset):
    def __init__(self, behaviors_path, user2int_path, num_clicked=50):
        self.num_clicked = num_clicked
        self.behaviors = pd.read_table(behaviors_path, header=None, usecols=[1,3], names=['user','clicked_news'])
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        for row in self.behaviors.itertuples():
            self.behaviors.at[row.Index, 'user'] = user2int.get(row.user, 0)
    def __len__(self): return len(self.behaviors)
    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        items = row.clicked_news.split()[:self.num_clicked]
        pad = self.num_clicked - len(items)
        return {"user": row.user, "clicked_news_string": row.clicked_news,
                "clicked_news": ['PADDED_NEWS']*pad + items}

class BehaviorsDataset(Dataset):
    def __init__(self, behaviors_path):
        self.behaviors = pd.read_table(behaviors_path, header=None, usecols=range(5),
            names=['impression_id','user','time','clicked_news','impressions'])
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()
    def __len__(self): return len(self.behaviors)
    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        return {"impression_id": row.impression_id, "user": row.user,
                "clicked_news_string": row.clicked_news, "impressions": row.impressions}

# ===================== MODEL =====================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None):
        if K is None: K = Q
        if V is None: V = Q
        B = Q.size(0)
        q = self.W_Q(Q).view(B,-1,self.n_heads,self.d_k).transpose(1,2)
        k = self.W_K(K).view(B,-1,self.n_heads,self.d_k).transpose(1,2)
        v = self.W_V(V).view(B,-1,self.n_heads,self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-1,-2)) / (self.d_k**0.5)
        # BUGFIX: Use F.softmax to prevent NaNs from exp() overflow
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        return ctx.transpose(1,2).contiguous().view(B,-1,self.n_heads*self.d_k)

class AdditiveAttention(nn.Module):
    def __init__(self, q_dim, c_dim):
        super().__init__()
        self.linear = nn.Linear(c_dim, q_dim)
        self.query = nn.Parameter(torch.empty(q_dim).uniform_(-0.1, 0.1))

    def forward(self, x):
        temp = torch.tanh(self.linear(x))
        w = F.softmax(torch.matmul(temp, self.query), dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)

class NewsEncoder(nn.Module):
    def __init__(self, pretrained='bert-base-uncased', n_heads=16, dropout=0.2, q_dim=200,
                 topo_feat_dim=0, hyp_dim=256, use_hyperbolic=True,
                 hyp_fix=False, fusion='add', unfreeze_bert=0, collab_dim=0):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.hyp_fix = hyp_fix
        self.fusion = fusion
        self.collab_dim = collab_dim
        bert = AutoModel.from_pretrained(pretrained)
        self.dim = bert.config.hidden_size
        self.bert = bert
        for param in bert.parameters():
            param.requires_grad = False
        # Optionally fine-tune the top `unfreeze_bert` encoder layers
        if unfreeze_bert > 0:
            for layer in bert.encoder.layer[-unfreeze_bert:]:
                for param in layer.parameters():
                    param.requires_grad = True
        self.mhsa = MultiHeadSelfAttention(self.dim, n_heads)
        self.additive = AdditiveAttention(q_dim, self.dim)
        self.dropout_p = dropout
        self.topo_feat_dim = topo_feat_dim
        if topo_feat_dim > 0:
            self.topo_proj = nn.Sequential(
                nn.Linear(topo_feat_dim, 64), nn.ReLU(),
                nn.Linear(64, self.dim))
            if fusion == 'gate':
                self.gate = nn.Linear(2 * self.dim, self.dim)
        if collab_dim > 0:
            # project the co-click collaborative context and gate-fuse it
            self.collab_proj = nn.Sequential(
                nn.Linear(collab_dim, self.dim), nn.ReLU(),
                nn.Linear(self.dim, self.dim))
            self.collab_gate = nn.Linear(2 * self.dim, self.dim)

        self.to_hyperbolic = nn.Linear(self.dim, hyp_dim)
        # Learnable scale that keeps tangent norms in tanh's responsive
        # region so the exp-map spreads radii across the ball instead of
        # collapsing them onto a thin boundary shell.
        self.hyp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, news):
        inp = {"input_ids": news["title"][:,0].to(device),
               "attention_mask": news["title"][:,1].to(device)}
        v = self.bert(**inp)[0]
        v = self.mhsa(v)
        v = F.dropout(v, p=self.dropout_p, training=self.training)
        v = self.additive(v)
        if self.topo_feat_dim > 0 and 'ph_feat' in news:
            p = self.topo_proj(news['ph_feat'].to(device))
            if self.fusion == 'gate':
                g = torch.sigmoid(self.gate(torch.cat([v, p], dim=-1)))
                v = g * v + (1 - g) * p
            else:
                v = v + p
        if self.collab_dim > 0 and 'collab_feat' in news:
            cp = self.collab_proj(news['collab_feat'].to(device))
            cg = torch.sigmoid(self.collab_gate(torch.cat([v, cp], dim=-1)))
            v = cg * v + (1 - cg) * cp

        raw = self.to_hyperbolic(v)
        if not self.use_hyperbolic:
            return torch.tanh(raw) * 0.5           # base: Euclidean path (unchanged)
        if self.hyp_fix:
            # radial-spread projection: scale then exp-map (no pre-tanh)
            tangent = raw * self.hyp_scale
            tangent = HyperbolicOps.clip_norm(tangent, max_norm=2.5)
            return HyperbolicOps.expmap0(tangent)
        # original (broken) hyperbolic path, kept for reproducing the ablation
        return HyperbolicOps.expmap0(torch.tanh(raw) * 0.5)

class UserEncoder(nn.Module):
    def __init__(self, dim=256, n_heads=16, q_dim=200, use_hyperbolic=True):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        # Cross attention in Tangent space
        self.mhsa = MultiHeadSelfAttention(dim, n_heads)
        self.additive = AdditiveAttention(q_dim, dim)

    def forward(self, history_vectors):
        if not self.use_hyperbolic:
            v = self.mhsa(history_vectors)
            return self.additive(v)
        # Move to tangent space for operations
        hist_tg = HyperbolicOps.logmap0(history_vectors)
        v = self.mhsa(hist_tg)
        user_tg = self.additive(v)
        # Move back to hyperbolic space
        user_hyperbolic = HyperbolicOps.expmap0(user_tg)
        return user_hyperbolic

class TopoNRS(nn.Module):
    def __init__(self, pretrained='bert-base-uncased', n_heads=16, dropout=0.2,
                 q_dim=200, topo_feat_dim=0, hyp_dim=256, use_hyperbolic=True,
                 hyp_fix=False, fusion='add', unfreeze_bert=0, collab_dim=0):
        super().__init__()
        self.use_hyperbolic = use_hyperbolic
        self.news_encoder = NewsEncoder(pretrained, n_heads, dropout, q_dim,
                                        topo_feat_dim, hyp_dim, use_hyperbolic,
                                        hyp_fix, fusion, unfreeze_bert, collab_dim)
        self.user_encoder = UserEncoder(hyp_dim, n_heads, q_dim, use_hyperbolic)
        self.temperature = nn.Parameter(torch.tensor([0.07]))
        self.topo_feat_dim = topo_feat_dim
        self.hyp_dim = hyp_dim

    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        cand_v = torch.stack([self.news_encoder(x) for x in candidate_news], dim=1)
        click_v = torch.stack([self.news_encoder(x) for x in clicked_news], dim=1)

        # Mask padded history items
        mask = torch.stack(clicked_news_mask).to(device).transpose(0,1).unsqueeze(-1).float()

        if self.use_hyperbolic:
            # We must mask in tangent space, because 0 in hyperbolic space is the origin
            click_v_tg = HyperbolicOps.logmap0(click_v) * mask
            click_v = HyperbolicOps.expmap0(click_v_tg)
        else:
            click_v = click_v * mask

        user_v = self.user_encoder(click_v)

        batch_size, num_candidates, _ = cand_v.shape
        if self.use_hyperbolic:
            # Predict using Hyperbolic Distance
            user_vector_expanded = user_v.unsqueeze(1).expand(-1, num_candidates, -1)
            distances = HyperbolicOps.hyperbolic_distance_poincare(
                user_vector_expanded.reshape(-1, self.hyp_dim),
                cand_v.reshape(-1, self.hyp_dim)
            )
            distances = distances.reshape(batch_size, num_candidates)
            scores = -distances / self.temperature
        else:
            # Euclidean dot-product scorer (NRMS)
            scores = torch.bmm(cand_v, user_v.unsqueeze(-1)).squeeze(-1) / self.temperature

        return scores, cand_v, click_v

    def get_news_vector(self, news):
        return self.news_encoder(news)
    def get_user_vector(self, clicked_news_vector):
        return self.user_encoder(clicked_news_vector)
    def get_prediction(self, news_vector, user_vector):
        if self.use_hyperbolic:
            distances = HyperbolicOps.hyperbolic_distance_poincare(news_vector, user_vector)
            return -distances / self.temperature
        return torch.sum(news_vector * user_vector, dim=-1) / self.temperature

# ===================== TOPOLOGY REGULARIZATION =====================
class TopologyRegularizer:
    """Encourages embeddings of co-clicked news to be close."""
    def __init__(self, adj_path, news2idx_path, weight=0.1):
        self.weight = weight
        if os.path.exists(adj_path) and os.path.exists(news2idx_path):
            self.adj = sparse.load_npz(adj_path)
            with open(news2idx_path) as f:
                self.news2idx = json.load(f)
            print(f"Loaded topology graph: {self.adj.shape[0]} nodes")
        else:
            self.adj = None
            self.news2idx = {}

    def compute_loss(self, candidate_news, cand_vectors):
        """Given batch of news candidates and their vectors, compute topology reg loss."""
        if self.adj is None:
            return torch.tensor(0.0, device=device)
        
        loss = torch.tensor(0.0, device=device)
        count = 0
        batch_size = cand_vectors.size(0)
        num_candidates = cand_vectors.size(1)
        
        for b in range(batch_size):
            c_ids = [candidate_news[c]['id'][b] for c in range(num_candidates)]
            indices = [self.news2idx.get(nid, -1) for nid in c_ids]
            
            for i in range(len(indices)):
                if indices[i] < 0: continue
                for j in range(i+1, len(indices)):
                    if indices[j] < 0: continue
                    w = self.adj[indices[i], indices[j]]
                    if w > 0:
                        # Use HYPERBOLIC distance for regularization
                        dist = HyperbolicOps.hyperbolic_distance_poincare(cand_vectors[b, i], cand_vectors[b, j])
                        loss = loss + dist
                        count += 1
                        
        if count > 0:
            loss = loss / count
        return loss * self.weight

# ===================== METRICS =====================
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr) / np.sum(y_true) if np.sum(y_true) > 0 else 0

def calc_metrics(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        n5 = ndcg_score(*pair, 5)
        n10 = ndcg_score(*pair, 10)
        return [auc, mrr, n5, n10]
    except: return [np.nan]*4

@torch.no_grad()
def evaluate(model, dev_dir, train_dir, batch_size=64, ph_features_path=None,
             return_per_impression=False):
    model.eval()
    news_ds = NewsDataset(os.path.join(dev_dir, 'news_parsed.tsv'), ph_features_path)
    news_dl = DataLoader(news_ds, batch_size=batch_size*8, shuffle=False, num_workers=4)
    news2vec = {}
    for mb in tqdm(news_dl, desc="News vectors"):
        ids = mb["id"]
        vecs = model.get_news_vector(mb) if not isinstance(model, nn.DataParallel) else model.module.get_news_vector(mb)
        for i, v in zip(ids, vecs):
            if i not in news2vec: news2vec[i] = v.cpu()
    news2vec['PADDED_NEWS'] = torch.zeros(list(news2vec.values())[0].size())

    user_ds = UserDataset(os.path.join(dev_dir,'behaviors.tsv'),
                          os.path.join(train_dir,'user2int.tsv'))
    user_dl = DataLoader(user_ds, batch_size=batch_size*8, shuffle=False, num_workers=4)
    user2vec = {}
    for mb in tqdm(user_dl, desc="User vectors"):
        for us, cn in zip(mb["clicked_news_string"], zip(*[mb["clicked_news"][i] for i in range(len(mb["clicked_news"]))])):
            if us not in user2vec:
                cv = torch.stack([news2vec[x].to(device) for x in cn]).unsqueeze(0)
                uv = model.get_user_vector(cv) if not isinstance(model, nn.DataParallel) else model.module.get_user_vector(cv)
                user2vec[us] = uv.squeeze(0).cpu()

    beh_ds = BehaviorsDataset(os.path.join(dev_dir,'behaviors.tsv'))
    beh_dl = DataLoader(beh_ds, batch_size=1, shuffle=False, num_workers=4)
    tasks = []
    imp_ids = []
    _get_pred = model.get_prediction if not isinstance(model, nn.DataParallel) else model.module.get_prediction
    for mb in tqdm(beh_dl, desc="Predictions"):
        cv = torch.stack([news2vec[n[0].split('-')[0]].to(device) for n in mb['impressions']])
        uv = user2vec[mb['clicked_news_string'][0]].to(device)
        pred = _get_pred(cv, uv.unsqueeze(0).expand(cv.size(0), -1))
        imp_ids.append(int(mb['impression_id'][0]))
        tasks.append(([int(n[0].split('-')[1]) for n in mb['impressions']], pred.cpu().tolist()))

    with Pool(8) as pool:
        results = list(pool.imap(calc_metrics, tasks))
    r = np.array(results).T
    means = (np.nanmean(r[0]), np.nanmean(r[1]), np.nanmean(r[2]), np.nanmean(r[3]))
    if return_per_impression:
        per_imp = {"impression_ids": imp_ids,
                   "auc": r[0].tolist(), "mrr": r[1].tolist(),
                   "ndcg5": r[2].tolist(), "ndcg10": r[3].tolist()}
        return means, per_imp
    return means

# ===================== TRAINING =====================
def train_model(model_name, model, train_dir, dev_dir, epochs=6, batch_size=64,
                lr=1e-4, topo_reg=None, validate_every=200, ph_features_path=None,
                neg_sampler=None, neg_gamma=0.0, seed=42, out_dir=None,
                max_steps=None, skip_final_eval=False):
    out_dir = out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    ds = TrainDataset(os.path.join(train_dir,'behaviors_parsed.tsv'),
                      os.path.join(train_dir,'news_parsed.tsv'),
                      ph_features_path=ph_features_path,
                      neg_sampler=neg_sampler, neg_gamma=neg_gamma)
    gen = torch.Generator(); gen.manual_seed(seed)
    def _worker_init(worker_id):
        ws = (seed * 1000 + worker_id) % 2**32
        np.random.seed(ws); random.seed(ws)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                    generator=gen, worker_init_fn=_worker_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_auc = 0
    step = 0
    results_log = []
    start = time.time()
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}\nTraining {model_name} | epochs={epochs} batch={batch_size}\n{'='*60}")

    for epoch in range(epochs):
        model.train()
        losses = []
        for i, mb in enumerate(tqdm(dl, desc=f"Epoch {epoch+1}")):
            if max_steps is not None and step >= max_steps:
                break
            step += 1
            pred, cand_v, click_v = model(mb["candidate_news"], mb["clicked_news"], mb["clicked_news_mask"])
            y = torch.zeros(len(pred), dtype=torch.long).to(device)
            loss = criterion(pred, y)
            
            if topo_reg is not None:
                t_loss = topo_reg.compute_loss(mb["candidate_news"], cand_v)
                loss = loss + t_loss
                
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % validate_every == 0:
                auc, mrr, n5, n10 = evaluate(model, dev_dir, train_dir, batch_size, ph_features_path)
                elapsed = time.time() - start
                print(f"\nStep {step} | AUC:{auc:.4f} MRR:{mrr:.4f} nDCG@5:{n5:.4f} nDCG@10:{n10:.4f} | Loss:{np.mean(losses[-100:]):.4f}")
                results_log.append({"model": model_name, "step": step, "epoch": epoch+1,
                    "auc": auc, "mrr": mrr, "ndcg5": n5, "ndcg10": n10,
                    "loss": np.mean(losses[-100:]), "time": elapsed})
                if auc > best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_best.pth"))
                    print(f"  -> New best AUC: {auc:.4f}")
                model.train()

        print(f"Epoch {epoch+1} done. Avg loss: {np.mean(losses):.4f}")

    if skip_final_eval:
        print(f"{model_name}: smoke run finished (final loss {np.mean(losses[-20:]):.4f}), skipping final eval")
        return {"model": model_name, "seed": seed, "smoke": True,
                "loss": float(np.mean(losses[-20:]))}

    if best_auc > 0:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{model_name}_best.pth"), weights_only=True))
    (auc, mrr, n5, n10), per_imp = evaluate(model, dev_dir, train_dir, batch_size,
                                            ph_features_path, return_per_impression=True)
    print(f"\n{'='*60}\n{model_name} FINAL: AUC:{auc:.4f} MRR:{mrr:.4f} nDCG@5:{n5:.4f} nDCG@10:{n10:.4f}\n{'='*60}")

    results_log.append({"model": model_name, "step": "final", "seed": seed,
        "auc": auc, "mrr": mrr, "ndcg5": n5, "ndcg10": n10})

    with open(os.path.join(out_dir, f"{model_name}_results.json"), 'w') as f:
        json.dump({"log": results_log, "final": {"seed": seed, "auc": auc, "mrr": mrr,
                   "ndcg5": n5, "ndcg10": n10}, "per_impression": per_imp}, f)
    return {"model": model_name, "seed": seed, "AUC": auc, "MRR": mrr, "nDCG@5": n5, "nDCG@10": n10}

# ===================== MAIN =====================
# Ablation variants (paper Table "Component ablation"):
#   base = plain NRMS backbone (Euclidean dot-product scorer)
#   gf   = base + local co-click graph features
#   hyp  = base + Poincare-ball scorer
#   ns   = base + topology-guided hard negative sampling
#   full = all components + topology regularisation
def _v(gf=False, hyp=False, ns=False, reg=False, fusion='add',
       neg_mode='1hop', hyp_fix=False, unfreeze=0, collab=False):
    return dict(gf=gf, hyp=hyp, ns=ns, reg=reg, fusion=fusion,
                neg_mode=neg_mode, hyp_fix=hyp_fix, unfreeze=unfreeze,
                collab=collab)

VARIANTS = {
    # --- original (flawed) variants, kept to reproduce the diagnosis ---
    'base': _v(),
    'gf':   _v(gf=True),
    'hyp':  _v(hyp=True),
    'ns':   _v(ns=True),
    'full': _v(gf=True, hyp=True, ns=True, reg=True),
    # --- revised design: 2-hop ambiguous negatives, gated fusion,
    #     radial-spread hyperbolic projection ---
    # Approach B: Euclidean scorer, co-click topology carries the gain
    'e_hn':   _v(ns=True, neg_mode='2hop'),                       # hard negatives alone
    'e_gf':   _v(gf=True, fusion='gate'),                         # gated features alone
    'e_topo': _v(gf=True, ns=True, fusion='gate', neg_mode='2hop'),  # B (full)
    # Approach A: fixed hyperbolic scorer + co-click topology
    'h_fix':  _v(hyp=True, hyp_fix=True),                         # fixed scorer alone
    'h_topo': _v(gf=True, hyp=True, ns=True, reg=True,
                 fusion='gate', neg_mode='2hop', hyp_fix=True),   # A (full)
    # Collaborative co-click context (neighbour title-embedding mean)
    'e_collab':    _v(collab=True),                              # collab feature alone
    'e_collab_gf': _v(collab=True, gf=True, fusion='gate'),      # + graph stats
    'e_best':      _v(collab=True, gf=True, ns=True,
                      fusion='gate', neg_mode='2hop'),           # Euclidean best
    'h_best':      _v(collab=True, gf=True, hyp=True, reg=True,
                      hyp_fix=True, fusion='gate'),              # hyperbolic best
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='full', choices=list(VARIANTS.keys()),
                        help="base=NRMS, full=TopoNRS, gf/hyp/ns=single-component ablations")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.3,
                        help="fraction of negatives replaced by topology-guided hard negatives")
    parser.add_argument('--validate_every', type=int, default=500)
    parser.add_argument('--out_dir', default=None,
                        help="output directory (default: results/seed_runs)")
    parser.add_argument('--max_steps', type=int, default=None,
                        help="stop each epoch after N steps (debugging)")
    parser.add_argument('--smoke', action='store_true',
                        help="short smoke test: 10 steps, no final eval")
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.max_steps = 1, 10

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    cfg = VARIANTS[args.variant]
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, 'seed_runs')
    os.makedirs(out_dir, exist_ok=True)
    run_name = f"{args.variant}_seed{args.seed}"

    adj_path = os.path.join(TOPO_DIR, 'coclick_adj.npz')
    news2idx_path = os.path.join(TOPO_DIR, 'news2idx.json')

    topo_reg = None
    if cfg['reg']:
        topo_reg = TopologyRegularizer(adj_path, news2idx_path, weight=0.1)

    neg_sampler = None
    if cfg['ns']:
        neg_sampler = TopoNegativeSampler(
            adj_path, news2idx_path, mode=cfg['neg_mode'],
            cand_path=os.path.join(TOPO_DIR, 'hard_neg_candidates.pkl'))

    ph_feat_path = os.path.join(TOPO_DIR, 'news_ph_features.pkl') if cfg['gf'] else None

    if cfg.get('collab'):
        global COLLAB_FEATURES
        with open(os.path.join(TOPO_DIR, 'neighbor_context.pkl'), 'rb') as f:
            COLLAB_FEATURES = pickle.load(f)
        print(f"Collaborative context loaded: {len(COLLAB_FEATURES)} articles")

    print("\n" + "="*60 +
          f"\n  Variant: {args.variant} (GF={cfg['gf']} Hyp={cfg['hyp']} "
          f"NS={cfg['ns']}/{cfg['neg_mode']} Reg={cfg['reg']} "
          f"fusion={cfg['fusion']} hyp_fix={cfg['hyp_fix']}) | seed={args.seed}\n" + "="*60)

    model = TopoNRS(topo_feat_dim=6 if cfg['gf'] else 0, hyp_dim=256,
                    use_hyperbolic=cfg['hyp'], hyp_fix=cfg['hyp_fix'],
                    fusion=cfg['fusion'], unfreeze_bert=cfg['unfreeze'],
                    collab_dim=COLLAB_DIM if cfg.get('collab') else 0).to(device)
    r = train_model(run_name, model, MIND_TRAIN, MIND_DEV, args.epochs,
                    args.batch_size, args.lr, topo_reg,
                    validate_every=args.validate_every,
                    ph_features_path=ph_feat_path,
                    neg_sampler=neg_sampler,
                    neg_gamma=args.gamma if cfg['ns'] else 0.0,
                    seed=args.seed, out_dir=out_dir,
                    max_steps=args.max_steps, skip_final_eval=args.smoke)
    print(r)

if __name__ == '__main__':
    main()
