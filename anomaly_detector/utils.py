import pandas as pd
import numpy as np
import json
from pathlib import Path
from functools import lru_cache
from scipy.special import softmax
from itertools import chain
from ipaddress import IPv4Network
import pickle

def read_csv_empty(*args, **kwargs):
    try: return pd.read_csv(*args, **kwargs)
    except pd.errors.EmptyDataError: return pd.DataFrame()

def approx_knee_point(x):
    x, y = np.unique(x, return_counts=True)
    _x = (x-x.min())/(x.max()-x.min())
    _y = y.cumsum()/y.sum()
    idx = np.argmax(np.abs(_y-_x))
    return x[idx], _y[idx]

def load_emb_distance(train_dir, return_emb=False):
    train_dir = Path(train_dir)

    node_emb_path = train_dir / "node.emb"
    link_emb_path = train_dir / "link.emb"
    rela_emb_path = train_dir / "rela.emb"

    node_emb = pickle.load(open(node_emb_path, "rb"))
    link_emb = pickle.load(open(link_emb_path, "rb"))
    rela_emb = pickle.load(open(rela_emb_path, "rb"))
    rela = rela_emb["p2c"]
    link = link_emb["p2c"]
    link = softmax(link)

    @lru_cache(maxsize=100000)
    def _emb_distance(a, b): # could be cluster-like, e.g. '{123,456}'
        a = a.strip("{}").split(",")[0]
        b = b.strip("{}").split(",")[0]
        if a == b: return 0.
        if a not in node_emb or b not in node_emb:
            return np.inf
        xi = node_emb[a]
        xj = node_emb[b]
        return np.sum((xj-xi)**2*link) + np.abs(np.sum((xj-xi)*rela))

    def emb_distance(a, b):
        return _emb_distance(str(a), str(b))

    @lru_cache(maxsize=100000)
    def _dtw_distance(s, t):
        s = [v for i,v in enumerate(s) if i == 0 or v != s[i-1]]
        t = [v for i,v in enumerate(t) if i == 0 or v != t[i-1]]
        ls, lt = len(s), len(t)
        DTW = np.full((ls+1, lt+1), np.inf)
        DTW[0,0] = 0.
        for i in range(ls):
            for j in range(lt):
                cost = emb_distance(s[i], t[j])
                DTW[i+1, j+1] = cost + min(DTW[i  , j+1],
                                           DTW[i+1, j  ],
                                           DTW[i  , j  ])
        return DTW[ls, lt]

    def dtw_distance(s, t):
        return _dtw_distance(tuple(s), tuple(t))

    @lru_cache(maxsize=100000)
    def _path_emb_length(s):
        d = np.array([emb_distance(a,b) for a,b in zip(s[:-1], s[1:])])
        d = d[(d > 0) & (d < np.inf)]
        return np.nan if d.size == 0 else d.sum()

    def path_emb_length(s):
        return _path_emb_length(tuple(s))

    if return_emb:
        return emb_distance, dtw_distance, path_emb_length, node_emb, link, rela

    return emb_distance, dtw_distance, path_emb_length

def root_cause_localize_2set(df, th=0.95):
    set1_asn_cnt, set2_asn_cnt = {}, {}
    for i,j in df[["path1", "path2"]].values:
        set_i = set(i.split(" "))
        set_j = set(j.split(" "))
        set_ij = set_i - set_j
        set_ji = set_j - set_i
        for asn in set_ij:
            if asn not in set1_asn_cnt: set1_asn_cnt[asn] = 1
            else: set1_asn_cnt[asn] += 1
        for asn in set_ji:
            if asn not in set2_asn_cnt: set2_asn_cnt[asn] = 1
            else: set2_asn_cnt[asn] += 1

    set1, cnt1 = list(set1_asn_cnt.keys()), list(set1_asn_cnt.values())
    idx1 = np.argsort(cnt1)[::-1]
    set1 = np.array(set1)[idx1]
    cnt1 = np.array(cnt1)[idx1]

    set2, cnt2 = list(set2_asn_cnt.keys()), list(set2_asn_cnt.values())
    idx2 = np.argsort(cnt2)[::-1]
    set2 = np.array(set2)[idx2]
    cnt2 = np.array(cnt2)[idx2]
   
    rc_1, rc_2 = [], []
    for a,b in zip(set1, cnt1):
        if b/df.shape[0] > th: rc_1.append(a)
    for a,b in zip(set2, cnt2):
        if b/df.shape[0] > th: rc_2.append(a)

    return sorted(rc_1), sorted(rc_2)

def root_cause_localize_1set(df, th=0.95):
    set_asn_cnt = {}
    for i,j in df[["path1", "path2"]].values:
        set_i = set(i.split(" "))
        set_j = set(j.split(" "))
        set_xor = set_i^set_j
        for asn in set_xor:
            if asn not in set_asn_cnt: set_asn_cnt[asn] = 1
            else: set_asn_cnt[asn] += 1

    set_asn, cnt = list(set_asn_cnt.keys()), list(set_asn_cnt.values())
    idx = np.argsort(cnt)[::-1]
    set_asn = np.array(set_asn)[idx]
    cnt = np.array(cnt)[idx]

    rc = []
    for a,b in zip(set_asn, cnt):
        if b/df.shape[0] > th: rc.append(a)

    return sorted(rc)


def link_root_cause(culprit_to_df):
    rcs = list(culprit_to_df.keys())
    dfs = list(culprit_to_df.values())

    def rc_to_set(rc):
        culprit_type, culprit_tuple = rc
        assert culprit_type in ["Prefix", "AS"]
        if culprit_type == "AS":
            culprit_set = set(chain(*culprit_tuple))
        else: # must be "Prefix"
            culprit_set = {IPv4Network(p) for p in culprit_tuple}
        return culprit_type, culprit_set

    def rc_set_related(rc1, rc2):
        t1, set1 = rc1
        t2, set2 = rc2
        if t1 != t2:
            return False
        if t1 == "AS":
            return set1&set2
        else: # t1 and t2 must be "Prefix"
            for i in set1:
                for j in set2:
                    if i.overlaps(j): # check if they overlap
                        return True
                    if i.prefixlen == j.prefixlen: # check if they're two consecutive prefixes
                        return abs((int(i[0])>>(32-i.prefixlen))
                                -(int(j[0])>>(32-j.prefixlen))) <= 1
            return False

    pool = list(map(rc_to_set, rcs))
    group_id = [-1]*len(culprit_to_df)
    id_group = dict()
    next_id = 0
    for i in range(len(culprit_to_df)):
        if group_id[i] == -1: 
            group_id[i] = next_id
            next_id += 1
            id_group[group_id[i]] = [i]
        for j in range(i+1, len(culprit_to_df)):
            if group_id[j] == group_id[i]: continue
            if rc_set_related(pool[i], pool[j]):
                if group_id[j] == -1:
                    group_id[j] = group_id[i]
                    id_group[group_id[i]].append(j)
                else:
                    to_be_merged = id_group.pop(group_id[j])
                    id_group[group_id[i]] += to_be_merged
                    for k in to_be_merged: group_id[k] = group_id[i]
    group_id_set = set(group_id)
    group_id_remapping = dict(zip(group_id_set, range(len(group_id_set))))
    for idx, df in enumerate(dfs):
        df["group_id"] = group_id_remapping[group_id[idx]]
    return id_group, pd.concat(dfs, ignore_index=True)

def event_aggregate(events):
    culprit2eventkey = {}
    eventkey2culprit = {}

    for k,v in events.items():
        rc_1, rc_2 = root_cause_localize_2set(v)
        rc_3 = root_cause_localize_1set(v)
        if rc_1 or rc_2:
            culprit = "AS", (tuple(rc_1), tuple(rc_2))
        elif rc_3:
            culprit = "AS", (tuple(rc_3),)
        else:
            culprit = "Prefix", k
        culprit2eventkey.setdefault(culprit, set()).add(k)
        eventkey2culprit[k] = culprit

    culprit_to_df = {k: pd.concat([events[i] for i in v])
                                        for k, v in culprit2eventkey.items()}
    for k, v in culprit_to_df.items():
        _, culprit_tuple = k
        v["culprit"] = json.dumps(culprit_tuple)
    rc_groups, df = link_root_cause(culprit_to_df)

    return rc_groups, df
