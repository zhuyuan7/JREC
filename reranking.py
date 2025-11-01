#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU kNN + Graph Propagation reranker (no faiss).
- 후보 세트(각 유저 topK) 안에서만 kNN(top-k) 그래프 구성
- 전파 1스텝:  X' = D^{-1/2} A D^{-1/2} X
- A_global: fusion item emb 기반
- A_cross : text emb를 user-context(평균) 방향에 대해 residual 후 구성
- 평가: GT 비어있는 유저는 자동 제외

Usage (예):
python /root/JREC/reranking.py --fusion_item /root/JREC/data/arts/fusion/fusion_coninfo_item_emb.pkl --candidates /root/JREC/data/arts/fusion/coninfo_candidates_top5000.pkl --item_text_emb /root/EasyRec_2/data/arts/text_emb/item_Llama-2-7b-hf.pkl --items_meta /root/EasyRec_2/data/arts/prompts/items_with_meta.jsonl --user_prompt /root/EasyRec_2/data/arts/prompts/user_prompt.json --val_mat /root/EasyRec_2/data/arts/val_mat.pkl --tst_mat /root/EasyRec_2/data/arts/tst_mat.pkl --knn_k 40 --alpha 0.6 --top_eval 200 --mask_history_from_candidates --device cuda --block_size 1024 
"""

import os
import math
import json
import time
import pickle
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np
import torch


# ---------- I/O ----------
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def l2n_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n

def l2n_t(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def stat(name, X: np.ndarray):
    n = np.linalg.norm(X, axis=1)
    print(f"[check] {name:10s} | shape={X.shape} | ||x|| mean={n.mean():.4f} std={n.std():.2e} "
          f"min={n.min():.4f} max={n.max():.4f}")

# ---------- meta / history ----------
def parse_items_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            iid = int(obj["item_id"])
            p = json.loads(obj["prompt"])
            meta[iid] = {
                "title": p.get("title", ""),
                "description": p.get("description", ""),
                "brand": p.get("brand", ""),
                "category": " ".join(p.get("category", [])) if isinstance(p.get("category", []), list)
                            else str(p.get("category", ""))
            }
    return meta

def load_user_histories(path_user_prompt: str) -> Dict[int, List[int]]:
    def _extract_items_from_obj(obj) -> Tuple[int, List[int]]:
        uid_raw = obj.get("user_id", None)
        try:
            uid = int(uid_raw) if uid_raw is not None else None
        except Exception:
            uid = None
        prompt = obj.get("prompt", {})
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except Exception:
                prompt = {}
        purchased = None
        if isinstance(prompt, dict):
            purchased = (prompt.get("PURCHASED ITEMS") or prompt.get("PURCHASED_ITEMS")
                         or prompt.get("purchased_items") or prompt.get("Purchased Items"))
        items: List[int] = []
        if isinstance(purchased, list):
            for it in purchased:
                if isinstance(it, dict) and "item_id" in it:
                    try:
                        items.append(int(it["item_id"]))
                    except Exception:
                        pass
        return uid, items

    with open(path_user_prompt, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    histories: Dict[int, List[int]] = {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        uid, items = _extract_items_from_obj(parsed)
        histories[uid or 0] = items
        return histories
    elif isinstance(parsed, list):
        for obj in parsed:
            if isinstance(obj, dict):
                uid, items = _extract_items_from_obj(obj)
                if uid is not None:
                    histories[uid] = items
        return histories

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        uid, items = _extract_items_from_obj(obj)
        if uid is not None:
            histories[uid] = items
    return histories

# ---------- evaluation ----------
def load_eval_matrix(pkl_path: str) -> List[List[int]]:
    obj = load_pkl(pkl_path)
    if isinstance(obj, dict):
        keys = sorted(obj.keys(), key=lambda x: int(x))
        return [list(map(int, obj[k])) for k in keys]
    try:
        import scipy.sparse as sp  # only for reading
        if sp.issparse(obj):
            obj = obj.tocsr()
            res = []
            for u in range(obj.shape[0]):
                idx = obj.indices[obj.indptr[u]:obj.indptr[u+1]]
                res.append(list(map(int, idx.tolist())))
            return res
    except Exception:
        pass
    if isinstance(obj, list):
        return [list(map(int, x)) for x in obj]
    raise ValueError(f"Unsupported eval pkl format: {type(obj)}")

def eval_recall_ndcg_atk(ground_truth, pred_rank, ks=(5,10,20)):
    gt_sets = [set(g) for g in ground_truth]
    valid_users = [i for i, gt in enumerate(gt_sets) if len(gt) > 0]
    N_valid = len(valid_users)
    if N_valid == 0:
        return {"recall": [0]*len(ks), "ndcg": [0]*len(ks)}

    res = {"recall": [], "ndcg": []}
    for k in ks:
        r_sum = 0.0; n_sum = 0.0
        for u in valid_users:
            topk = pred_rank[u][:k]
            gt = gt_sets[u]
            # recall@k
            hit = sum(1 for x in topk if x in gt)
            r_sum += hit / len(gt)
            # ndcg@k
            dcg = 0.0
            for rank, x in enumerate(topk, start=1):
                if x in gt:
                    dcg += 1.0 / math.log2(rank + 1)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(gt)) + 1))
            n_sum += (dcg / idcg) if idcg > 0 else 0.0
        res["recall"].append(r_sum / N_valid)
        res["ndcg"].append(n_sum / N_valid)
    return res

def candidate_upper_bound(cands: np.ndarray, gt: List[List[int]]):
    gt_sets = [set(g) for g in gt]
    valid, hits, empty = 0, 0, 0
    for u in range(min(len(cands), len(gt_sets))):
        if len(gt_sets[u]) == 0:
            empty += 1
            continue
        valid += 1
        if any(i in gt_sets[u] for i in cands[u]):
            hits += 1
    ratio = (hits / valid) if valid > 0 else 0.0
    return {"hits": hits, "valid_users": valid, "empty_gt_users": empty, "hit_ratio": ratio}

# ---------- GPU kNN (block top-k) ----------
@torch.no_grad()
def block_topk_knn_cosine(emb: torch.Tensor, k: int, block_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    emb: (K, D) L2-normalized
    returns:
      idx: (K, k)  neighbor indices per row (self 포함일 수 있어 제외 처리 필요)
      sim: (K, k)  cosine scores
    메모리 폭발을 피하려고, 행을 block_size씩 나눠 전체와 곱해가며 row-wise topk를 유지.
    """
    device = emb.device
    K = emb.size(0)
    k = min(k, max(1, K - 1))

    idx_all = torch.empty((K, k), dtype=torch.int64, device=device)
    sim_all = torch.empty((K, k), dtype=emb.dtype, device=device)

    # 전체 후보를 한 번에 emb.T로 준비
    emb_t = emb.t().contiguous()

    for start in range(0, K, block_size):
        end = min(start + block_size, K)
        Q = emb[start:end]                   # (B, D)
        sims = Q @ emb_t                     # (B, K)

        # self-connection 제거
        row_idx = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), row_idx] = -1e9

        topv, topi = torch.topk(sims, k=k, dim=1)  # (B, k)
        idx_all[start:end] = topi
        sim_all[start:end] = topv

    return idx_all, sim_all

@torch.no_grad()
def propagate_scores_topk(idx: torch.Tensor, sim: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
    """
    idx: (K, k)
    sim: (K, k)   similarity (>=0 권장, 현재 cosine이라 음수 있을 수 있음 → relu로 양수화 권장)
    base_scores: (K,)  원래 후보 점수 벡터, 정규화된 상태 권장
    returns:
      out_scores: (K,)
    """
    device = base_scores.device
    # 유사도를 양수 가중치로 변환 (음수 억제)
    w = torch.relu(sim) + 1e-8            # (K, k)
    deg = w.sum(dim=1)                    # (K,)
    dinv_sqrt = 1.0 / torch.sqrt(deg)     # (K,)

    # D^{-1/2} A D^{-1/2} x  ~  각 행 i: sum_j ( w_ij / sqrt(d_i d_j) * x_j )
    x = base_scores                       # (K,)
    # j쪽 정규화
    xn = x[idx] * (1.0 / torch.sqrt((w.new_zeros(w.size(0)).index_add_(0, idx.reshape(-1), w.reshape(-1))).clamp_min(1e-8)))[idx]
    # 위 한 줄은 복잡하니 간단/안전한 버전으로 바꿈:
    # j-degree를 직접 계산
    deg_j = torch.zeros_like(base_scores)
    deg_j.index_add_(0, idx.reshape(-1), w.reshape(-1))
    dinv_sqrt_j = 1.0 / torch.sqrt(deg_j.clamp_min(1e-8))
    xn = x[idx] * dinv_sqrt_j[idx]        # (K, k)

    y = (w * xn).sum(dim=1)               # (K,)
    out = dinv_sqrt * y                   # (K,)
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--fusion_item", required=True)
    ap.add_argument("--candidates",  required=True)
    ap.add_argument("--item_text_emb", required=True)
    ap.add_argument("--items_meta",  required=True)
    ap.add_argument("--user_prompt", required=True)
    ap.add_argument("--val_mat", required=True)
    ap.add_argument("--tst_mat", required=True)
    # params
    ap.add_argument("--knn_k", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--top_eval", type=int, default=200)
    ap.add_argument("--mask_history_from_candidates", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block_size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[device] using {device}")

    # load emb
    I_fusion = l2n_np(np.asarray(load_pkl(args.fusion_item), dtype=np.float32))
    item_text = l2n_np(np.asarray(load_pkl(args.item_text_emb), dtype=np.float32))
    cand = load_pkl(args.candidates)  # {"top_index": (N_u,K), "top_score": (N_u,K)}
    _ = parse_items_jsonl(args.items_meta)
    histories = load_user_histories(args.user_prompt)

    stat("item_fused", I_fusion)
    stat("item_text", item_text)

    top_index = cand["top_index"]
    top_score = cand["top_score"]
    assert top_index.shape == top_score.shape
    N_u, K0 = top_index.shape
    print(f"[info] candidates loaded: users={N_u}, topK={K0}, score range=({top_score.min():.4f},{top_score.max():.4f})")
    print(f"[info] items: fused={I_fusion.shape[0]}, text={item_text.shape[0]}")

    # candidate index 범위 체크
    max_cand = top_index.max()
    if max_cand >= I_fusion.shape[0]:
        raise RuntimeError(f"candidate index {max_cand} out of range (items={I_fusion.shape[0]})")

    # GT 로드 + 사용자 수 정렬
    gt_val = load_eval_matrix(args.val_mat)
    gt_tst = load_eval_matrix(args.tst_mat)
    print(f"[info] gt users: val={len(gt_val)}, tst={len(gt_tst)} | non_empty: "
          f"val={sum(1 for g in gt_val if g)}, tst={sum(1 for g in gt_tst if g)}")

    if len(gt_val) != N_u or len(gt_tst) != N_u:
        print(f"[WARN] user count mismatch. candidates={N_u}, val={len(gt_val)}, tst={len(gt_tst)}")
        def align(gt, n): return [gt[u] if u < len(gt) else [] for u in range(n)]
        gt_val = align(gt_val, N_u)
        gt_tst = align(gt_tst, N_u)

    # 상한 히트 비율
    ub_val = candidate_upper_bound(top_index, gt_val)
    ub_tst = candidate_upper_bound(top_index, gt_tst)
    print("\n[Upper-Bound Coverage]")
    print(f"  VAL: hits={ub_val['hits']} / valid={ub_val['valid_users']} (empty={ub_val['empty_gt_users']})"
          f"  -> hit_ratio={ub_val['hit_ratio']*100:.2f}%")
    print(f"  TST: hits={ub_tst['hits']} / valid={ub_tst['valid_users']} (empty={ub_tst['empty_gt_users']})"
          f"  -> hit_ratio={ub_tst['hit_ratio']*100:.2f}%")

    # to torch
    I_fusion_t = torch.from_numpy(I_fusion).to(device)
    item_text_t = torch.from_numpy(item_text).to(device)

    final_ranked: List[List[int]] = []
    K_eval = min(args.top_eval, K0)

    t0 = time.time()
    print(f"\n[rerank-gpu] Start (users={N_u}, knn_k={args.knn_k}, block={args.block_size}) ...")

    for uid in range(N_u):
        cand_ids = top_index[uid].tolist()
        base_scores = top_score[uid].astype(np.float32)

        if args.mask_history_from_candidates:
            hist = set(histories.get(uid, []))
            keep = [i for i, iid in enumerate(cand_ids) if iid not in hist]
            if keep:
                cand_ids = [cand_ids[i] for i in keep]
                base_scores = base_scores[keep]
            if len(cand_ids) == 0:
                final_ranked.append(top_index[uid][:K_eval].tolist())
                continue

        cand_ids_t = torch.tensor(cand_ids, device=device, dtype=torch.long)
        # (K, D)
        emb_cand = l2n_t(I_fusion_t.index_select(0, cand_ids_t))
        txt_cand = l2n_t(item_text_t.index_select(0, cand_ids_t))

        # base score → L2 / L1 정규화(스칼라 벡터 정규화)
        s = torch.tensor(base_scores, device=device, dtype=torch.float32)
        s = s / (torch.norm(s) + 1e-8)

        # A_global (fusion)
        idx_g, sim_g = block_topk_knn_cosine(emb_cand, k=args.knn_k, block_size=args.block_size)
        Xg = propagate_scores_topk(idx_g, sim_g, s)

        # A_cross (text residual)
        hist_items = histories.get(uid, [])
        if len(hist_items) > 0:
            user_ctx = item_text_t.index_select(0, torch.tensor(hist_items, device=device, dtype=torch.long)).mean(dim=0)
            # residualize: x - (x·û)û
            u = user_ctx / (user_ctx.norm() + 1e-8)
            proj = (txt_cand @ u)[:, None] * u[None, :]
            txt_resid = l2n_t(txt_cand - proj)
        else:
            txt_resid = txt_cand

        idx_c, sim_c = block_topk_knn_cosine(txt_resid, k=args.knn_k, block_size=args.block_size)
        Xc = propagate_scores_topk(idx_c, sim_c, s)

        final_score = args.alpha * Xg + (1.0 - args.alpha) * Xc
        order = torch.topk(final_score, k=K_eval, largest=True).indices.tolist()
        reranked = [cand_ids[i] for i in order]
        final_ranked.append(reranked)

        if (uid+1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (uid+1) * (N_u - (uid+1))
            print(f"  processed {uid+1}/{N_u} users | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    # 평가
    ks = (5, 10, 20)
    val_metrics = eval_recall_ndcg_atk(gt_val, final_ranked, ks=ks)
    tst_metrics = eval_recall_ndcg_atk(gt_tst, final_ranked, ks=ks)

    print("\n[files]")
    print(f"  val_mat: {args.val_mat}")
    print(f"  tst_mat: {args.tst_mat}")

    def _print_block(name, metrics):
        print(f"\n[Eval] {name}")
        for k, (r, n) in zip(ks, zip(metrics["recall"], metrics["ndcg"])):
            print(f"  Recall@{k}: {r:.6f} ({r*100:.2f}%) | NDCG@{k}: {n:.6f} ({n*100:.2f}%)")
    _print_block("Validation", val_metrics)
    _print_block("Test",        tst_metrics)

    out_path = "/root/JREC/data/arts/fusion/final_reranked.pkl"
    save_pkl(out_path, final_ranked)
    print(f"\n[rerank-gpu] saved: {out_path}")
    print(f"[rerank-gpu] total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
