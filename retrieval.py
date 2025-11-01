# # retrieval.py
# import os, json, pickle, argparse
# import numpy as np

# def load_pkl(p):
#     with open(p, "rb") as f:
#         return pickle.load(f)

# def save_pkl(p, obj):
#     os.makedirs(os.path.dirname(p), exist_ok=True)
#     with open(p, "wb") as f:
#         pickle.dump(obj, f)

# def l2n(x):
#     n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
#     return x / n

# def stat(name, X):
#     n = np.linalg.norm(X, axis=1)
#     print(f"[check] {name:9s} | shape={X.shape} | ||x|| mean={n.mean():.4f} std={n.std():.4e} min={n.min():.4f} max={n.max():.4f}")

# def cosine_topk(users, items, topk=7000, batch=1024):
#     users = l2n(users.astype(np.float32))
#     items = l2n(items.astype(np.float32))
#     n_u, n_i = users.shape[0], items.shape[0]
#     all_idx = np.empty((n_u, topk), dtype=np.int32)
#     all_sc  = np.empty((n_u, topk), dtype=np.float32)
#     for st in range(0, n_u, batch):
#         ed = min(st+batch, n_u)
#         sims = users[st:ed] @ items.T
#         idx  = np.argpartition(-sims, kth=topk-1, axis=1)[:, :topk]
#         part = np.take_along_axis(sims, idx, axis=1)
#         order = np.argsort(-part, axis=1)
#         all_idx[st:ed] = np.take_along_axis(idx, order, axis=1)
#         all_sc[st:ed]  = np.take_along_axis(part, order, axis=1)
#     return all_idx, all_sc

# def users_in_val(p):
#     obj = load_pkl(p)
#     if isinstance(obj, dict):
#         return len(obj)
#     try:
#         import scipy.sparse as sp
#         if sp.issparse(obj): return obj.shape[0]
#     except Exception:
#         pass
#     if isinstance(obj, list): return len(obj)
#     return None

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--fusion_user", default="/root/JREC/data/arts/fusion/fusion_coninfo_user_emb.pkl")
#     ap.add_argument("--fusion_item", default="/root/JREC/data/arts/fusion/fusion_coninfo_item_emb.pkl")
#     ap.add_argument("--val_mat",     default="/root/EasyRec/data/arts/val_mat.pkl")
#     ap.add_argument("--topk", type=int, default=5000)
#     ap.add_argument("--save_candidates", default="/root/JREC/data/arts/fusion/coninfo_candidates_top5000.pkl")
#     args = ap.parse_args()

#     U = np.asarray(load_pkl(args.fusion_user), dtype=np.float32)
#     I = np.asarray(load_pkl(args.fusion_item), dtype=np.float32)
#     print("[info] embeddings loaded")
#     stat("user_fused", U)
#     stat("item_fused", I)

#     n_users_val = users_in_val(args.val_mat)
#     print(f"[info] users in val_mat = {n_users_val}")

#     # 전형적 실수: user/item 파일이 뒤바뀜
#     if n_users_val is not None and U.shape[0] != n_users_val and I.shape[0] == n_users_val:
#         print("[WARN] fusion_user / fusion_item 행수가 뒤바뀐 것으로 보입니다. 자동 스왑합니다.")
#         U, I = I, U
#         stat("[fix] user_fused", U)
#         stat("[fix] item_fused", I)

#     if U.shape[0] < 1000 and I.shape[0] > U.shape[0]:
#         print("[WARN] 사용자 수가 비정상적으로 작습니다. 데이터 매핑/경로 확인 필요.")

#     top_idx, top_sc = cosine_topk(U, I, topk=args.topk)
#     print(f"[ok] cosine_topk done: top_idx.shape={top_idx.shape}, scores range=({top_sc.min():.4f}, {top_sc.max():.4f})")

#     # index 범위 검증
#     max_idx = top_idx.max()
#     if max_idx >= I.shape[0]:
#         print(f"[ERROR] candidate index {max_idx} out of range for items {I.shape[0]}")
#         raise SystemExit(1)

#     out = {"top_index": top_idx, "top_score": top_sc}
#     save_pkl(args.save_candidates, out)
#     print(f"[retrieval] saved: {args.save_candidates}")

# if __name__ == "__main__":
#     main()



# /root/JREC/retrieval.py
# GPU-accelerated cosine topK 후보 생성기
# - L2 정규화 후 users @ items^T
# - block-wise topK로 메모리 절약
# - user/item 파일 스왑 자동 감지
# - (선택) train/val에서 본 아이템 마스킹
# - 저장: {"top_index": (U, K), "top_score": (U, K)}

import os, json, pickle, argparse
from typing import List, Set, Dict, Tuple, Optional

import numpy as np
import torch

# ============ I/O ============
def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def save_pkl(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)

def stat(name, X: np.ndarray):
    n = np.linalg.norm(X, axis=1)
    print(f"[check] {name:9s} | shape={X.shape} | ||x|| mean={n.mean():.4f} std={n.std():.4e} "
          f"min={n.min():.4f} max={n.max():.4f}")

# ============ Eval utils ============
def users_len_from_mat(p) -> Optional[int]:
    try:
        obj = load_pkl(p)
    except Exception:
        return None
    if isinstance(obj, dict):
        return len(obj)
    try:
        import scipy.sparse as sp
        if sp.issparse(obj):
            return obj.shape[0]
    except Exception:
        pass
    if isinstance(obj, list):
        return len(obj)
    return None

def mat_to_poslists(p, n_users: int, n_items: int) -> List[Set[int]]:
    """CSR/dict/list → list[set]"""
    pos = [set() for _ in range(n_users)]
    if not p:
        return pos
    obj = load_pkl(p)
    try:
        import scipy.sparse as sp
        if sp.issparse(obj):
            csr = obj.tocsr()
            nu = min(n_users, csr.shape[0])
            for u in range(nu):
                idx = csr.indices[csr.indptr[u]:csr.indptr[u+1]]
                pos[u].update(int(i) for i in idx if 0 <= int(i) < n_items)
            return pos
    except Exception:
        pass
    if isinstance(obj, dict):
        for k, v in obj.items():
            u = int(k)
            if 0 <= u < n_users:
                for i in v:
                    ii = int(i)
                    if 0 <= ii < n_items:
                        pos[u].add(ii)
        return pos
    if isinstance(obj, list):
        nu = min(n_users, len(obj))
        for u in range(nu):
            for i in obj[u]:
                ii = int(i)
                if 0 <= ii < n_items:
                    pos[u].add(ii)
        return pos
    return pos

# ============ Core (GPU block topK) ============
@torch.no_grad()
def cosine_topk_block(users: np.ndarray,
                      items: np.ndarray,
                      topk: int = 5000,
                      block: int = 1024,
                      device: str = "cpu",
                      use_half: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    users: (U, D), items: (I, D)  — both will be L2-normalized inside.
    Returns: indices (U, K), scores (U, K) in desc order.
    """
    assert users.ndim == 2 and items.ndim == 2
    U, D = users.shape
    I, D2 = items.shape
    assert D == D2, "dim mismatch"

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    dtype = torch.float16 if (use_half and dev.type == "cuda") else torch.float32

    u = torch.from_numpy(users).to(dev, dtype=dtype)
    v = torch.from_numpy(items).to(dev, dtype=dtype)

    # L2 norm
    u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    topk = min(topk, I)
    all_idx = torch.empty((U, topk), dtype=torch.int32, device="cpu")
    all_sc  = torch.empty((U, topk), dtype=torch.float32, device="cpu")

    vT = v.t().contiguous()  # (D, I)

    for st in range(0, U, block):
        ed = min(st + block, U)
        sims = (u[st:ed] @ vT)  # (B, I)
        # torch.topk가 전체에서 바로 topk를 뽑음
        sc, ix = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)  # (B, K)
        all_idx[st:ed] = ix.to(torch.int32).cpu()
        all_sc[st:ed]  = sc.to(torch.float32).cpu()

    return all_idx.numpy(), all_sc.numpy()

# ============ Post-filter (mask seen) ============
def apply_mask_seen(top_idx: np.ndarray,
                    top_sc: np.ndarray,
                    seen_lists: List[Set[int]],
                    replace_with: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 유저별 seen 아이템을 제거. 간단히 score를 -inf로 주고 재정렬하지 않음(이미 내림차순).
    이후 상위 K를 그대로 사용하면 seen이 밀려나 효과.
    """
    U, K = top_idx.shape
    NEG_INF = np.float32(-1e9)
    for u in range(min(U, len(seen_lists))):
        seen = seen_lists[u]
        if not seen:
            continue
        mask = np.isin(top_idx[u], list(seen))
        top_sc[u][mask] = NEG_INF
    # 점수 기준 재정렬(동일 index 배열에서 점수만 수정 → 재정렬 필요)
    order = np.argsort(-top_sc, axis=1)
    row = np.arange(U)[:, None]
    top_idx = top_idx[row, order]
    top_sc  = top_sc[row, order]
    return top_idx, top_sc

# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_user", default="/root/EasyRec_2/data/arts/cf_emb/user_simgcl.pkl")
    ap.add_argument("--fusion_item", default="/root/EasyRec_2/data/arts/cf_emb/item_simgcl.pkl")
    ap.add_argument("--val_mat",     default="/root/EasyRec_2/data/arts/val_mat.pkl")
    ap.add_argument("--train_mat",   default="/root/EasyRec_2/data/arts/trn_mat.pkl", help="(선택) train interactions for seen-mask")
    ap.add_argument("--also_mask_val", action="store_true", help="val_mat도 seen-mask에 포함")
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--save_candidates", default="/root/JREC/data/arts/fusion/simgcl_candidates_top5000.pkl")
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda"])
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--no_half", action="store_true", help="FP16 비활성화")
    args = ap.parse_args()

    U = np.asarray(load_pkl(args.fusion_user), dtype=np.float32)
    I = np.asarray(load_pkl(args.fusion_item), dtype=np.float32)
    print("[info] embeddings loaded")
    stat("user_fused", U)
    stat("item_fused", I)

    n_users_val = users_len_from_mat(args.val_mat)
    print(f"[info] users in val_mat = {n_users_val}")

    # 스왑 감지
    if n_users_val is not None and U.shape[0] != n_users_val and I.shape[0] == n_users_val:
        print("[WARN] fusion_user / fusion_item 행수가 뒤바뀐 것으로 보입니다. 자동 스왑합니다.")
        U, I = I, U
        stat("[fix] user_fused", U)
        stat("[fix] item_fused", I)

    if U.shape[0] < 1000 and I.shape[0] > U.shape[0]:
        print("[WARN] 사용자 수가 비정상적으로 작습니다. 데이터 매핑/경로 확인 필요.")

    # GPU block topK
    print(f"[run] cosine_topk (device={args.device}, block={args.block_size}, half={not args.no_half})")
    top_idx, top_sc = cosine_topk_block(
        users=U, items=I, topk=args.topk,
        block=args.block_size, device=args.device, use_half=not args.no_half
    )
    print(f"[ok] cosine_topk done: top_idx.shape={top_idx.shape}, scores range=({top_sc.min():.4f}, {top_sc.max():.4f})")

    # 인덱스 범위 검증
    max_idx = top_idx.max()
    if max_idx >= I.shape[0]:
        print(f"[ERROR] candidate index {max_idx} out of range for items {I.shape[0]}")
        raise SystemExit(1)

    # (선택) seen-mask (train(+val)에서 본 아이템 제거)
    if args.train_mat or args.also_mask_val:
        print("[mask] build seen lists ...")
        seen = mat_to_poslists(args.train_mat or "", n_users=U.shape[0], n_items=I.shape[0])
        if args.also_mask_val and args.val_mat:
            seen_val = mat_to_poslists(args.val_mat, n_users=U.shape[0], n_items=I.shape[0])
            for u in range(min(len(seen), len(seen_val))):
                seen[u].update(seen_val[u])
        top_idx, top_sc = apply_mask_seen(top_idx, top_sc, seen)
        print("[mask] apply done")

    out = {"top_index": top_idx, "top_score": top_sc}
    save_pkl(args.save_candidates, out)
    print(f"[retrieval] saved: {args.save_candidates}")

if __name__ == "__main__":
    main()
