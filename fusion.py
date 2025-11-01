# /root/JREC/fusion.py
# CF/Text 임베딩 결합 + (선택) BPR 공동학습
# - 방법: whiten_concat | concat_mlp | two_tower
# - 정렬(InfoNCE/MSE) + BPR(u,i+,i-)를 결합하여 사용자/아이템 fused 임베딩을 학습
# - 사용자/아이템 개수 뒤바뀜 자동 감지(--auto_fix_mismatch)




'''
python /root/JREC/fusion.py \
  --method concat_mlp \
  --user_cf  /root/EasyRec_2/data/arts/cf_emb/user_simgcl.pkl \
  --user_txt /root/EasyRec_2/data/arts/text_emb/user_Llama-2-7b-hf.pkl \
  --item_cf  /root/EasyRec_2/data/arts/cf_emb/item_simgcl.pkl \
  --item_txt /root/EasyRec_2/data/arts/text_emb/item_Llama-2-7b-hf.pkl \
  --train_mat /root/EasyRec_2/data/arts/trn_mat.pkl \
  --bpr_lambda 0.5 --bpr_neg 1 --bpr_steps 300 \
  --epochs 50 --batch_size 512 --device cuda --use_amp \
  --save_user /root/JREC/data/arts/fusion/fusion_lla_sim_con_tower_user_emb.pkl \
  --save_item /root/JREC/data/arts/fusion/fusion_lla_sim_con_tower_item_emb.pkl
    
'''


import os
import math
import json
import pickle
import argparse
from typing import Tuple, List, Set, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- I/O ----------------
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def as_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def l2norm_t(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def l2norm_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / n).astype(np.float32)

def log_stats(name: str, arr: np.ndarray):
    print(f"[info] {name:>10s} | shape={tuple(arr.shape)} | mean={arr.mean():.4f} | std={arr.std():.4f}")

# --------------- Preproc ---------------
def zca_whitening(X: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    U, S, _ = np.linalg.svd(cov + eps * np.eye(cov.shape[0], dtype=cov.dtype), full_matrices=False)
    Xw = (Xc @ U) / np.sqrt(S + eps)
    return Xw.astype(np.float32), U.astype(np.float32), S.astype(np.float32)

# --------------- Models ---------------
class ConcatMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        z = self.net(x)
        return l2norm_t(z, -1)

class TwoTowerAlign(nn.Module):
    """
    CF / TEXT 각각 Linear로 같은 공간으로 사상 → concat+fc → fused
    또한 CF↔TEXT InfoNCE 정렬을 기본 제공
    """
    def __init__(self, cf_dim: int, txt_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.cf_proj = nn.Linear(cf_dim, out_dim)
        self.txt_proj = nn.Linear(txt_dim, out_dim)
        self.fuse = nn.Linear(out_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_pair(self, cf, txt):
        cf_z = l2norm_t(self.cf_proj(cf), -1)
        txt_z = l2norm_t(self.txt_proj(txt), -1)
        fused = torch.cat([cf_z, txt_z], dim=-1)
        fused = self.fuse(self.dropout(fused))
        fused = l2norm_t(fused, -1)
        return cf_z, txt_z, fused

    def forward(self, cf, txt):
        _, _, fused = self.encode_pair(cf, txt)
        return fused

    @staticmethod
    def info_nce_loss(a, p, temperature: float = 0.07):
        a = F.normalize(a, dim=-1); p = F.normalize(p, dim=-1)
        sim = (a @ p.t()) / temperature
        labels = torch.arange(a.size(0), device=a.device)
        # symmetric
        return F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)

def build_fuser(method: str,
                user_cf_dim: int, user_txt_dim: int,
                item_cf_dim: int, item_txt_dim: int,
                out_dim: int,
                hidden: int = 1024, dropout: float = 0.1):
    if method == "concat_mlp":
        user_model = ConcatMLP(user_cf_dim + user_txt_dim, out_dim, hidden, dropout)
        item_model = ConcatMLP(item_cf_dim + item_txt_dim, out_dim, hidden, dropout)
        return user_model, item_model
    elif method == "two_tower":
        user_model = TwoTowerAlign(user_cf_dim, user_txt_dim, out_dim, dropout=dropout)
        item_model = TwoTowerAlign(item_cf_dim, item_txt_dim, out_dim, dropout=dropout)
        return user_model, item_model
    else:
        raise ValueError(f"Unknown method: {method}")

# --------------- BPR Utils ---------------
def load_train_interactions(path: str, num_users: int, num_items: int):
    """
    return: pos_lists (list[set])  길이=num_users, 각 유저의 positive 아이템 집합
    지원 포맷: scipy.sparse CSR / dict[u->list[i]] / list[list[i]]
    """
    if not path:
        return None
    obj = load_pkl(path)
    pos_lists: List[Set[int]] = [set() for _ in range(num_users)]
    try:
        import scipy.sparse as sp
        if sp.issparse(obj):
            obj = obj.tocsr()
            for u in range(min(num_users, obj.shape[0])):
                idx = obj.indices[obj.indptr[u]:obj.indptr[u+1]]
                pos_lists[u].update(int(i) for i in idx if 0 <= int(i) < num_items)
            return pos_lists
    except Exception:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            u = int(k)
            if 0 <= u < num_users:
                for i in v:
                    ii = int(i)
                    if 0 <= ii < num_items:
                        pos_lists[u].add(ii)
        return pos_lists
    if isinstance(obj, list):
        for u, v in enumerate(obj[:num_users]):
            for i in v:
                ii = int(i)
                if 0 <= ii < num_items:
                    pos_lists[u].add(ii)
        return pos_lists

    print(f"[WARN] unsupported train_mat type: {type(obj)}")
    return None

def sample_bpr_triplets(pos_lists: List[Set[int]], num_items: int, batch_users: np.ndarray,
                        neg_per_pos: int = 1, device: str = "cpu"):
    """
    유저 배치에 대해 (u, i_pos, i_neg) 텐서 생성. 비어있는 유저는 제외
    """
    u_idx = []
    i_pos = []
    i_neg = []
    rng = np.random.default_rng()

    for u in batch_users:
        positives = list(pos_lists[u])
        if not positives:
            continue
        ip = positives[rng.integers(0, len(positives))]
        for _ in range(neg_per_pos):
            j = rng.integers(0, num_items)
            tries = 0
            while (j in pos_lists[u]) and tries < 16:
                j = rng.integers(0, num_items); tries += 1
            u_idx.append(u); i_pos.append(ip); i_neg.append(int(j))

    if len(u_idx) == 0:
        return None
    dev = torch.device(device)
    return (torch.tensor(u_idx, dtype=torch.long, device=dev),
            torch.tensor(i_pos, dtype=torch.long, device=dev),
            torch.tensor(i_neg, dtype=torch.long, device=dev))

def bpr_loss(u_f: torch.Tensor, i_pos_f: torch.Tensor, i_neg_f: torch.Tensor, margin: float = 0.0):
    s_pos = (u_f * i_pos_f).sum(-1)
    s_neg = (u_f * i_neg_f).sum(-1)
    return -torch.log(torch.sigmoid(s_pos - s_neg - margin)).mean()

# --------------- Two-tower trainer (with optional BPR) ---------------
def train_two_tower(user_model: TwoTowerAlign, item_model: TwoTowerAlign,
                    u_cf, u_txt, i_cf, i_txt,
                    epochs: int = 3, lr: float = 1e-3, bs: int = 256, device: str = "cuda",
                    temperature: float = 0.07, mse_weight: float = 1.0, infonce_weight: float = 1.0,
                    use_amp: bool = True,
                    # BPR 옵션
                    bpr_lambda: float = 0.0, bpr_steps: int = 200, bpr_neg: int = 1,
                    pos_lists: Optional[List[Set[int]]] = None):
    user_model.to(device)
    item_model.to(device)
    opt = torch.optim.AdamW(list(user_model.parameters()) + list(item_model.parameters()), lr=lr)

    u_cf_t = torch.from_numpy(u_cf).to(device)
    u_txt_t = torch.from_numpy(u_txt).to(device)
    i_cf_t = torch.from_numpy(i_cf).to(device)
    i_txt_t = torch.from_numpy(i_txt).to(device)

    num_u = u_cf.shape[0]
    num_i = i_cf.shape[0]

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and str(device).startswith("cuda")))

    for ep in range(epochs):
        user_model.train(); item_model.train()
        print(f"[two_tower] epoch {ep+1}/{epochs}")

        # ---- users: InfoNCE + MSE alignment
        perm_u = torch.randperm(num_u, device=device)
        for st in tqdm(range(0, num_u, bs), desc=" users  ", ncols=100):
            idx = perm_u[st:st+bs]
            with torch.cuda.amp.autocast(enabled=(use_amp and str(device).startswith("cuda"))):
                u_cf_z, u_txt_z, u_fz = user_model.encode_pair(u_cf_t[idx], u_txt_t[idx])
                loss_u = (infonce_weight * TwoTowerAlign.info_nce_loss(u_cf_z, u_txt_z, temperature=temperature)
                          + mse_weight * (F.mse_loss(u_fz, u_cf_z.detach()) + F.mse_loss(u_fz, u_txt_z.detach())))
            opt.zero_grad(); scaler.scale(loss_u).backward(); scaler.step(opt); scaler.update()

        # ---- items: InfoNCE + MSE alignment
        perm_i = torch.randperm(num_i, device=device)
        for st in tqdm(range(0, num_i, bs), desc=" items  ", ncols=100):
            idx = perm_i[st:st+bs]
            with torch.cuda.amp.autocast(enabled=(use_amp and str(device).startswith("cuda"))):
                i_cf_z, i_txt_z, i_fz = item_model.encode_pair(i_cf_t[idx], i_txt_t[idx])
                loss_i = (infonce_weight * TwoTowerAlign.info_nce_loss(i_cf_z, i_txt_z, temperature=temperature)
                          + mse_weight * (F.mse_loss(i_fz, i_cf_z.detach()) + F.mse_loss(i_fz, i_txt_z.detach())))
            opt.zero_grad(); scaler.scale(loss_i).backward(); scaler.step(opt); scaler.update()

        # ---- (선택) BPR joint 학습
        if bpr_lambda > 0.0 and pos_lists is not None:
            print(f"[BPR] epoch {ep+1}: steps={bpr_steps}, neg={bpr_neg}, lambda={bpr_lambda}")
            for _ in range(bpr_steps):
                batch_users = np.random.randint(0, num_u, size=(bs,))
                trip = sample_bpr_triplets(pos_lists, num_i, batch_users,
                                           neg_per_pos=bpr_neg, device=device)
                if trip is None:
                    continue
                u_b, ip_b, in_b = trip
                with torch.cuda.amp.autocast(enabled=(use_amp and str(device).startswith("cuda"))):
                    # fused 임베딩으로 직접 BPR
                    _, _, u_f   = user_model.encode_pair(u_cf_t[u_b], u_txt_t[u_b])
                    _, _, i_pos = item_model.encode_pair(i_cf_t[ip_b], i_txt_t[ip_b])
                    _, _, i_neg = item_model.encode_pair(i_cf_t[in_b], i_txt_t[in_b])
                    loss_bpr = bpr_loss(u_f, i_pos, i_neg)
                    loss = bpr_lambda * loss_bpr
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    user_model.eval(); item_model.eval()
    with torch.no_grad():
        _, _, u_fused = user_model.encode_pair(u_cf_t, u_txt_t)
        _, _, i_fused = item_model.encode_pair(i_cf_t, i_txt_t)
    return u_fused.cpu().numpy().astype(np.float32), i_fused.cpu().numpy().astype(np.float32)

# --------------- Utils ---------------
def fix_if_swapped(ucf, utx, icf, itx, auto_fix: bool):
    """
    user 수가 item 수보다 작으면, user/item이 뒤바뀐 것으로 간주하고 자동 스왑(옵션)
    """
    nu, ni = ucf.shape[0], icf.shape[0]
    swapped = False
    if nu < ni:
        print(f"[WARN] #user({nu}) < #item({ni}) → 파일 스왑 의심")
        if auto_fix:
            print("[INFO] auto-fix ON → user/item 임베딩을 서로 교환합니다.")
            ucf, icf = icf, ucf
            utx, itx = itx, utx
            swapped = True
        else:
            print("[HINT] --auto_fix_mismatch 를 주면 자동으로 교환합니다.")
    return ucf, utx, icf, itx, swapped

def set_seed(seed: int = 1958):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------- Main ---------------
def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--user_cf",  default="/root/EasyRec/data/arts/cf_emb/user_simgcl.pkl")
    p.add_argument("--user_txt", default="/root/EasyRec/data/arts/text_emb/user_Llama-2-7b-hf.pkl")
    p.add_argument("--item_cf",  default="/root/EasyRec/data/arts/cf_emb/item_simgcl.pkl")
    p.add_argument("--item_txt", default="/root/EasyRec/data/arts/text_emb/item_Llama-2-7b-hf.pkl")
    # method
    p.add_argument("--method", choices=["concat_mlp", "two_tower", "whiten_concat"], default="concat_mlp")
    p.add_argument("--out_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_amp", action="store_true", help="mixed precision(amp) 사용")
    # concat_mlp에서 InfoNCE 사용 여부/하이퍼
    p.add_argument("--use_infonce_concat", action="store_true", help="concat_mlp 경로에서도 CF↔TEXT InfoNCE 사용")
    p.add_argument("--infonce_lambda", type=float, default=0.5, help="concat_mlp InfoNCE 가중치 (0.3~0.7 권장)")
    p.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    # BPR
    p.add_argument("--train_mat", type=str, default="", help="train interactions pkl (CSR/dict/list)")
    p.add_argument("--bpr_lambda", type=float, default=0.0, help=">0이면 BPR joint 학습")
    p.add_argument("--bpr_neg", type=int, default=1)
    p.add_argument("--bpr_steps", type=int, default=200)  # epoch당 triplet 스텝 수
    # save
    p.add_argument("--save_user", default="/root/JREC/data/arts/fusion/fusion_user_emb.pkl")
    p.add_argument("--save_item", default="/root/JREC/data/arts/fusion/fusion_item_emb.pkl")
    # options
    p.add_argument("--seed", type=int, default=1958)
    p.add_argument("--auto_fix_mismatch", action="store_true", help="user/item count 뒤바뀜 자동 교정")
    p.add_argument("--no_l2norm", action="store_true", help="최종 임베딩 L2 정규화 생략")
    args = p.parse_args()

    set_seed(args.seed)

    # ====== 로드 및 요약 ======
    ucf = as_numpy(load_pkl(args.user_cf))
    utx = as_numpy(load_pkl(args.user_txt))
    icf = as_numpy(load_pkl(args.item_cf))
    itx = as_numpy(load_pkl(args.item_txt))

    print("\n[check] ---- embedding file summary ----")
    log_stats("user_cf", ucf)
    log_stats("user_txt", utx)
    log_stats("item_cf", icf)
    log_stats("item_txt", itx)

    # ====== 개수/차원 검사 & 자동 스왑 ======
    if ucf.shape[0] != utx.shape[0]:
        print(f"[WARN] user_cf({ucf.shape[0]}) != user_txt({utx.shape[0]}) → 사용자 임베딩 불일치!")
    if icf.shape[0] != itx.shape[0]:
        print(f"[WARN] item_cf({icf.shape[0]}) != item_txt({itx.shape[0]}) → 아이템 임베딩 불일치!")

    ucf, utx, icf, itx, swapped = fix_if_swapped(ucf, utx, icf, itx, args.auto_fix_mismatch)
    if swapped:
        print("[INFO] 스왑 적용 후:")
        log_stats("user_cf", ucf); log_stats("item_cf", icf)

    # ====== 방법별 결합 ======
    device = args.device

    if args.method == "whiten_concat":
        print("\n[method] whiten_concat")
        ucf_w, _, _ = zca_whitening(ucf)
        utx_w, _, _ = zca_whitening(utx)
        icf_w, _, _ = zca_whitening(icf)
        itx_w, _, _ = zca_whitening(itx)

        u_fused = np.concatenate([ucf_w, utx_w], axis=1)
        i_fused = np.concatenate([icf_w, itx_w], axis=1)

        if u_fused.shape[1] != args.out_dim:
            rng = np.random.RandomState(args.seed)
            Wu = rng.normal(0, 1.0 / math.sqrt(u_fused.shape[1]),
                            size=(u_fused.shape[1], args.out_dim)).astype(np.float32)
            Wi = rng.normal(0, 1.0 / math.sqrt(i_fused.shape[1]),
                            size=(i_fused.shape[1], args.out_dim)).astype(np.float32)
            u_fused = (u_fused @ Wu)
            i_fused = (i_fused @ Wi)

    elif args.method == "concat_mlp":
        print("\n[method] concat_mlp")
        user_model, item_model = build_fuser(
            "concat_mlp", ucf.shape[1], utx.shape[1], icf.shape[1], itx.shape[1],  # shape =(14470, 32), shape = (14470, 4096), shape = (8537, 32), shape = (8537, 4096)
            args.out_dim, args.hidden, args.dropout
        )
        user_model.to(device); item_model.to(device)
        opt = torch.optim.AdamW(list(user_model.parameters()) + list(item_model.parameters()), lr=args.lr)

        u_in = torch.from_numpy(np.concatenate([ucf, utx], axis=1)).to(device)
        i_in = torch.from_numpy(np.concatenate([icf, itx], axis=1)).to(device)

        # 정렬 타깃 생성을 위한 선형 투영
        proj_u_cf = nn.Linear(ucf.shape[1], args.out_dim).to(device)
        proj_u_tx = nn.Linear(utx.shape[1], args.out_dim).to(device)
        proj_i_cf = nn.Linear(icf.shape[1], args.out_dim).to(device)
        proj_i_tx = nn.Linear(itx.shape[1], args.out_dim).to(device)
        opt.add_param_group({"params": proj_u_cf.parameters()})
        opt.add_param_group({"params": proj_u_tx.parameters()})
        opt.add_param_group({"params": proj_i_cf.parameters()})
        opt.add_param_group({"params": proj_i_tx.parameters()})

        def info_nce(a, p, t=0.07):
            a = F.normalize(a, dim=-1); p = F.normalize(p, dim=-1)
            logits = (a @ p.t()) / t
            y = torch.arange(a.size(0), device=a.device)
            return F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y)

        # (선택) BPR용 상호작용 로드
        pos_lists = None
        if args.bpr_lambda > 0.0 and args.train_mat:
            pos_lists = load_train_interactions(args.train_mat, u_in.size(0), i_in.size(0))
            if pos_lists is None:
                print("[WARN] train_mat을 읽지 못해 BPR을 건너뜁니다.")
        
        bs = args.batch_size
        scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and str(device).startswith("cuda")))
        for ep in range(args.epochs):
            print(f"[concat_mlp] epoch {ep+1}/{args.epochs}")
            perm_u = torch.randperm(u_in.size(0), device=device)
            perm_i = torch.randperm(i_in.size(0), device=device)

            user_model.train(); item_model.train()

            # ---- users (정렬)
            for st in tqdm(range(0, u_in.size(0), bs), desc=" users  ", ncols=100):
                idx = perm_u[st:st+bs]
                with torch.cuda.amp.autocast(enabled=(args.use_amp and str(device).startswith("cuda"))):
                    u_f = user_model(u_in[idx])
                    u_cf_b = torch.from_numpy(ucf).to(device)[idx]
                    u_tx_b = torch.from_numpy(utx).to(device)[idx]
                    u_cf_proj = l2norm_t(proj_u_cf(u_cf_b), -1)
                    u_tx_proj = l2norm_t(proj_u_tx(u_tx_b), -1)

                    if args.use_infonce_concat:
                        loss_align_u = info_nce(u_cf_proj, u_tx_proj, t=args.temperature)
                        loss_u = F.mse_loss(u_f, l2norm_t((u_cf_proj + u_tx_proj) / 2.0, -1)) \
                                 + args.infonce_lambda * loss_align_u
                    else:
                        target_u = l2norm_t(u_cf_proj + u_tx_proj, -1)
                        loss_u = F.mse_loss(u_f, target_u)

                opt.zero_grad(); scaler.scale(loss_u).backward(); scaler.step(opt); scaler.update()

            # ---- items (정렬)
            for st in tqdm(range(0, i_in.size(0), bs), desc=" items  ", ncols=100):
                idx = perm_i[st:st+bs]
                with torch.cuda.amp.autocast(enabled=(args.use_amp and str(device).startswith("cuda"))):
                    i_f = item_model(i_in[idx])
                    i_cf_b = torch.from_numpy(icf).to(device)[idx]
                    i_tx_b = torch.from_numpy(itx).to(device)[idx]
                    i_cf_proj = l2norm_t(proj_i_cf(i_cf_b), -1)
                    i_tx_proj = l2norm_t(proj_i_tx(i_tx_b), -1)

                    if args.use_infonce_concat:
                        loss_align_i = info_nce(i_cf_proj, i_tx_proj, t=args.temperature)
                        loss_i = F.mse_loss(i_f, l2norm_t((i_cf_proj + i_tx_proj) / 2.0, -1)) \
                                 + args.infonce_lambda * loss_align_i
                    else:
                        target_i = l2norm_t(i_cf_proj + i_tx_proj, -1)
                        loss_i = F.mse_loss(i_f, target_i)

                opt.zero_grad(); scaler.scale(loss_i).backward(); scaler.step(opt); scaler.update()

            # ---- (선택) BPR joint
            if args.bpr_lambda > 0.0 and pos_lists is not None:
                print(f"[BPR] epoch {ep+1}: steps={args.bpr_steps}, neg={args.bpr_neg}, lambda={args.bpr_lambda}")
                num_u = u_in.size(0); num_i = i_in.size(0)
                for _ in range(args.bpr_steps):
                    batch_users = np.random.randint(0, num_u, size=(bs,))
                    trip = sample_bpr_triplets(pos_lists, num_i, batch_users,
                                               neg_per_pos=args.bpr_neg, device=device)
                    if trip is None:
                        continue
                    u_b, ip_b, in_b = trip
                    with torch.cuda.amp.autocast(enabled=(args.use_amp and str(device).startswith("cuda"))):
                        u_f   = user_model(u_in[u_b])
                        i_pos = item_model(i_in[ip_b])
                        i_neg = item_model(i_in[in_b])
                        loss_bpr = bpr_loss(u_f, i_pos, i_neg)
                        loss = args.bpr_lambda * loss_bpr
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        with torch.no_grad():
            u_fused = user_model(u_in).cpu().numpy().astype(np.float32)
            i_fused = item_model(i_in).cpu().numpy().astype(np.float32)

    elif args.method == "two_tower":
        print("\n[method] two_tower (InfoNCE 기본 포함)")
        user_model, item_model = build_fuser(
            "two_tower", ucf.shape[1], utx.shape[1], icf.shape[1], itx.shape[1],
            args.out_dim, args.hidden, args.dropout
        )
        pos_lists = None
        if args.bpr_lambda > 0.0 and args.train_mat:
            pos_lists = load_train_interactions(args.train_mat, ucf.shape[0], icf.shape[0])
            if pos_lists is None:
                print("[WARN] train_mat을 읽지 못해 BPR을 건너뜁니다.")

        u_fused, i_fused = train_two_tower(
            user_model, item_model, ucf, utx, icf, itx,
            epochs=args.epochs, lr=args.lr, bs=args.batch_size, device=args.device,
            temperature=args.temperature, mse_weight=1.0, infonce_weight=1.0, use_amp=args.use_amp,
            bpr_lambda=args.bpr_lambda, bpr_steps=args.bpr_steps, bpr_neg=args.bpr_neg,
            pos_lists=pos_lists
        )
    else:
        raise ValueError(args.method)

    # ====== 최종 L2 정규화 ======
    if args.method != "whiten_concat":
        if not args.no_l2norm:
            u_fused = l2norm_np(u_fused)
            i_fused = l2norm_np(i_fused)
    else:
        if not args.no_l2norm:
            u_fused = l2norm_np(u_fused)
            i_fused = l2norm_np(i_fused)

    save_pkl(args.save_user, u_fused)
    save_pkl(args.save_item, i_fused)

    print(f"\n[fusion] saved:")
    print(f"  user → {args.save_user}  | shape={u_fused.shape}")
    print(f"  item → {args.save_item}  | shape={i_fused.shape}")

if __name__ == "__main__":
    main()
