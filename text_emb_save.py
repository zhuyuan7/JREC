# embed_save.py
import os
import re
import json
import torch
import pickle
import argparse
from tqdm import tqdm
import torch.nn.functional as F

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

"""
예시 실행:
python embed_save.py \
  --datasets arts electronics games home tools \
  --models \
    intfloat/e5-mistral-7b-instruct \
    BAAI/bge-m3 \
    jinaai/jina-embeddings-v3 \
    Alibaba-NLP/gte-Qwen2-7B-instruct \
    Salesforce/SFR-Embedding-Mistral \
    meta-llama/Llama-2-7b-hf \
    mistralai/Mistral-7B-v0.1 \
    Qwen/Qwen2-7B-Instruct \
    deepseek-ai/deepseek-llm-7b-base \
  --batch_size 128 \
  --max_len 512 \
  --cuda 0 \
  --pooling mean
"""

# -----------------------------
# 로딩/정렬 유틸
# -----------------------------
def load_kv_list(json_path, id_key, text_key="profile"):
    kv = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            kv.append((d[id_key], d[text_key]))
    # 숫자/문자 혼재 안전 정렬
    def _key(x):
        try:
            return (0, int(x[0]))
        except Exception:
            return (1, str(x[0]))
    kv.sort(key=_key)
    return kv  # [(id, text), ...]

def sanitize_tag(name: str) -> str:
    tag = name.split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", tag)
    return tag

def ensure_pad_token(tokenizer, model=None):
    # pad 토큰이 없으면 eos로 대체 (둘 다 없으면 [PAD] 추가)
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))
    # 모델 config에도 반영
    if model is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

# -----------------------------
# 모델 타입/인스트럭션 판별
# -----------------------------
def is_encoder_name(name: str) -> bool:
    n = name.lower()
    # 대표 encoder 키워드 (+ SFR는 인코더 취급)
    enc_keys = [
        "e5",
        "bge",
        "gte",
        "roberta",
        "bert",
        "mpnet",
        "jina-embeddings",
        "nv-embed",
        "nvidia/",
        "sfr-embedding",     # Salesforce/SFR-Embedding-Mistral
    ]
    return any(k in n for k in enc_keys)

def is_decoder_name(name: str) -> bool:
    n = name.lower()
    dec_keys = ["llama", "mistral", "qwen", "deepseek", "mixtral"]
    return any(k in n for k in dec_keys)

def model_instruction(texts, model_name, role="passage"):
    """
    모델 권장 프롬프트를 붙여줌. 기본은 문서/아이템/프로파일용 'passage' 스타일.
    (user/item 모두 동일하게 문서 임베딩으로 취급)
    """
    n = model_name.lower()
    if "e5" in n:
        # E5 계열: query/passsage 템플릿
        prefix = f"{role}: "
        return [prefix + t for t in texts]
    if "bge" in n:
        # BGE 계열
        prefix = "Represent this sentence for retrieval: "
        return [prefix + t for t in texts]
    if "gte" in n:
        # GTE 계열
        prefix = "Represent the document for retrieval: "
        return [prefix + t for t in texts]
    if "jina-embeddings" in n:
        # Jina v3
        prefix = "document: " if role != "query" else "query: "
        return [prefix + t for t in texts]
    if "nv-embed" in n or "nvidia/" in n:
        # NV-Embed (문서)
        prefix = "passage: "
        return [prefix + t for t in texts]
    if "sfr-embedding" in n:
        # SFR-Embedding-Mistral
        prefix = "passage: "
        return [prefix + t for t in texts]
    # 일반 디코더 (Llama/Mistral/Qwen/DeepSeek)
    return texts

# -----------------------------
# 인코딩 함수 (배치)
# -----------------------------
@torch.inference_mode()
def encode_encoder(model, tokenizer, texts, batch_size=128, max_len=512, device="cuda", pooling="mean", model_name=""):
    embs = []
    n_lower = model_name.lower()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Jina 등 일부 모델은 sentence_embedding(s) 키를 리턴
        out = model(**inputs, output_hidden_states=False, return_dict=True)
        x = None

        # 1) 가장 먼저 'sentence_embedding' 키/속성 검사
        if isinstance(out, dict):
            if "sentence_embeddings" in out:
                x = out["sentence_embeddings"]
            elif "sentence_embedding" in out:
                x = out["sentence_embedding"]
        else:
            if hasattr(out, "sentence_embeddings"):
                x = out.sentence_embeddings
            elif hasattr(out, "sentence_embedding"):
                x = out.sentence_embedding

        # 2) 위가 없으면 일반 풀링
        if x is None:
            last_hidden = out.last_hidden_state  # [B, T, H]
            if pooling == "cls":
                x = last_hidden[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                x = summed / counts

        x = F.normalize(x.detach().float(), dim=-1)
        embs.append(x.cpu())
    return torch.cat(embs, dim=0)

@torch.inference_mode()
def encode_decoder(model, tokenizer, texts, batch_size=64, max_len=512, device="cuda"):
    # 디코더는 보통 left-padding이 안정적
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1]  # [B, T, H]
        # 실제 마지막 토큰 인덱스(패딩 제외)
        # last_tok = (inputs["attention_mask"].sum(dim=1) - 1).clamp(min=0)
        seq_positions = torch.arange(last_hidden.size(1), device=last_hidden.device)
        last_tok = (inputs["attention_mask"] * seq_positions).argmax(dim=1)
        x = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), last_tok]
        x = F.normalize(x.detach().float(), dim=-1)
        embs.append(x.cpu())
    return torch.cat(embs, dim=0)

# -----------------------------
# 모델/토크나이저 로더 (가드 포함)
# -----------------------------
def load_tokenizer(model_name: str):
    trust_remote = True if "jina-embeddings" in model_name.lower() else False
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote)

def load_model(model_name: str, encoder: bool):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    trust_remote = True if "jina-embeddings" in model_name.lower() else False
    if encoder:
        return AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote)

# -----------------------------
# 메인
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+",
                        default=[
                            "intfloat/e5-mistral-7b-instruct",
                            "BAAI/bge-m3",
                            "jinaai/jina-embeddings-v3",
                            "Alibaba-NLP/gte-Qwen2-7B-instruct",
                            "Salesforce/SFR-Embedding-Mistral",
                            # 디코더 LLM 7B들
                            "meta-llama/Llama-2-7b-hf",
                            "mistralai/Mistral-7B-v0.1",
                            "Qwen/Qwen2-7B-Instruct",
                            "deepseek-ai/deepseek-llm-7b-base",
                        ])
    parser.add_argument("--datasets", type=str, nargs="+", default=["arts", "electronics", "games", "home", "tools"])
    parser.add_argument("--data_root", type=str, default="/root/EasyRec/data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in args.models:
        print(f"\n=== Loading model: {model_name} ===")
        # 토크나이저 로드
        try:
            tokenizer = load_tokenizer(model_name)
        except Exception as e:
            msg = str(e)
            print(f"[SKIP] Tokenizer load failed for '{model_name}': {msg}")
            continue

        use_encoder = is_encoder_name(model_name)
        use_decoder = is_decoder_name(model_name) and not use_encoder

        # 모델 로드
        try:
            model = load_model(model_name, encoder=use_encoder)
        except Exception as e:
            msg = str(e)
            # Gated/Private 모델 접근 오류는 스킵
            if "gated repo" in msg.lower() or "Repository Not Found".lower() in msg.lower():
                print(f"[SKIP] Gated/NotFound model '{model_name}': {msg}")
                continue
            print(f"[SKIP] Model load failed for '{model_name}': {msg}")
            continue

        model = model.to(device).eval()
        ensure_pad_token(tokenizer, model if use_decoder else None)  # 디코더 패딩 가드

        if use_encoder:
            encode_fn = lambda txts: encode_encoder(
                model, tokenizer, txts,
                batch_size=args.batch_size,
                max_len=args.max_len,
                device=device,
                pooling=args.pooling,
            )
        else:
            # 디코더는 연산비 크므로 배치 축소 권장
            dec_bs = 32 if any(x in model_name.lower() for x in ["llama", "mistral", "qwen", "deepseek"]) else args.batch_size
            encode_fn = lambda txts: encode_decoder(
                model, tokenizer, txts,
                batch_size=dec_bs,
                max_len=args.max_len,
                device=device,
            )

        tag = sanitize_tag(model_name)

        for dataset in args.datasets:
            ds_root = os.path.join(args.data_root, dataset)
            user_json = os.path.join(ds_root, "user_profile.json")
            item_json = os.path.join(ds_root, "item_profile.json")
            save_dir  = os.path.join(ds_root, "text_emb")
            os.makedirs(save_dir, exist_ok=True)

            if not os.path.exists(user_json) or not os.path.exists(item_json):
                print(f"[Skip {dataset}] user/item_profile.json not found in {ds_root}")
                continue

            # 1) 로드 & 정렬
            user_kv = load_kv_list(user_json, id_key="user_id", text_key="profile")
            item_kv = load_kv_list(item_json, id_key="item_id", text_key="profile")
            user_ids = [u for u,_ in user_kv]
            item_ids = [i for i,_ in item_kv]
            user_txt = [p for _,p in user_kv]
            item_txt = [p for _,p in item_kv]

            # 2) 모델별 인스트럭션 적용 (문서/프로파일 → passage로 통일)
            user_txt_i = model_instruction(user_txt, model_name, role="passage")
            item_txt_i = model_instruction(item_txt, model_name, role="passage")
            all_txt = user_txt_i + item_txt_i

            print(f"[{dataset} | {tag}] users={len(user_txt)} items={len(item_txt)}")
            embs = encode_fn(all_txt)
            u_emb = embs[:len(user_txt)].numpy()
            i_emb = embs[len(user_txt):].numpy()
            print(u_emb.shape)
            print(i_emb.shape)

            # 3) 저장
            u_path = os.path.join(save_dir, f"user_{tag}.pkl")
            i_path = os.path.join(save_dir, f"item_{tag}.pkl")
            with open(u_path, "wb") as f:
                pickle.dump(u_emb, f)
            with open(i_path, "wb") as f:
                pickle.dump(i_emb, f)
            with open(os.path.join(save_dir, f"user_ids_{tag}.json"), "w", encoding="utf-8") as f:
                json.dump(user_ids, f, ensure_ascii=False)
            with open(os.path.join(save_dir, f"item_ids_{tag}.json"), "w", encoding="utf-8") as f:
                json.dump(item_ids, f, ensure_ascii=False)

            print(f"  -> saved: {u_path}, {i_path}")

if __name__ == "__main__":
    main()


