#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCTIG.py

Requirement 1: Fine-tune ChatGLM-6B with LoRA on Real_CTI.csv
and then generate a fake CTI dataset using the fine-tuned model.

Paper (Table 1) training settings for the Generation model:
- Epochs: 10
- Batch size: 8
- Learning rate: 2e-5  # adjusted from paper to prevent NaN loss
- LoRA r: 8
- LoRA alpha: 16

Base model path (absolute): /root/ChatGLM-6B/model/chatglm-6b
Output: LoRA adapter => ./FCTIG_lora_ckpt ; Generated CSV => Dataset/Fake_CTI_generated.csv
Optimized for RTX 4090 with mixed precision (bf16 preferred, otherwise fp16) and AMP.

Usage examples:
    # Train only
    python FCTIG.py --do_generate=False

    # Generate only, using existing LoRA adapter
    python FCTIG.py --do_train=False --generation_source_limit=5000
"""

import os
import argparse
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

# Enable TF32 for better numerical stability/perf on Ampere+ GPUs
import torch.backends.cuda as cuda_backends
try:
    cuda_backends.matmul.allow_tf32 = True
except Exception:
    pass

# -------------------------------
# Constants
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_BASE_MODEL_PATH = "/root/ChatGLM-6B/model/chatglm-6b"
DEFAULT_OUTPUT_LORA_DIR = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "FCTIG_lora_ckpt")
DEFAULT_GENERATED_CSV = os.path.join(PROJECT_ROOT, "data", "Fake_CTI_generated.csv")
DEFAULT_TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "Real_CTI.csv")

# Paper params (defaults)
PAPER_EPOCHS = 10
PAPER_BATCH_SIZE = 8
PAPER_LR = 2e-5  # Lowered learning rate to prevent NaN loss
PAPER_LORA_R = 8
PAPER_LORA_ALPHA = 16

# Reasonable defaults
DEFAULT_BLOCK_SIZE = 256  # reduce context to save memory and improve stability
DEFAULT_WARMUP_RATIO = 0.1  # longer warmup helps avoid loss spikes/NaN
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_MAX_GEN_LEN = 512
DEFAULT_GENERATION_SOURCE_LIMIT = 5000


def bf16_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()

    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


class CausalTextDataset(Dataset):
    """A simple CLM dataset from a CSV file.

    It detects the column containing text (trying common names first), cleans NaNs,
    and returns tokenized chunks for causal LM training compatible with ChatGLM (gMASK/sop).
    """

    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, block_size: int = DEFAULT_BLOCK_SIZE,
                 text_col: Optional[str] = None):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if text_col is None:
            candidates = [
                "text", "content", "src", "article", "input", "body", "desc", "description", "TitleAndContent"
            ]
            for c in candidates:
                if c in df.columns:
                    text_col = c
                    break
            if text_col is None:
                # fallback to first object/string-like column
                for c in df.columns:
                    if df[c].dtype == object:
                        text_col = c
                        break
        if text_col is None:
            raise ValueError("Could not infer text column from CSV. Please specify --text_col explicitly.")

        self.texts: List[str] = [str(x) for x in df[text_col].fillna("").tolist() if str(x).strip() != ""]
        if len(self.texts) == 0:
            raise ValueError("No non-empty texts found in the specified column.")
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Resolve special tokens required by ChatGLM
        mask_id = getattr(tokenizer, "mask_token_id", None)
        sop_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if hasattr(tokenizer, "get_command"):
            try:
                if mask_id is None:
                    mask_id = tokenizer.get_command("gMASK")
            except Exception:
                try:
                    mask_id = tokenizer.get_command("MASK")
                except Exception:
                    pass
            try:
                if sop_id is None:
                    sop_id = tokenizer.get_command("sop")
            except Exception:
                sop_id = None
            try:
                if eos_id is None:
                    eos_id = tokenizer.get_command("eos")
            except Exception:
                eos_id = tokenizer.eos_token_id
        # Fallback to tokenizer special ids if commands unavailable
        if mask_id is None:
            mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise RuntimeError("ChatGLM mask token id could not be resolved (mask/gMASK).")

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
            pad_id = tokenizer.pad_token_id

        # Defer tokenization to __getitem__ to leverage ChatGLM tokenizer's internal logic
        self.pad_id = pad_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        # DO NOT request attention_mask explicitly; let ChatGLM tokenizer build what it needs internally
        enc = self.tokenizer(
            txt,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)

        labels = input_ids.clone()
        if self.pad_id is not None:
            labels[input_ids == self.pad_id] = -100

        return {"input_ids": input_ids, "labels": labels}


def _prepare_tokenizer_and_model(base_model: str):
    print("[FCTIG] Loading tokenizer and base model from:", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Ensure ChatGLM prefers left padding
    tokenizer.padding_side = "left"

    # Force bfloat16 for stability on RTX 4090
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    if torch.cuda.is_available():
        model = model.cuda()
        # Force cast all weights to bfloat16 to reduce memory and keep consistency
        try:
            model = model.to(dtype=torch.bfloat16)
        except Exception:
            pass
    return tokenizer, model


# New: prepare CSV from JSON if missing
def prepare_training_csv_if_missing(csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "cti_test.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "data2.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "data3.json"),
    ]
    for jp in candidates:
        if os.path.exists(jp):
            try:
                dfj = pd.read_json(jp)
                # Prefer human-written entries if label exists
                if "label" in dfj.columns:
                    dfj = dfj[dfj["label"].astype(str) == "0"] if (dfj["label"].astype(str) == "0").any() else dfj
                text_col = None
                for c in ["src", "text", "content", "article", "input", "body", "desc", "description", "TitleAndContent"]:
                    if c in dfj.columns:
                        text_col = c
                        break
                if text_col is None:
                    continue
                out_df = pd.DataFrame({"src": dfj[text_col].astype(str)})
                out_df = out_df[out_df["src"].str.strip() != ""].drop_duplicates()
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                out_df.to_csv(csv_path, index=False)
                print(f"[FCTIG] Prepared training CSV from {os.path.basename(jp)} -> {csv_path} ({len(out_df)} rows)")
                return csv_path
            except Exception as e:
                print(f"[FCTIG][Warn] Failed to convert {jp} to CSV: {e}")
                continue
    raise FileNotFoundError(f"No training CSV at {csv_path} and no convertible JSON found under data/")


# Replace entire train function with bf16-optimized version
def train(args):
    os.makedirs(os.path.dirname(args.generated_csv), exist_ok=True)
    os.makedirs(args.output_lora_dir, exist_ok=True)

    tokenizer, base_model = _prepare_tokenizer_and_model(args.base_model)

    # Attach LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)

    # Memory-saving features to mitigate OOM
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if getattr(args, "gradient_checkpointing", True):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass

    # Dataset and DataLoader
    dataset = CausalTextDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        block_size=args.block_size,
        text_col=args.text_col,
    )
    loader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=True, drop_last=True)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_update_steps = (len(loader) // max(1, args.gradient_accumulation_steps)) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    model.train()
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"[FCTIG][Train] Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            # bfloat16 mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            lr_show = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.lr
            progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_show:.2e}"})

        # Save LoRA adapter checkpoint each epoch
        save_dir = os.path.join(args.output_lora_dir, f"epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        print(f"[FCTIG] Saved LoRA adapter to {save_dir}")

    # Save final LoRA adapter
    model.save_pretrained(args.output_lora_dir)
    print(f"[FCTIG] Training finished. Final LoRA saved to {args.output_lora_dir}")
    return tokenizer, model


def _build_generation_prompt(src_text: str) -> str:
    # Instruction in Chinese to guide ChatGLM to produce a synthetic CTI report
    # Keep style aligned with project datasets using fields 'src' and 'rewrite'
    return (
        "请根据下列‘真实’网络安全威胁情报内容，生成一份新的、表述清晰、连贯完整的‘合成’网络安全威胁情报，"
        "避免简单同义替换，尽量保持事实逻辑但改变表述方式，长度约300字：\n\n"
        f"【真实情报】\n{src_text}\n\n"
        "【生成要求】\n1) 不要逐条分点；2) 用通顺自然的中文叙述；3) 不要包含元指令。\n\n"
        "【合成情报】："
    )


def generate_csv(args, tokenizer: AutoTokenizer, model):
    model.eval()
    src_df = pd.read_csv(args.csv_path)

    # Limit the number of source rows for generation
    if args.generation_source_limit is not None and args.generation_source_limit > 0:
        src_df = src_df.head(args.generation_source_limit)

    # Find text column again for generation
    text_col = args.text_col
    if text_col is None:
        for c in ["text", "content", "src", "article", "input", "body", "desc", "description", "TitleAndContent"]:
            if c in src_df.columns:
                text_col = c
                break
        if text_col is None:
            for c in src_df.columns:
                if src_df[c].dtype == object:
                    text_col = c
                    break
    if text_col is None:
        raise ValueError("Could not infer text column for generation; please pass --text_col.")

    out_rows = []
    os.makedirs(os.path.dirname(args.generated_csv), exist_ok=True)

    num_variants = max(1, getattr(args, 'num_variants_per_source', 1))
    total = len(src_df) * num_variants

    with torch.no_grad():
        pbar = tqdm(total=total, desc="[FCTIG] Generating")
        produced = 0
        for _, row in src_df.iterrows():
            src_text = str(row[text_col])
            if not isinstance(src_text, str) or src_text.strip() == "":
                # Skip but advance progress for all would-be variants
                pbar.update(num_variants)
                produced += num_variants
                continue

            # Token-level truncation before building the prompt
            try:
                tokenized_source = tokenizer(src_text, return_tensors=None, add_special_tokens=False)['input_ids']
                if len(tokenized_source) > args.max_source_length:
                    truncated_ids = tokenized_source[:args.max_source_length]
                    src_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            except Exception:
                pass

            for _ in range(num_variants):
                prompt = _build_generation_prompt(src_text)

                generated = None
                if hasattr(model, "chat"):
                    try:
                        response, _ = model.chat(tokenizer, prompt, history=[], temperature=0.7, top_p=0.9, max_length=args.max_gen_len)
                        generated = response
                    except TypeError:
                        response = model.chat(tokenizer, prompt, history=[])
                        generated = response[0] if isinstance(response, (list, tuple)) else response
                if generated is None:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.block_size)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_gen_len,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,
                    )
                    generated = tokenizer.decode(gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                out_rows.append({
                    "src": src_text,
                    "rewrite": str(generated).strip(),
                    "label": 1,
                })

                produced += 1
                pbar.update(1)
                if torch.cuda.is_available() and (produced % 256 == 0):
                    torch.cuda.empty_cache()
        pbar.close()

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.generated_csv, index=False)
    print(f"[FCTIG] Saved generated fake CTI to {args.generated_csv} ({len(out_df)} rows)")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ChatGLM-6B with LoRA and generate fake CTI dataset")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Absolute path to base ChatGLM-6B model")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_TRAIN_CSV, help="Path to Real_CTI.csv")
    parser.add_argument("--text_col", type=str, default=None, help="Name of the text column in CSV (auto-detect if None)")
    parser.add_argument("--output_lora_dir", type=str, default=DEFAULT_OUTPUT_LORA_DIR, help="Where to save LoRA adapter")
    parser.add_argument("--generated_csv", type=str, default=DEFAULT_GENERATED_CSV, help="Where to save generated fake CTI CSV")

    # Core paper params
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=PAPER_BATCH_SIZE, help="Target effective batch size per paper")
    # Use a very small per-device batch size by default to fit 24GB GPUs; accumulate to match effective batch size
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Actual dataloader batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=PAPER_LR)
    parser.add_argument("--lora_r", type=int, default=PAPER_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=PAPER_LORA_ALPHA)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max_gen_len", type=int, default=DEFAULT_MAX_GEN_LEN)

    # New: max token length for source text before truncation
    parser.add_argument("--max_source_length", type=int, default=512, help="Max token length for source text before truncation")

    # New: number of variants to generate per source
    parser.add_argument("--num_variants_per_source", type=int, default=1, help="How many variants to generate for each source item")

    # Optional: override output CSV path via --output_file
    parser.add_argument("--output_file", type=str, default=None, help="Override output CSV path (alias of --generated_csv)")

    # New staged execution flags
    parser.add_argument("--do_train", type=str2bool, default=True, help="Whether to run the fine-tuning stage")
    parser.add_argument("--do_generate", type=str2bool, default=True, help="Whether to run the generation stage")
    parser.add_argument("--generation_source_limit", type=int, default=DEFAULT_GENERATION_SOURCE_LIMIT, help="Limit number of source rows for generation")

    # Memory-saving toggle (unused in current train loop but kept for CLI compatibility)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True, help="Enable gradient checkpointing to reduce memory usage")

    # Stability option (unused in current train loop but kept for CLI compatibility)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Clip gradient norm for stability (0 to disable)")

    args = parser.parse_args()

    # Derive gradient accumulation to achieve target effective batch size if needed
    if args.per_device_batch_size > 0:
        effective = args.per_device_batch_size * max(1, torch.cuda.device_count()) * max(1, args.gradient_accumulation_steps)
        if effective != args.batch_size:
            if args.gradient_accumulation_steps == 1 and args.per_device_batch_size > 0:
                denom = args.per_device_batch_size * max(1, torch.cuda.device_count())
                args.gradient_accumulation_steps = max(1, args.batch_size // max(1, denom))

    # Map --output_file to internal --generated_csv if provided
    if getattr(args, 'output_file', None):
        args.generated_csv = args.output_file

    return args


if __name__ == "__main__":
    args = parse_args()

    # Ensure training CSV exists (or create from JSON in data/ if available)
    try:
        args.csv_path = prepare_training_csv_if_missing(args.csv_path)
    except Exception as e:
        print(f"[FCTIG][Warn] {e}")

    tokenizer = None
    model = None

    if args.do_train:
        tokenizer, model = train(args)

    if args.do_generate:
        # If we didn't just train in this run, load tokenizer + model with LoRA adapter
        if tokenizer is None or model is None:
            tokenizer, base_model = _prepare_tokenizer_and_model(args.base_model)
            # Load LoRA adapter
            if os.path.isdir(args.output_lora_dir):
                try:
                    model = PeftModel.from_pretrained(base_model, args.output_lora_dir)
                    if torch.cuda.is_available():
                        model = model.cuda()
                except Exception as e:
                    print(f"[FCTIG][Warn] Failed to load LoRA from {args.output_lora_dir}: {e}. Using base model for generation.")
                    model = base_model
            else:
                print(f"[FCTIG][Warn] LoRA directory not found: {args.output_lora_dir}. Using base model for generation.")
                model = base_model
        generate_csv(args, tokenizer, model)