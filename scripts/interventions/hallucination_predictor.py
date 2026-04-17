#!/usr/bin/env python3
"""
Track C — Hallucination Predictor from Frequency Probe Activation.

Tests whether the magnitude of the frequency probe's activation on an entity
token predicts whether the model will hallucinate on a rare-entity QA question.

Pipeline:
  1. Train a frequency probe on ScaleJSD at a given layer L* (the "peak layer"
     for each architecture) using train_probe_at_layer from frequency_steering.
  2. Load a QA dataset: PopQA (preferred), TriviaQA (fallback), or NQ.
  3. For each question:
       a. Tokenize question.
       b. Forward pass; capture residual at L* at the "last entity token"
          (last content-word-ish position before the final '?' — simple
          last-non-punctuation-token heuristic).
       c. Compute signed probe magnitude = <residual, v_L*>.
       d. Greedy-decode an answer (max_new_tokens=32).
       e. Exact-match / F1 against ground-truth answer(s) (normalized).
       f. Record token log-prob of the ground-truth answer (if expressible).
       g. Also compute a baseline signal: average per-token entropy of the
          greedy decoded answer ("logit-entropy baseline").
  4. Write per-question CSV + correlation_summary.json with:
       - Spearman rho between probe_magnitude and correct
       - Spearman rho between abs_probe_magnitude and correct
       - Stratification by popularity bin (PopQA only)
       - ROC-AUC of probe_magnitude as classifier of correctness
       - Comparison to the entropy baseline

Usage:
    python -m scripts.interventions.hallucination_predictor \
        --model allenai/OLMo-7B-hf \
        --revision main \
        --dataset popqa \
        --n-questions 2000 \
        --output-dir results/halluc_predictor/olmo-7b \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

# Reuse infrastructure
try:
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        _select_dtype,
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.interventions.frequency_steering import (  # noqa: E402
        ResidualCapture,
        _select_dtype,
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (  # noqa: E402
        load_synonym_pairs,
        pairs_to_samples,
    )


DATASET_FILES = {
    "emotion": "emotion_ngrams_dedup_filtered.jsonl",
    "medical": "medical_ngrams_dedup_filtered.jsonl",
    "legal": "legal_ngrams_dedup_filtered.jsonl",
    "scientific": "scientific_ngrams_dedup_filtered.jsonl",
    "verb": "verb_ngrams_dedup_filtered.jsonl",
}


# ─────────────────────────────────────────────────────────────────────
#  Model → peak-layer heuristic (starting guesses; used when not overridden)
# ─────────────────────────────────────────────────────────────────────

def default_peak_layer(model_id: str) -> int:
    """Return the default peak probe layer for each model family.

    These are starting guesses from the frequency_steering pilots. They can
    be overridden via --probe-layer.
    """
    mid = model_id.lower()
    if "olmo" in mid and "7b" in mid:
        return 16
    if "llama-3.1-8b" in mid or "llama3.1-8b" in mid or "llama-3-8b" in mid or "llama3-8b" in mid:
        return 15
    if "pythia-6.9b" in mid or "pythia6.9b" in mid:
        return 12
    if "pythia" in mid and "2.8b" in mid:
        return 10
    if "pythia" in mid and "1.4b" in mid:
        return 8
    if "pythia" in mid and "1b" in mid:
        return 7
    if "olmo" in mid and "1b" in mid:
        return 8
    # Conservative fallback — middle of the model
    return 12


# ─────────────────────────────────────────────────────────────────────
#  Dataset loaders
# ─────────────────────────────────────────────────────────────────────

def load_popqa(n_questions: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Load PopQA from HuggingFace datasets.

    Fields: question, possible_answers (list[str]), s_pop (subject Wikipedia
    popularity), o_pop (object popularity), s_wiki_title, o_wiki_title,
    prop (property).
    """
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    print(f"PopQA loaded: {len(ds)} rows, fields: {list(ds.features.keys())}")

    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(ds))[:n_questions]
    out = []
    for i in idxs:
        row = ds[int(i)]
        # possible_answers is JSON-encoded string in PopQA
        raw_answers = row.get("possible_answers") or row.get("answers") or row.get("answer")
        if isinstance(raw_answers, str):
            try:
                answers = json.loads(raw_answers)
            except Exception:
                answers = [raw_answers]
        elif isinstance(raw_answers, list):
            answers = raw_answers
        else:
            answers = []
        if not answers:
            continue
        pop = row.get("s_pop") or row.get("popularity")
        try:
            pop_val = float(pop) if pop is not None else None
        except Exception:
            pop_val = None
        out.append({
            "q_id": str(row.get("id", int(i))),
            "question": row["question"],
            "answers": [str(a) for a in answers],
            "entity_popularity": pop_val,
            "entity": row.get("subj", row.get("s_wiki_title", "")),
        })
    print(f"  -> kept {len(out)} PopQA questions (non-empty answers)")
    return out


def load_triviaqa(n_questions: int, seed: int = 42) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split="validation")
    print(f"TriviaQA loaded: {len(ds)} rows")
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(ds))[:n_questions]
    out = []
    for i in idxs:
        row = ds[int(i)]
        answer_obj = row.get("answer", {})
        aliases = list(answer_obj.get("aliases", []))
        norm_aliases = list(answer_obj.get("normalized_aliases", []))
        value = answer_obj.get("value", "")
        all_ans = []
        for a in [value] + aliases + norm_aliases:
            if a and a not in all_ans:
                all_ans.append(a)
        if not all_ans:
            continue
        out.append({
            "q_id": str(row.get("question_id", int(i))),
            "question": row["question"],
            "answers": all_ans,
            "entity_popularity": None,
            "entity": "",
        })
    print(f"  -> kept {len(out)} TriviaQA questions")
    return out


def load_nq(n_questions: int, seed: int = 42) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/natural_questions", "default",
                      split="validation")
    print(f"NQ loaded: {len(ds)} rows")
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(ds))
    out = []
    for i in idxs:
        row = ds[int(i)]
        ann = row.get("annotations", {})
        short = ann.get("short_answers", [])
        q = row["question"]["text"]
        # Pick first short answer span
        answers = []
        if short:
            sa = short[0]
            starts = sa.get("start_token", [])
            ends = sa.get("end_token", [])
            doc_tokens = row["document"]["tokens"]["token"]
            for st, en in zip(starts, ends):
                if 0 <= st < en <= len(doc_tokens):
                    txt = " ".join(doc_tokens[st:en])
                    if txt:
                        answers.append(txt)
        if not answers:
            continue
        out.append({
            "q_id": str(row.get("id", int(i))),
            "question": q,
            "answers": answers,
            "entity_popularity": None,
            "entity": "",
        })
        if len(out) >= n_questions:
            break
    print(f"  -> kept {len(out)} NQ questions with short answers")
    return out


# ─────────────────────────────────────────────────────────────────────
#  Normalization + scoring
# ─────────────────────────────────────────────────────────────────────

_PUNCT = set(string.punctuation)
_ARTICLES = {"the", "a", "an"}


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation, drop articles, collapse WS."""
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, golds: List[str]) -> int:
    pn = normalize_answer(pred)
    if not pn:
        return 0
    for g in golds:
        if pn == normalize_answer(g):
            return 1
    # Contains match — gold appears as substring in pred (common when model
    # says "The author is X." and gold is just "X")
    for g in golds:
        gn = normalize_answer(g)
        if gn and gn in pn:
            return 1
    return 0


def token_f1(pred: str, golds: List[str]) -> float:
    pn_tokens = normalize_answer(pred).split()
    if not pn_tokens:
        return 0.0
    best = 0.0
    for g in golds:
        gn_tokens = normalize_answer(g).split()
        if not gn_tokens:
            continue
        common = Counter(pn_tokens) & Counter(gn_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        prec = n_common / len(pn_tokens)
        rec = n_common / len(gn_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


# ─────────────────────────────────────────────────────────────────────
#  Last entity-token heuristic
# ─────────────────────────────────────────────────────────────────────

_PUNCT_TOKEN_RE = re.compile(r"^[\s\W_]+$")  # whitespace + non-word


def find_last_entity_token(tokenizer, question: str, input_ids: torch.Tensor) -> int:
    """Heuristic: last non-punctuation token position in the question.

    For a question like "Who wrote Vihangamayogasamhita?", the final '?' is
    punctuation and the last proper-noun token ("samhita"-ish suffix) is what
    we want. We iterate backwards over decoded tokens and pick the first one
    that contains at least one alphanumeric char.
    """
    ids_list = input_ids[0].tolist()
    for i in range(len(ids_list) - 1, -1, -1):
        tok_text = tokenizer.decode([ids_list[i]])
        if _PUNCT_TOKEN_RE.match(tok_text):
            continue
        # Require at least one alphanumeric
        if any(c.isalnum() for c in tok_text):
            return i
    return len(ids_list) - 1


# ─────────────────────────────────────────────────────────────────────
#  Per-question forward pass
# ─────────────────────────────────────────────────────────────────────

def score_question(
    model,
    tokenizer,
    question: str,
    golds: List[str],
    direction_t: torch.Tensor,   # [D]
    layer: int,
    device: str,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Run one QA question through the pipeline."""
    # Tokenize as a QA-style prompt so the model completes the answer.
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Capture residual at L* with the question-only ids (we want the entity
    # representation within the question itself).
    q_ids = tokenizer.encode(question, return_tensors="pt").to(device)
    entity_pos = find_last_entity_token(tokenizer, question, q_ids)

    cap = ResidualCapture(model, [layer])
    try:
        with torch.no_grad():
            model(input_ids=q_ids)
        h = cap.outputs[layer]
    finally:
        cap.remove()

    pos = min(entity_pos, h.shape[1] - 1)
    residual = h[0, pos, :].float().cpu().numpy()
    probe_mag = float(np.dot(residual, direction_t.cpu().numpy()))

    # Greedy generate.
    prompt_len = int(input_ids.shape[1])
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=(tokenizer.pad_token_id
                          if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
        )

    new_ids = gen[0, prompt_len:].cpu().tolist()
    # Trim at first newline/period-question-endish boundary for cleaner EM
    raw_answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    answer = raw_answer.split("\n")[0].strip()

    em = exact_match(answer, golds)
    f1 = token_f1(answer, golds)

    # Per-token entropy baseline for the greedy-generated answer.
    # We do one teacher-forced pass over the full sequence and measure the
    # entropy of the logits at each generated position.
    entropy_baseline = float("nan")
    answer_logprob = float("nan")
    if gen.shape[1] > prompt_len:
        with torch.no_grad():
            out = model(input_ids=gen)
        logits = out.logits[0, prompt_len - 1:-1, :].float()  # predicts new_ids
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        # Entropy at each generated position (nats).
        ent = -(p * logp).sum(dim=-1)
        entropy_baseline = float(ent.mean().cpu())

        # answer_logprob = sum of log probs of the greedy-decoded answer
        # (this is the model's confidence in what it actually said, not in
        # the gold answer). Re-using it as a negative-uncertainty score.
        targets = torch.tensor(new_ids, device=device)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        answer_logprob = float(tok_logp.mean().cpu())

    return {
        "question": question,
        "ground_truth": " | ".join(golds[:5]),
        "model_answer": answer,
        "correct": int(em),
        "f1": float(f1),
        "probe_magnitude": probe_mag,
        "abs_probe_magnitude": abs(probe_mag),
        "answer_logprob": answer_logprob,
        "entropy_baseline": entropy_baseline,
        "entity_pos": int(pos),
        "n_question_tokens": int(q_ids.shape[1]),
    }


# ─────────────────────────────────────────────────────────────────────
#  Correlation / AUC reporting
# ─────────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def safe_spearman(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
            return float("nan"), float("nan")
        r = spearmanr(x[mask], y[mask])
        return float(r.correlation), float(r.pvalue)

    def safe_auc(y_true, scores):
        if roc_auc_score is None:
            return float("nan")
        y_true = np.asarray(y_true)
        scores = np.asarray(scores, dtype=float)
        mask = np.isfinite(scores)
        y = y_true[mask]
        s = scores[mask]
        if len(np.unique(y)) < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y, s))
        except Exception:
            return float("nan")

    correct = df["correct"].to_numpy()
    probe = df["probe_magnitude"].to_numpy()
    abs_probe = df["abs_probe_magnitude"].to_numpy()
    logprob = df["answer_logprob"].to_numpy()
    entropy = df["entropy_baseline"].to_numpy()

    r, p = safe_spearman(probe, correct)
    out["spearman_probe_vs_correct"] = {"rho": r, "pvalue": p}
    r, p = safe_spearman(abs_probe, correct)
    out["spearman_abs_probe_vs_correct"] = {"rho": r, "pvalue": p}
    r, p = safe_spearman(logprob, correct)
    out["spearman_answer_logprob_vs_correct"] = {"rho": r, "pvalue": p}
    r, p = safe_spearman(entropy, correct)
    out["spearman_entropy_vs_correct"] = {"rho": r, "pvalue": p}

    out["auc_probe_vs_correct"] = safe_auc(correct, probe)
    out["auc_abs_probe_vs_correct"] = safe_auc(correct, abs_probe)
    out["auc_answer_logprob_vs_correct"] = safe_auc(correct, logprob)
    out["auc_neg_entropy_vs_correct"] = safe_auc(correct, -entropy)

    # Stratify by popularity bin if available.
    if "entity_popularity" in df.columns and df["entity_popularity"].notna().any():
        pops = df["entity_popularity"].to_numpy()
        finite = np.isfinite(pops.astype(float))
        if finite.sum() >= 30:
            p_vals = pops[finite]
            try:
                bins = np.nanquantile(p_vals, [0.0, 0.25, 0.5, 0.75, 1.0])
                bins[0] -= 1e-9
                bins[-1] += 1e-9
                labels = ["Q1_rarest", "Q2", "Q3", "Q4_common"]
                stratified = {}
                for i, lab in enumerate(labels):
                    mask = (pops >= bins[i]) & (pops < bins[i + 1]) & finite
                    if mask.sum() < 15:
                        continue
                    sub = df[mask]
                    rr, pp = safe_spearman(sub["probe_magnitude"], sub["correct"])
                    stratified[lab] = {
                        "n": int(mask.sum()),
                        "accuracy": float(sub["correct"].mean()),
                        "rho_probe_vs_correct": rr,
                        "pvalue": pp,
                        "auc_probe": safe_auc(sub["correct"].to_numpy(),
                                              sub["probe_magnitude"].to_numpy()),
                    }
                out["popularity_stratified"] = stratified
            except Exception as e:
                out["popularity_stratification_error"] = str(e)

    out["overall_accuracy"] = float(df["correct"].mean())
    out["n_questions"] = int(len(df))
    return out


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset", choices=["popqa", "triviaqa", "nq"],
                        default="popqa")
    parser.add_argument("--n-questions", type=int, default=2000)
    parser.add_argument("--dataset-dir", required=True,
                        help="ScaleJSD dir for training the frequency probe")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-layer", type=int, default=None,
                        help="Layer for probe. If None, uses default_peak_layer.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dtype == "auto":
        dtype = _select_dtype(args.model, args.device)
    else:
        dtype = {"float32": torch.float32,
                 "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]

    print(f"Loading model {args.model} (rev={args.revision}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: Dict[str, Any] = {"torch_dtype": dtype}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kw["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(args.device)
    model.eval()

    # ── 1. Train frequency probe at L* ─────────────────────────────────
    probe_layer = (args.probe_layer
                   if args.probe_layer is not None
                   else default_peak_layer(args.model))
    print(f"\n=== Training frequency probe at L{probe_layer} ===")

    all_samples = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
        all_samples.extend(samples)
    if not all_samples:
        raise RuntimeError(f"No ScaleJSD samples found in {args.dataset_dir}")
    print(f"Total ScaleJSD samples: {len(all_samples)}")

    direction, auroc, _ = train_probe_at_layer(
        model, all_samples, probe_layer, args.device)
    print(f"  Probe AUROC at L{probe_layer}: {auroc:.3f}")

    direction_t = torch.tensor(direction, dtype=torch.float32, device=args.device)

    # Save probe direction.
    np.save(Path(args.output_dir) / "probe_direction.npy", direction)
    with open(Path(args.output_dir) / "probe_meta.json", "w") as f:
        json.dump({
            "model": args.model,
            "revision": args.revision,
            "probe_layer": int(probe_layer),
            "probe_auroc": float(auroc),
            "n_scalejsd_samples": len(all_samples),
            "dtype": str(dtype),
        }, f, indent=2)

    # ── 2. Load QA dataset ─────────────────────────────────────────────
    print(f"\n=== Loading QA dataset: {args.dataset} ===")
    if args.dataset == "popqa":
        qa = load_popqa(args.n_questions, seed=args.seed)
    elif args.dataset == "triviaqa":
        qa = load_triviaqa(args.n_questions, seed=args.seed)
    else:
        qa = load_nq(args.n_questions, seed=args.seed)
    print(f"Got {len(qa)} QA questions")

    # ── 3. Score each question ─────────────────────────────────────────
    print(f"\n=== Scoring {len(qa)} questions ===")
    rows: List[Dict[str, Any]] = []
    n_errors = 0
    for i, item in enumerate(qa):
        if i % args.progress_every == 0:
            acc = (np.mean([r["correct"] for r in rows]) if rows else 0.0)
            print(f"  [{i}/{len(qa)}] acc={acc:.3f} errs={n_errors}", flush=True)
        try:
            r = score_question(
                model, tokenizer,
                question=item["question"],
                golds=item["answers"],
                direction_t=direction_t,
                layer=probe_layer,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            r["q_id"] = item["q_id"]
            r["entity"] = item.get("entity", "")
            r["entity_popularity"] = item.get("entity_popularity")
            rows.append(r)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"    error on q{i}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.output_dir) / "per_question.csv", index=False)
    print(f"\nWrote per_question.csv with {len(df)} rows ({n_errors} errors)")

    # ── 4. Correlation summary ─────────────────────────────────────────
    summary = compute_correlations(df)
    summary["model"] = args.model
    summary["revision"] = args.revision
    summary["dataset"] = args.dataset
    summary["probe_layer"] = int(probe_layer)
    summary["probe_auroc_scalejsd"] = float(auroc)
    summary["n_errors"] = int(n_errors)
    with open(Path(args.output_dir) / "correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  CORRELATION SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
