#!/usr/bin/env python3
"""
Method 1: Max Activating Examples — Token-Level Feature Interpretation
(Three-tier corpus strategy)

For each target SAE feature, stream multilingual text through the model,
record per-token feature activations, and collect the top-K highest
activating tokens with their surrounding context window.

Three corpus tiers (run separately or combined via --corpus mixed):

  flores:    FLORES+ parallel sentences (~2000 per lang). Because sentences
             are aligned across languages, you can directly compare whether
             a feature fires on the same semantic position in English vs Hindi.
             This is the unique methodological advantage for cross-lingual analysis.

  culturax:  Broad multilingual web text (streamed). Lets you discover what
             the feature encodes in the wild, outside any safety framing.

  xsafety:   The XSafety harmful/benign dataset you already have. Running
             features on this separately lets you compare: does the feature
             fire on the same tokens in safety text vs general text?

Outputs per corpus tier:
  results/max_act/{tier}/top_activations.json
  results/max_act/{tier}/activation_stats.json
  results/max_act/{tier}/dashboards/feat_{id}.txt

The "mixed" mode runs all three and produces a combined output with
a "source" tag on every example so you can filter/compare downstream.

Usage:
  python -m saefty.analysis.max_activating_examples --corpus flores
  python -m saefty.analysis.max_activating_examples --corpus culturax --tokens-per-lang 500000
  python -m saefty.analysis.max_activating_examples --corpus xsafety
  python -m saefty.analysis.max_activating_examples --corpus mixed
"""

import argparse
import heapq
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from saefty.analysis.feature_identification import (
    MODEL_NAME, CHECKPOINT_PATH, HOOK_LAYER, LANGUAGES, DEVICE, load_sae,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────
TOP_K = 30
CONTEXT_WINDOW = 40
BATCH_SIZE = 8
SEQ_LEN = 256
TOKENS_PER_LANG = 500_000
RANDOM_SAMPLE_SIZE = 2000

# ── FLORES+ config ─────────────────────────────────────────────────────────
# map our internal language names -> FLORES+ config names
FLORES_LANG_MAP = {
    "english":            "eng_Latn",
    "hindi":              "hin_Deva",
    "bengali":            "ben_Beng",
    "standard_arabic":    "arb_Arab",
    "german":             "deu_Latn",
    "french":             "fra_Latn",
    "spanish":            "spa_Latn",
    "japanese":           "jpn_Jpan",
    "simplified_chinese": "zho_Hans",
}

# ── CulturaX config ───────────────────────────────────────────────────────
CULTURAX_LANG_MAP = {
    "english":            ("uonlp/CulturaX", "en"),
    "hindi":              ("uonlp/CulturaX", "hi"),
    "bengali":            ("uonlp/CulturaX", "bn"),
    "standard_arabic":    ("uonlp/CulturaX", "ar"),
    "german":             ("uonlp/CulturaX", "de"),
    "french":             ("uonlp/CulturaX", "fr"),
    "spanish":            ("uonlp/CulturaX", "es"),
    "japanese":           ("uonlp/CulturaX", "ja"),
    "simplified_chinese": ("uonlp/CulturaX", "zh"),
}


# ── data structures ────────────────────────────────────────────────────────
@dataclass(order=True)
class ActivatingExample:
    activation: float
    feature_id: int = field(compare=False)
    token_id: int = field(compare=False)
    token_str: str = field(compare=False)
    token_pos: int = field(compare=False)
    context_tokens: List[str] = field(compare=False, default_factory=list)
    context_token_ids: List[int] = field(compare=False, default_factory=list)
    context_activations: List[float] = field(compare=False, default_factory=list)
    peak_idx_in_context: int = field(compare=False, default=0)
    language: str = field(compare=False, default="")
    source: str = field(compare=False, default="")        # corpus tier tag
    sentence_id: Optional[str] = field(compare=False, default=None)  # FLORES+ alignment id

    def to_dict(self) -> dict:
        d = {
            "activation": round(self.activation, 6),
            "feature_id": self.feature_id,
            "token": self.token_str,
            "token_id": self.token_id,
            "token_pos": self.token_pos,
            "context_tokens": self.context_tokens,
            "context_activations": [round(a, 4) for a in self.context_activations],
            "peak_idx_in_context": self.peak_idx_in_context,
            "language": self.language,
            "source": self.source,
        }
        if self.sentence_id is not None:
            d["sentence_id"] = self.sentence_id
        return d


class FeatureTracker:
    def __init__(self, feature_id: int, top_k: int):
        self.feature_id = feature_id
        self.top_k = top_k
        self.heap: List[ActivatingExample] = []
        self.total_activations = 0
        self.nonzero_count = 0
        self.sum_act = 0.0
        self.sum_act_sq = 0.0
        self.max_act = 0.0
        self.random_samples: List[float] = []
        # per-language firing counts (for the paper)
        self.lang_nonzero: Dict[str, int] = {}
        self.lang_total: Dict[str, int] = {}

    @property
    def min_activation(self) -> float:
        return self.heap[0].activation if len(self.heap) >= self.top_k else -1.0

    def maybe_add(self, ex: ActivatingExample):
        if len(self.heap) < self.top_k:
            heapq.heappush(self.heap, ex)
        elif ex.activation > self.heap[0].activation:
            heapq.heapreplace(self.heap, ex)

    def update_stats(self, acts: np.ndarray, lang: str):
        n = len(acts)
        self.total_activations += n
        nz = int((acts > 0).sum())
        self.nonzero_count += nz
        self.sum_act += float(acts.sum())
        self.sum_act_sq += float((acts ** 2).sum())
        if n > 0:
            self.max_act = max(self.max_act, float(acts.max()))
        # per-lang
        self.lang_total[lang] = self.lang_total.get(lang, 0) + n
        self.lang_nonzero[lang] = self.lang_nonzero.get(lang, 0) + nz
        # reservoir sampling
        for v in acts:
            if len(self.random_samples) < RANDOM_SAMPLE_SIZE:
                self.random_samples.append(float(v))
            else:
                j = np.random.randint(0, self.total_activations)
                if j < RANDOM_SAMPLE_SIZE:
                    self.random_samples[j] = float(v)

    def get_sorted_examples(self) -> List[dict]:
        return [e.to_dict() for e in sorted(self.heap, reverse=True)]

    def get_stats(self) -> dict:
        t = max(self.total_activations, 1)
        mean = self.sum_act / t
        var = max((self.sum_act_sq / t) - mean ** 2, 0)
        per_lang_rate = {}
        for lang in self.lang_total:
            lt = self.lang_total[lang]
            ln = self.lang_nonzero.get(lang, 0)
            per_lang_rate[lang] = round(ln / max(lt, 1), 6)
        return {
            "feature_id": self.feature_id,
            "total_tokens_seen": self.total_activations,
            "nonzero_count": self.nonzero_count,
            "firing_rate": round(self.nonzero_count / t, 6),
            "mean_activation": round(mean, 6),
            "std_activation": round(var ** 0.5, 6),
            "max_activation": round(self.max_act, 6),
            "per_lang_firing_rate": per_lang_rate,
        }


# ── corpus loaders ─────────────────────────────────────────────────────────
def load_flores_for_lang(lang: str):
    """
    Load FLORES+ parallel sentences for one language.
    Returns list of (sentence_id, text) tuples.
    """
    from datasets import load_dataset

    config = FLORES_LANG_MAP.get(lang)
    if config is None:
        print(f"    No FLORES+ config for {lang}, skipping")
        return []

    try:
        ds = load_dataset(
            "openlanguagedata/flores_plus", config,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"    Failed to load FLORES+ {config}: {e}")
        return []

    sentences = []
    for split_name in ["dev", "devtest"]:
        if split_name in ds:
            for row in ds[split_name]:
                sid = row.get("id", str(len(sentences)))
                text = row.get("text", "")
                if text:
                    sentences.append((str(sid), text))

    print(f"    FLORES+ {config}: {len(sentences)} sentences")
    return sentences


def stream_flores(lang, tokenizer, seq_len, batch_size):
    """
    Yield (input_ids, raw_ids, sentence_ids) batches from FLORES+.
    Since FLORES+ sentences are short (~20 words), we concatenate
    multiple sentences into one sequence, tracking boundaries.
    """
    sentences = load_flores_for_lang(lang)
    if not sentences:
        return

    buffer_ids = []
    buffer_sids = []  # sentence id per token
    batch_buf_ids = []
    batch_buf_sids = []

    for sid, text in sentences:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer_ids.extend(ids)
        buffer_sids.extend([sid] * len(ids))

        while len(buffer_ids) >= seq_len:
            chunk_ids = buffer_ids[:seq_len]
            chunk_sids = buffer_sids[:seq_len]
            buffer_ids = buffer_ids[seq_len:]
            buffer_sids = buffer_sids[seq_len:]
            batch_buf_ids.append(chunk_ids)
            batch_buf_sids.append(chunk_sids)

            if len(batch_buf_ids) >= batch_size:
                b_ids = batch_buf_ids[:batch_size]
                b_sids = batch_buf_sids[:batch_size]
                batch_buf_ids = batch_buf_ids[batch_size:]
                batch_buf_sids = batch_buf_sids[batch_size:]
                yield (
                    torch.tensor(b_ids, dtype=torch.long),
                    b_ids,
                    b_sids,
                )

    # flush
    if batch_buf_ids:
        pad = tokenizer.pad_token_id or 0
        mx = max(len(b) for b in batch_buf_ids)
        padded = [b + [pad] * (mx - len(b)) for b in batch_buf_ids]
        sid_padded = [s + [None] * (mx - len(s)) for s in batch_buf_sids]
        yield (
            torch.tensor(padded, dtype=torch.long),
            batch_buf_ids,
            sid_padded,
        )


def stream_culturax(lang, tokenizer, tokens_needed, seq_len, batch_size):
    """Yield (input_ids, raw_ids, None) from CulturaX / mc4."""
    from datasets import load_dataset

    dataset_name, config = CULTURAX_LANG_MAP.get(lang, ("uonlp/CulturaX", "en"))
    ds = None
    for name in [dataset_name, "mc4"]:
        try:
            print(f"    Loading {name}/{config} (streaming) ...")
            ds = load_dataset(name, config, split="train", streaming=True, trust_remote_code=True)
            break
        except Exception as e:
            print(f"    {name} failed: {e}")

    if ds is None:
        print(f"    CulturaX/mc4 unavailable for {lang}")
        return

    tokens_yielded = 0
    buf = []
    batch_buf = []

    for example in ds:
        text = example.get("text", example.get("content", ""))
        if not text or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)

        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            batch_buf.append(chunk)

            if len(batch_buf) >= batch_size:
                b = batch_buf[:batch_size]
                batch_buf = batch_buf[batch_size:]
                yield torch.tensor(b, dtype=torch.long), b, None
                tokens_yielded += len(b) * seq_len
                if tokens_yielded >= tokens_needed:
                    return

    if batch_buf:
        pad = tokenizer.pad_token_id or 0
        mx = max(len(b) for b in batch_buf)
        padded = [b + [pad] * (mx - len(b)) for b in batch_buf]
        yield torch.tensor(padded, dtype=torch.long), batch_buf, None


def stream_xsafety(lang, tokenizer, seq_len, batch_size):
    """Yield (input_ids, raw_ids, None) from xsafety, tagged harmful/benign."""
    from saefty.analysis.feature_identification import load_xsafety_data, XSAFETY_DATA_DIR

    harmful, benign = load_xsafety_data(XSAFETY_DATA_DIR, [lang])
    # tag with sub-source so we can distinguish harmful from benign downstream
    all_texts = []
    for t in harmful.get(lang, []):
        all_texts.append(("xsafety_harmful", t))
    for t in benign.get(lang, []):
        all_texts.append(("xsafety_benign", t))

    buf = []
    batch_buf = []

    for _tag, text in all_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)

        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            batch_buf.append(chunk)

            if len(batch_buf) >= batch_size:
                b = batch_buf[:batch_size]
                batch_buf = batch_buf[batch_size:]
                yield torch.tensor(b, dtype=torch.long), b, None

    if batch_buf:
        pad = tokenizer.pad_token_id or 0
        mx = max(len(b) for b in batch_buf)
        padded = [b + [pad] * (mx - len(b)) for b in batch_buf]
        yield torch.tensor(padded, dtype=torch.long), batch_buf, None


# ── core: token-level SAE activations ─────────────────────────────────────
@torch.no_grad()
def get_token_sae_activations(input_ids, model, hook_layer, sae, device):
    store = {}

    def hook_fn(module, inp, out):
        store["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = model.model.layers[hook_layer].register_forward_hook(hook_fn)
    model(input_ids.to(device))
    handle.remove()

    h = store["h"]
    B, S, D = h.shape
    flat = h.reshape(-1, D).float()

    if hasattr(sae, "encode"):
        feat = sae.encode(flat)
    else:
        feat = torch.relu((flat - sae.b_dec) @ sae.W_enc + sae.b_enc)

    return feat.reshape(B, S, -1).cpu().numpy()


# ── process one corpus tier for one language ──────────────────────────────
def process_batches(
    stream,              # generator of (input_ids, raw_ids, sentence_ids_or_None)
    source_tag: str,     # "flores", "culturax", "xsafety"
    lang: str,
    feature_ids: List[int],
    feat_arr: np.ndarray,
    trackers: Dict[int, FeatureTracker],
    model, sae, tokenizer,
    ctx_win: int,
):
    """Process all batches from a stream, updating trackers."""
    batch_count = 0
    tokens_done = 0

    for batch_data in stream:
        input_ids, raw_batch, sentence_ids_batch = batch_data
        B, S = input_ids.shape

        all_acts = get_token_sae_activations(input_ids, model, HOOK_LAYER, sae, DEVICE)
        target_acts = all_acts[:, :, feat_arr]  # [B, S, n_features]

        for fi, fid in enumerate(feature_ids):
            col = target_acts[:, :, fi]
            trackers[fid].update_stats(col.reshape(-1), lang)
            threshold = trackers[fid].min_activation

            for b in range(B):
                seq_acts = col[b]
                seq_ids = raw_batch[b] if b < len(raw_batch) else input_ids[b].tolist()
                seq_sids = (sentence_ids_batch[b]
                            if sentence_ids_batch is not None and b < len(sentence_ids_batch)
                            else None)

                above = np.where(seq_acts > threshold)[0]
                if len(above) == 0:
                    continue

                for pos in above:
                    act_val = float(seq_acts[pos])
                    cs = max(0, pos - ctx_win)
                    ce = min(S, pos + ctx_win + 1)
                    ctx_ids = seq_ids[cs:ce]
                    ctx_toks = tokenizer.convert_ids_to_tokens(ctx_ids)
                    ctx_acts = seq_acts[cs:ce].tolist()
                    peak_in = pos - cs
                    tok_str = tokenizer.convert_ids_to_tokens([seq_ids[pos]])[0]

                    sid = None
                    if seq_sids is not None and pos < len(seq_sids):
                        sid = seq_sids[pos]

                    ex = ActivatingExample(
                        activation=act_val,
                        feature_id=fid,
                        token_id=seq_ids[pos],
                        token_str=tok_str,
                        token_pos=pos,
                        context_tokens=ctx_toks,
                        context_token_ids=ctx_ids,
                        context_activations=ctx_acts,
                        peak_idx_in_context=peak_in,
                        language=lang,
                        source=source_tag,
                        sentence_id=sid,
                    )
                    trackers[fid].maybe_add(ex)

        tokens_done += B * S
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"      {tokens_done:,} tokens, {batch_count} batches")

    return tokens_done


# ── main collection ───────────────────────────────────────────────────────
def collect(
    feature_ids: List[int],
    corpus_mode: str,
    tokens_per_lang: int,
    top_k: int,
    ctx_win: int,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    print(f"Loading SAE: {CHECKPOINT_PATH}")
    sae = load_sae(CHECKPOINT_PATH, DEVICE)

    trackers = {fid: FeatureTracker(fid, top_k) for fid in feature_ids}
    feat_arr = np.array(feature_ids)
    timing = {}

    # determine which tiers to run
    tiers = []
    if corpus_mode in ("flores", "mixed"):
        tiers.append("flores")
    if corpus_mode in ("culturax", "mixed"):
        tiers.append("culturax")
    if corpus_mode in ("xsafety", "mixed"):
        tiers.append("xsafety")

    for lang in LANGUAGES:
        t0 = time.time()
        print(f"\n{'═' * 60}\n  {lang}\n{'═' * 60}")
        total_tokens = 0

        for tier in tiers:
            print(f"  ── {tier} ──")

            if tier == "flores":
                stream = stream_flores(lang, tokenizer, SEQ_LEN, BATCH_SIZE)
            elif tier == "culturax":
                stream = stream_culturax(
                    lang, tokenizer, tokens_per_lang, SEQ_LEN, BATCH_SIZE,
                )
            elif tier == "xsafety":
                stream = stream_xsafety(lang, tokenizer, SEQ_LEN, BATCH_SIZE)
            else:
                continue

            n = process_batches(
                stream, tier, lang, feature_ids, feat_arr, trackers,
                model, sae, tokenizer, ctx_win,
            )
            total_tokens += n
            print(f"    {tier}/{lang}: {n:,} tokens")

        elapsed = time.time() - t0
        timing[lang] = round(elapsed, 1)
        print(f"  {lang} total: {total_tokens:,} tokens in {elapsed:.0f}s")

    return trackers, timing


# ── dashboard formatting ──────────────────────────────────────────────────
def format_dashboard(fid, examples, stats):
    lines = [
        f"{'═' * 90}",
        f"  FEATURE {fid}",
        f"{'═' * 90}",
        f"  firing_rate={stats['firing_rate']:.4%}  "
        f"mean={stats['mean_activation']:.4f}  "
        f"max={stats['max_activation']:.4f}  "
        f"tokens_seen={stats['total_tokens_seen']:,}",
    ]
    # per-lang firing rates
    plr = stats.get("per_lang_firing_rate", {})
    if plr:
        parts = [f"{l}={r:.4%}" for l, r in sorted(plr.items())]
        lines.append(f"  per-lang firing: {', '.join(parts)}")
    lines.append(f"{'─' * 90}")

    for i, ex in enumerate(examples):
        ctx = ex["context_tokens"]
        acts = ex["context_activations"]
        peak = ex["peak_idx_in_context"]
        parts = []
        for j, (tok, act) in enumerate(zip(ctx, acts)):
            if j == peak:
                parts.append(f">>>{tok}<<<[{act:.2f}]")
            elif act > 0.01:
                parts.append(f"{tok}[{act:.2f}]")
            else:
                parts.append(tok)

        src = ex.get("source", "?")
        sid = ex.get("sentence_id", "")
        sid_str = f"  sid={sid}" if sid else ""
        lines.append(
            f"\n  #{i+1}  act={ex['activation']:.4f}  "
            f"tok='{ex['token']}'  lang={ex['language']}  src={src}{sid_str}"
        )
        cur = "    "
        for p in parts:
            if len(cur) + len(p) + 1 > 100:
                lines.append(cur)
                cur = "    " + p
            else:
                cur += " " + p
        lines.append(cur)

    lines.append(f"\n{'═' * 90}")
    return "\n".join(lines)


# ── load target features ──────────────────────────────────────────────────
def load_target_features(path: str) -> List[int]:
    with open(path) as f:
        data = json.load(f)
    ids = sorted({feat["feature_id"] for feat in data["features"]})
    print(f"Loaded {len(ids)} target features from {path}")
    return ids


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", choices=["flores", "culturax", "xsafety", "mixed"],
        default="mixed",
        help="Which corpus tier(s) to use",
    )
    parser.add_argument("--tokens-per-lang", type=int, default=TOKENS_PER_LANG,
                        help="Tokens per lang for culturax tier")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--features-path", default="results/per_lang_combined.json")
    args = parser.parse_args()

    t0 = time.time()
    feature_ids = load_target_features(args.features_path)

    trackers, timing = collect(
        feature_ids, args.corpus, args.tokens_per_lang, args.top_k, CONTEXT_WINDOW,
    )

    # ── save ──
    out_dir = Path(f"results/max_act/{args.corpus}")
    dash_dir = out_dir / "dashboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    dash_dir.mkdir(parents=True, exist_ok=True)

    top_out = {
        "metadata": {
            "model": MODEL_NAME,
            "sae_checkpoint": CHECKPOINT_PATH,
            "hook_layer": HOOK_LAYER,
            "corpus_mode": args.corpus,
            "top_k": args.top_k,
            "context_window": CONTEXT_WINDOW,
            "tokens_per_lang": args.tokens_per_lang,
            "languages": LANGUAGES,
            "timing": timing,
        },
        "features": [],
    }
    stats_out = {"metadata": {"model": MODEL_NAME, "corpus": args.corpus}, "features": []}

    for fid in feature_ids:
        t = trackers[fid]
        examples = t.get_sorted_examples()
        stats = t.get_stats()
        top_out["features"].append({"feature_id": fid, "top_examples": examples})
        stats_out["features"].append(stats)
        (dash_dir / f"feat_{fid}.txt").write_text(format_dashboard(fid, examples, stats))

    top_path = out_dir / "top_activations.json"
    stats_path = out_dir / "activation_stats.json"

    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(top_out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {top_path}")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {stats_path}")

    # ── FLORES+ cross-lingual alignment analysis ──────────────────────────
    # for features collected from flores, check if the same sentence_id
    # appears across multiple languages (= same semantic content)
    if args.corpus in ("flores", "mixed"):
        print(f"\n{'─' * 60}")
        print(f"  FLORES+ CROSS-LINGUAL ALIGNMENT CHECK")
        print(f"{'─' * 60}")
        for fid in feature_ids[:10]:
            exs = trackers[fid].get_sorted_examples()
            flores_exs = [e for e in exs if e.get("source") == "flores"]
            if not flores_exs:
                continue
            # group by sentence_id
            by_sid = {}
            for e in flores_exs:
                sid = e.get("sentence_id")
                if sid:
                    by_sid.setdefault(sid, []).append(e)
            # find sentence_ids that appear in multiple languages
            cross = {
                sid: [e["language"] for e in es]
                for sid, es in by_sid.items()
                if len(set(e["language"] for e in es)) > 1
            }
            if cross:
                print(f"  feat {fid}: {len(cross)} sentence(s) fire across languages")
                for sid, langs in list(cross.items())[:3]:
                    print(f"    sid={sid} -> {', '.join(langs)}")
            else:
                print(f"  feat {fid}: no cross-lingual sentence overlap in top-{args.top_k}")

    # ── summary table ──
    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  corpus={args.corpus}  features={len(feature_ids)}  top_k={args.top_k}")
    print(f"  wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'─' * 60}")
    print(f"  {'feat':>8} | {'token':<20} | {'act':>8} | {'lang':<15} | {'src':<10}")
    print(f"{'─' * 60}")
    for fid in feature_ids[:20]:
        exs = trackers[fid].get_sorted_examples()
        if exs:
            e = exs[0]
            print(f"  {fid:>8} | {e['token']:<20.20} | "
                  f"{e['activation']:>8.3f} | {e['language']:<15} | {e.get('source', '?'):<10}")


if __name__ == "__main__":
    main()