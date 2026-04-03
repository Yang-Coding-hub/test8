import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import random
import re
from typing import Iterable, List

import pandas as pd

from src.core.paths import (
    ADV_BENCH_CSV,
    JAILBREAKBENCH_CSV,
    MALICIOUSINSTRUCT_CSV,
    MULTIBREAK_JSONL,
    CRESCENDO_JSON,
    GSM8K_JSONL,
    MT_BENCH_JSONL,
    MMLU_DIR,
    FITD_TRAIN_DATA,
    SAE_MALICIOUS_PROMPTS_JSON,
    SAE_BENIGN_PROMPTS_JSON,
    DATA_DIR,
)


def _clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        c = _clean_text(x)
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _read_csv_prompts(path: Path) -> List[str]:
    if not path.exists():
        return []
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ("goal", "Goal", "prompt", "text"):
                if k in row and row[k]:
                    prompts.append(row[k])
                    break
    return prompts


def _read_jsonl_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_json_array(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _extract_multiturn_last_user_prompt(rows: List[dict]) -> List[str]:
    prompts = []
    for r in rows:
        turns = r.get("turns", r.get("dialogue", []))
        if not isinstance(turns, list) or not turns:
            continue
        if isinstance(turns[0], str):
            clean = [_clean_text(t) for t in turns if isinstance(t, str) and _clean_text(t)]
            if clean:
                prompts.append(clean[-1])
        elif isinstance(turns[0], dict):
            users = []
            for t in turns:
                if t.get("role") == "user" and t.get("content"):
                    users.append(_clean_text(t["content"]))
            if users:
                prompts.append(users[-1])
    return prompts


def _extract_fitd_last_turn(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        turns = r.get("turns", [])
        labels = r.get("labels", [])
        if not turns or not labels or len(turns) != len(labels):
            continue
        for i in range(len(turns) - 1, -1, -1):
            if labels[i] == 1:
                out.append(turns[i])
                break
    return out


def _build_lexical_matched_benign(prompts: List[str], seed: int) -> List[str]:
    random.seed(seed)
    templates = [
        "For cybersecurity defense training, explain how to detect and prevent the following risk safely: {x}",
        "For legal and ethical education, summarize why this request is harmful and provide safe alternatives: {x}",
        "For incident response simulation, list warning signs and mitigation strategies related to: {x}",
    ]
    out = []
    for p in prompts:
        tpl = random.choice(templates)
        out.append(tpl.format(x=p))
    return out


def _sample_mmlu_prompts(max_per_subject: int, seed: int) -> List[str]:
    subjects = [
        "high_school_mathematics",
        "college_computer_science",
        "high_school_physics",
        "professional_law",
        "college_medicine",
    ]
    rng = random.Random(seed)
    prompts = []
    for s in subjects:
        p = MMLU_DIR / s / "test-00000-of-00001.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        idx = list(range(len(df)))
        rng.shuffle(idx)
        idx = idx[: max_per_subject]
        for i in idx:
            row = df.iloc[i]
            q = (
                f"Question: {row['question']}\n"
                f"A. {row['choices'][0]}\nB. {row['choices'][1]}\n"
                f"C. {row['choices'][2]}\nD. {row['choices'][3]}"
            )
            prompts.append(q)
    return prompts


def _sample_gsm8k_prompts(limit: int) -> List[str]:
    rows = _read_jsonl_rows(GSM8K_JSONL)
    out = []
    for r in rows[:limit]:
        q = _clean_text(r.get("question", ""))
        if q:
            out.append(f"Solve this math problem step by step: {q}")
    return out


def _sample_mt_bench_prompts(limit: int) -> List[str]:
    rows = _read_jsonl_rows(MT_BENCH_JSONL)
    out = []
    for r in rows[:limit]:
        turns = r.get("turns", [])
        if isinstance(turns, list):
            for t in turns:
                if isinstance(t, str) and _clean_text(t):
                    out.append(_clean_text(t))
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare stronger benign/malicious corpora for SAE feature isolation")
    parser.add_argument("--malicious-target", type=int, default=4000)
    parser.add_argument("--benign-target", type=int, default=5000)
    parser.add_argument("--mmlu-per-subject", type=int, default=500)
    parser.add_argument("--gsm8k-limit", type=int, default=1200)
    parser.add_argument("--mt-bench-limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build malicious pool from single-turn + multi-turn + FITD success turns.
    malicious = []
    malicious.extend(_read_csv_prompts(ADV_BENCH_CSV))
    malicious.extend(_read_csv_prompts(JAILBREAKBENCH_CSV))
    malicious.extend(_read_csv_prompts(MALICIOUSINSTRUCT_CSV))
    malicious.extend(_extract_multiturn_last_user_prompt(_read_jsonl_rows(MULTIBREAK_JSONL)))
    malicious.extend(_extract_multiturn_last_user_prompt(_read_json_array(CRESCENDO_JSON)))
    malicious.extend(_extract_fitd_last_turn(FITD_TRAIN_DATA))
    malicious = _dedup_keep_order(malicious)

    if len(malicious) > args.malicious_target:
        random.Random(args.seed).shuffle(malicious)
        malicious = malicious[: args.malicious_target]

    # Build benign pool with lexical-matched controls + hard reasoning + normal dialog.
    benign = []
    benign.extend(_build_lexical_matched_benign(malicious, seed=args.seed))

    if SAE_BENIGN_PROMPTS_JSON.exists():
        with open(SAE_BENIGN_PROMPTS_JSON, "r", encoding="utf-8") as f:
            try:
                benign.extend(json.load(f))
            except Exception:
                pass

    benign.extend(_sample_mmlu_prompts(args.mmlu_per_subject, seed=args.seed))
    benign.extend(_sample_gsm8k_prompts(args.gsm8k_limit))
    benign.extend(_sample_mt_bench_prompts(args.mt_bench_limit))
    benign = _dedup_keep_order(benign)

    if len(benign) > args.benign_target:
        random.Random(args.seed).shuffle(benign)
        benign = benign[: args.benign_target]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAE_MALICIOUS_PROMPTS_JSON, "w", encoding="utf-8") as f:
        json.dump(malicious, f, ensure_ascii=False, indent=2)
    with open(SAE_BENIGN_PROMPTS_JSON, "w", encoding="utf-8") as f:
        json.dump(benign, f, ensure_ascii=False, indent=2)

    manifest = {
        "malicious_count": len(malicious),
        "benign_count": len(benign),
        "malicious_target": args.malicious_target,
        "benign_target": args.benign_target,
        "sources": {
            "single_turn": [str(ADV_BENCH_CSV), str(JAILBREAKBENCH_CSV), str(MALICIOUSINSTRUCT_CSV)],
            "multi_turn": [str(MULTIBREAK_JSONL), str(CRESCENDO_JSON)],
            "fitd": str(FITD_TRAIN_DATA),
            "benign_existing": str(SAE_BENIGN_PROMPTS_JSON),
            "mmlu": str(MMLU_DIR),
            "gsm8k": str(GSM8K_JSONL),
            "mt_bench": str(MT_BENCH_JSONL),
        },
    }
    manifest_path = DATA_DIR / "sae_feature_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("SAE feature corpora prepared")
    print(f"malicious: {len(malicious)}")
    print(f"benign: {len(benign)}")
    print(f"saved: {SAE_MALICIOUS_PROMPTS_JSON}")
    print(f"saved: {SAE_BENIGN_PROMPTS_JSON}")
    print(f"manifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
