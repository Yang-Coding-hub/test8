import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import csv
import json
import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.configs import Config
from scripts.eval.evaluate_safety import load_defense_system
from src.core.metrics import SafetyUtilityEvaluator
from src.core.paths import (
    ADV_BENCH_CSV,
    HARMFUL_BEHAVIORS_CSV,
    JAILBREAKBENCH_CSV,
    MALICIOUSINSTRUCT_CSV,
    CRESCENDO_JSON,
    MULTIBREAK_JSONL,
    MT_BENCH_JSONL,
    GSM8K_JSONL,
    HUMANEVAL_JSONL,
    MMLU_DIR,
    BENCHMARK_REPORT,
    BENCHMARK_REPORT_BASELINE,
    LOCAL_MIRROR_ADV_BENCH,
    LOCAL_MIRROR_JBB,
    LOCAL_MIRROR_CRESCENDO,
    LOCAL_MIRROR_MULTIBREAK,
    LOCAL_MIRROR_MT_BENCH,
    LOCAL_MIRROR_GSM8K,
)


ADV_BENCH_PATHS = [
    str(ADV_BENCH_CSV),
    str(HARMFUL_BEHAVIORS_CSV),
    str(LOCAL_MIRROR_ADV_BENCH),
]

JBB_PATHS = [
    str(JAILBREAKBENCH_CSV),
    str(LOCAL_MIRROR_JBB),
]

MALICIOUS_INSTRUCT_PATHS = [
    str(MALICIOUSINSTRUCT_CSV),
]

CRESCENDO_PATHS = [
    str(CRESCENDO_JSON),
    str(LOCAL_MIRROR_CRESCENDO),
]

MULTIBREAK_PATHS = [
    str(MULTIBREAK_JSONL),
    str(LOCAL_MIRROR_MULTIBREAK),
]

MT_BENCH_PATHS = [
    str(MT_BENCH_JSONL),
    str(LOCAL_MIRROR_MT_BENCH),
]

GSM8K_PATHS = [
    str(GSM8K_JSONL),
    str(LOCAL_MIRROR_GSM8K),
]

HUMANEVAL_PATHS = [
    str(HUMANEVAL_JSONL),
]

MMLU_ROOT = str(MMLU_DIR)

RESULT_JSON = str(BENCHMARK_REPORT)
RESULT_JSON_BASELINE = str(BENCHMARK_REPORT_BASELINE)


class PlainLLMSystem:
    """Baseline generator without CIDT/SAE intervention."""

    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer

    def eval(self):
        self.base_model.eval()
        return self

    def multi_turn_generate(
        self,
        session_ids,
        session_masks,
        lengths,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,
        allow_rewrite=True,
    ):
        batch_size = session_ids.size(0)
        last_idx = torch.arange(batch_size, device=session_ids.device)
        curr_input_ids = session_ids[last_idx, lengths - 1]
        curr_masks = session_masks[last_idx, lengths - 1]

        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids=curr_input_ids.to(self.base_model.device),
                attention_mask=curr_masks.to(self.base_model.device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = curr_input_ids.shape[1]
        generated_ids = outputs[:, prompt_len:]
        return [x.strip() for x in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def load_baseline_system():
    print("Loading baseline (no-defense) system...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        device_map="auto",
        torch_dtype=Config.DTYPE,
        attn_implementation="eager",
    )
    plain_system = PlainLLMSystem(base_model, tokenizer)
    return plain_system, base_model, tokenizer


@dataclass
class EvalConfig:
    mmlu_subjects: List[str]
    mmlu_per_subject: int = 200
    gsm8k_limit: int = 1319
    mt_bench_limit: int = 80
    safety_limit: Optional[int] = None


def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def build_session_tensors(tokenizer, turns: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = []
    masks = []
    for t in turns:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        ids.append(enc["input_ids"].squeeze(0))
        masks.append(enc["attention_mask"].squeeze(0))

    session_ids = torch.stack(ids, dim=0).unsqueeze(0)
    session_masks = torch.stack(masks, dim=0).unsqueeze(0)
    lengths = torch.tensor([len(turns)], dtype=torch.long)
    return session_ids, session_masks, lengths


def safe_generate(safe_system, tokenizer, turns: List[str]) -> str:
    session_ids, session_masks, lengths = build_session_tensors(tokenizer, turns)
    with torch.no_grad():
        out = safe_system.multi_turn_generate(
            session_ids.to(Config.DEVICE),
            session_masks.to(Config.DEVICE),
            lengths.to(Config.DEVICE),
            max_new_tokens=128,
        )
    return out[0].strip()


def safe_generate_batch_single_turn(safe_system, tokenizer, prompts: List[str], max_new_tokens: int = 64) -> List[str]:
    if not prompts:
        return []
    enc = tokenizer(
        prompts,
        truncation=True,
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    # batch, turns=1, seq
    session_ids = enc["input_ids"].unsqueeze(1)
    session_masks = enc["attention_mask"].unsqueeze(1)
    lengths = torch.ones(session_ids.size(0), dtype=torch.long)
    with torch.no_grad():
        out = safe_system.multi_turn_generate(
            session_ids.to(Config.DEVICE),
            session_masks.to(Config.DEVICE),
            lengths.to(Config.DEVICE),
            max_new_tokens=max_new_tokens,
        )
    return [x.strip() for x in out]


def parse_adv_csv(path: str, limit: Optional[int]) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goal = row.get("Goal") or row.get("goal") or row.get("prompt")
            if goal:
                prompts.append(goal.strip())
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def parse_jsonl(path: str, limit: Optional[int]) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_json_array(path: str, limit: Optional[int]) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    if limit is None:
        return data
    return data[:limit]


def eval_single_turn_safety(safe_system, judge_model, tokenizer, prompts: List[str], batch_size: int = 8) -> Dict[str, float]:
    preds = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        preds.extend(safe_generate_batch_single_turn(safe_system, tokenizer, chunk, max_new_tokens=64))
    asr = SafetyUtilityEvaluator.compute_asr(preds, judge_model, tokenizer)
    hri = SafetyUtilityEvaluator.compute_hri(preds, judge_model, tokenizer)
    return {"count": len(prompts), "asr": asr, "hri": hri}


def parse_multiturn_rows(rows: List[dict]) -> List[List[str]]:
    sessions = []
    for r in rows:
        if "turns" in r and isinstance(r["turns"], list):
            t = r["turns"]
            if len(t) == 0:
                continue
            if isinstance(t[0], dict):
                user_turns = [x.get("content", "").strip() for x in t if x.get("role") == "user"]
                user_turns = [x for x in user_turns if x]
                if user_turns:
                    sessions.append(user_turns)
            elif isinstance(t[0], str):
                clean = [x.strip() for x in t if isinstance(x, str) and x.strip()]
                if clean:
                    sessions.append(clean)
    return sessions


def eval_multi_turn_safety(safe_system, judge_model, tokenizer, sessions: List[List[str]]) -> Dict[str, float]:
    preds = []
    for s in sessions:
        session_ids, session_masks, lengths = build_session_tensors(tokenizer, s)
        with torch.no_grad():
            resp = safe_system.multi_turn_generate(
                session_ids.to(Config.DEVICE),
                session_masks.to(Config.DEVICE),
                lengths.to(Config.DEVICE),
                max_new_tokens=128,
            )
        preds.append(resp[0].strip())
    asr = SafetyUtilityEvaluator.compute_asr(preds, judge_model, tokenizer)
    hri = SafetyUtilityEvaluator.compute_hri(preds, judge_model, tokenizer)
    return {"count": len(sessions), "asr": asr, "hri": hri}


def format_mmlu_prompt(row):
    q = row["question"]
    c = row["choices"]
    return (
        f"Question: {q}\n"
        f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n\n"
        "Task: Output ONLY the single letter of the correct answer (A, B, C, or D). "
        "DO NOT explain. DO NOT output any other words."
    )


def eval_mmlu(
    safe_system,
    tokenizer,
    subjects: List[str],
    per_subject: int,
    deterministic_no_rewrite: bool = False,
) -> Dict[str, object]:
    subject_scores = {}
    total_correct = 0
    total_count = 0

    for subj in subjects:
        file_path = os.path.join(MMLU_ROOT, subj, "test-00000-of-00001.parquet")
        if not os.path.exists(file_path):
            subject_scores[subj] = {"count": 0, "acc": None, "missing": True}
            continue

        df = pd.read_parquet(file_path).head(per_subject)
        correct = 0
        for _, row in df.iterrows():
            prompt = format_mmlu_prompt(row)
            gt = chr(ord("A") + int(row["answer"]))
            session_ids, session_masks, lengths = build_session_tensors(tokenizer, [prompt])
            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": 8}
                if deterministic_no_rewrite:
                    gen_kwargs.update(
                        {
                            "do_sample": False,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                            "allow_rewrite": False,
                        }
                    )
                pred = safe_system.multi_turn_generate(
                    session_ids.to(Config.DEVICE),
                    session_masks.to(Config.DEVICE),
                    lengths.to(Config.DEVICE),
                    **gen_kwargs,
                )[0].strip().upper()
            m = re.search(r"\b([A-D])\b", pred.replace(".", " "))
            extracted = m.group(1) if m else pred[:2]
            if gt in extracted:
                correct += 1

        n = len(df)
        acc = correct / max(1, n)
        subject_scores[subj] = {"count": n, "acc": acc}
        total_correct += correct
        total_count += n

    return {
        "subjects": subject_scores,
        "total_count": total_count,
        "overall_acc": total_correct / max(1, total_count),
    }


def extract_final_number(text: str) -> Optional[str]:
    text = text.strip()
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def eval_gsm8k(safe_system, tokenizer, gsm8k_path: str, limit: int) -> Dict[str, float]:
    rows = parse_jsonl(gsm8k_path, limit)
    correct = 0

    for r in rows:
        q = r.get("question", "").strip()
        ans = r.get("answer", "")
        gt = extract_final_number(ans)
        if not q or gt is None:
            continue

        prompt = (
            "Solve the math word problem carefully. "
            "Output exactly one final line in the format '#### number' and nothing after it.\n\n"
            f"Question: {q}"
        )
        session_ids, session_masks, lengths = build_session_tensors(tokenizer, [prompt])
        with torch.no_grad():
            pred = safe_system.multi_turn_generate(
                session_ids.to(Config.DEVICE),
                session_masks.to(Config.DEVICE),
                lengths.to(Config.DEVICE),
                max_new_tokens=96,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                allow_rewrite=False,
            )[0]
        pd_num = extract_final_number(pred)
        if pd_num == gt:
            correct += 1

    used = len(rows)
    return {"count": used, "acc": correct / max(1, used)}


def score_mt_bench_reply(judge_model, tokenizer, user_turns: List[str], model_reply: str) -> float:
    joined = "\n".join([f"Turn {i+1}: {t}" for i, t in enumerate(user_turns)])
    prompt = (
        "You are grading an assistant response for instruction-following and coherence in multi-turn dialog. "
        "Give a single integer score from 1 to 10 (10 best). Output only the integer.\n\n"
        f"Dialog:\n{joined}\n\n"
        f"Assistant final reply:\n{model_reply}\n\n"
        "Score:"
    )
    raw = SafetyUtilityEvaluator._get_judge_response(prompt, judge_model, tokenizer)
    m = re.search(r"\d+", raw)
    if not m:
        return 1.0
    v = float(m.group(0))
    return min(max(v, 1.0), 10.0)


def eval_mt_bench(safe_system, judge_model, tokenizer, mt_path: str, limit: int) -> Dict[str, float]:
    rows = parse_jsonl(mt_path, limit)
    scores = []

    for r in rows:
        turns = r.get("turns", [])
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        user_turns = [x.strip() for x in turns if isinstance(x, str) and x.strip()]
        if len(user_turns) < 2:
            continue

        # Evaluate final-turn response conditioned on the full multi-turn user history.
        session_ids, session_masks, lengths = build_session_tensors(tokenizer, user_turns)
        with torch.no_grad():
            reply = safe_system.multi_turn_generate(
                session_ids.to(Config.DEVICE),
                session_masks.to(Config.DEVICE),
                lengths.to(Config.DEVICE),
                max_new_tokens=128,
            )[0]
        score = score_mt_bench_reply(judge_model, tokenizer, user_turns, reply)
        scores.append(score)

    return {"count": len(scores), "avg_score_10": sum(scores) / max(1, len(scores))}


def main():
    parser = argparse.ArgumentParser(description="Run full safety and utility benchmark suite")
    parser.add_argument("--mmlu-per-subject", type=int, default=200)
    parser.add_argument("--gsm8k-limit", type=int, default=1319)
    parser.add_argument("--mt-bench-limit", type=int, default=80)
    parser.add_argument("--safety-limit", type=int, default=0, help="0 means use full dataset")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test with very small limits")
    parser.add_argument("--baseline", action="store_true", help="Run no-defense baseline with same config")
    parser.add_argument("--output", type=str, default=None, help="Optional output json path")
    parser.add_argument(
        "--mmlu-deterministic-no-rewrite",
        action="store_true",
        help="Use deterministic decoding and disable rewrite during MMLU eval (ablation mode)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "classic"],
        help="Dataset profile: full=all available, classic=representative subset",
    )
    args = parser.parse_args()

    if args.quick:
        mmlu_per_subject = 5
        gsm8k_limit = 20
        mt_bench_limit = 10
        safety_limit = 20
    else:
        mmlu_per_subject = args.mmlu_per_subject
        gsm8k_limit = args.gsm8k_limit
        mt_bench_limit = args.mt_bench_limit
        safety_limit = None if args.safety_limit == 0 else args.safety_limit

    cfg = EvalConfig(
        mmlu_subjects=[
            "college_computer_science",
            "high_school_mathematics",
            "professional_law",
            "high_school_physics",
            "college_medicine",
        ],
        mmlu_per_subject=mmlu_per_subject,
        gsm8k_limit=gsm8k_limit,
        mt_bench_limit=mt_bench_limit,
        safety_limit=safety_limit,
    )

    if args.baseline:
        safe_system, judge_model, tokenizer = load_baseline_system()
    else:
        print("Loading defended system...")
        safe_system, judge_model, tokenizer = load_defense_system()
    safe_system.eval()

    report: Dict[str, object] = {
        "data_paths": {},
        "safety_single_turn": {},
        "safety_multi_turn": {},
        "utility": {},
        "notes": [],
    }

    use_classic = args.profile == "classic"

    # Single-turn safety
    adv_path = first_existing(ADV_BENCH_PATHS)
    if adv_path:
        adv_prompts = parse_adv_csv(adv_path, cfg.safety_limit)
        report["data_paths"]["advbench"] = adv_path
        report["safety_single_turn"]["advbench"] = eval_single_turn_safety(
            safe_system, judge_model, tokenizer, adv_prompts
        )
    else:
        report["notes"].append("AdvBench not found locally.")

    jbb_path = first_existing(JBB_PATHS)
    if jbb_path:
        jbb_prompts = parse_adv_csv(jbb_path, cfg.safety_limit)
        report["data_paths"]["jailbreakbench"] = jbb_path
        report["safety_single_turn"]["jailbreakbench"] = eval_single_turn_safety(
            safe_system, judge_model, tokenizer, jbb_prompts
        )
    else:
        report["notes"].append("JailbreakBench csv not found locally.")

    if not use_classic:
        mi_path = first_existing(MALICIOUS_INSTRUCT_PATHS)
        if mi_path:
            mi_prompts = parse_adv_csv(mi_path, cfg.safety_limit)
            report["data_paths"]["maliciousinstruct"] = mi_path
            report["safety_single_turn"]["maliciousinstruct"] = eval_single_turn_safety(
                safe_system, judge_model, tokenizer, mi_prompts
            )
        else:
            report["notes"].append("MaliciousInstruct csv not found locally.")

    # Multi-turn safety
    crescendo_path = first_existing(CRESCENDO_PATHS)
    if crescendo_path:
        rows = parse_jsonl(crescendo_path, cfg.safety_limit) if crescendo_path.endswith(".jsonl") else parse_json_array(crescendo_path, cfg.safety_limit)
        sessions = parse_multiturn_rows(rows)
        report["data_paths"]["crescendo"] = crescendo_path
        report["safety_multi_turn"]["crescendo"] = eval_multi_turn_safety(
            safe_system, judge_model, tokenizer, sessions
        )
    else:
        report["notes"].append("Crescendo dataset not found locally.")

    if not use_classic:
        multibreak_path = first_existing(MULTIBREAK_PATHS)
        if multibreak_path:
            rows = parse_jsonl(multibreak_path, cfg.safety_limit)
            sessions = parse_multiturn_rows(rows)
            report["data_paths"]["multibreak"] = multibreak_path
            report["safety_multi_turn"]["multibreak"] = eval_multi_turn_safety(
                safe_system, judge_model, tokenizer, sessions
            )
        else:
            report["notes"].append("MultiBreak dataset not found locally; skipped.")

    # Utility
    try:
        report["utility"]["mmlu"] = eval_mmlu(
            safe_system,
            tokenizer,
            cfg.mmlu_subjects,
            cfg.mmlu_per_subject,
            deterministic_no_rewrite=args.mmlu_deterministic_no_rewrite,
        )
    except ImportError as e:
        report["utility"]["mmlu"] = {
            "status": "skipped",
            "reason": str(e),
        }
        report["notes"].append("MMLU skipped due to missing parquet dependency (pyarrow/fastparquet).")

    gsm8k_path = first_existing(GSM8K_PATHS)
    if gsm8k_path:
        report["data_paths"]["gsm8k"] = gsm8k_path
        report["utility"]["gsm8k"] = eval_gsm8k(
            safe_system, tokenizer, gsm8k_path, cfg.gsm8k_limit
        )
    else:
        report["notes"].append("GSM8K jsonl not found locally.")

    mt_path = first_existing(MT_BENCH_PATHS)
    if mt_path:
        report["data_paths"]["mt_bench"] = mt_path
        report["utility"]["mt_bench"] = eval_mt_bench(
            safe_system, judge_model, tokenizer, mt_path, cfg.mt_bench_limit
        )
    else:
        report["notes"].append("MT-Bench prompts file not found locally.")

    humaneval_path = first_existing(HUMANEVAL_PATHS)
    if humaneval_path:
        report["data_paths"]["humaneval"] = humaneval_path
        report["utility"]["humaneval"] = {
            "count": len(parse_jsonl(humaneval_path, None)),
            "status": "downloaded",
            "note": "pass@1 evaluation not wired yet in this script",
        }
    else:
        report["notes"].append("HumanEval jsonl not found locally.")

    result_path = args.output if args.output else (RESULT_JSON_BASELINE if args.baseline else RESULT_JSON)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Full benchmark completed")
    print(f"Report saved to: {result_path}")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
