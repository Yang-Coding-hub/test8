import os
import json
import random
import logging
import ast
import pandas as pd
from pathlib import Path
from datasets import load_dataset, load_from_disk
from src.core.paths import (
    DATA_DIR,
    ADV_BENCH_CSV,
    JAILBREAKBENCH_CSV,
    MALICIOUSINSTRUCT_CSV,
    MULTIBREAK_JSONL,
    CRESCENDO_JSON,
    MMLU_DIR,
    GSM8K_JSONL,
    MT_BENCH_JSONL,
    HUMANEVAL_JSONL,
    EXTERNAL_CRESCENDO_PROMPTS,
    LOCAL_MIRROR_ADV_BENCH,
    LOCAL_MIRROR_JBB,
    LOCAL_MIRROR_MT_BENCH,
    LOCAL_MIRROR_CRESCENDO,
)

# 配置独立的日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkLoader")


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _normalize_turns(turns):
    if not turns:
        return []
    if isinstance(turns[0], str):
        return [t.strip() for t in turns if isinstance(t, str) and t.strip()]
    if isinstance(turns[0], dict):
        user_turns = []
        for x in turns:
            if x.get("role") == "user" and x.get("content"):
                user_turns.append(str(x["content"]).strip())
        return [t for t in user_turns if t]
    return []

class BenchmarkLoader:
    # ==========================================
    # 1. Safety - 单轮攻击 (Single-turn)
    # ==========================================
    @staticmethod
    def load_advbench(filepath=str(ADV_BENCH_CSV)):
        """
        加载 AdvBench 核心有害行为 (520条)
        格式预期: CSV 文件，包含 'goal' 列。
        """
        logger.info(f"Loading AdvBench from {filepath}...")
        try:
            df = pd.read_csv(filepath)
            # 兼容不同版本列名
            col_name = None
            for c in ["goal", "Goal", "prompt", "text"]:
                if c in df.columns:
                    col_name = c
                    break
            if col_name is None:
                raise ValueError(f"No prompt column found in AdvBench csv. columns={list(df.columns)}")
            prompts = df[col_name].tolist()[:520]
            # 单轮攻击，Label 统一设为 1 (代表 Malicious)
            return [[p] for p in prompts], [1] * len(prompts)
        except Exception as e:
            logger.error(f"❌ Failed to load AdvBench: {e}")
            return [], []

    @staticmethod
    def load_jailbreakbench(filepath=str(JAILBREAKBENCH_CSV)):
        """
        加载 JailbreakBench (JBB-Behaviors) 标准测试集
        直接通过 Hugging Face datasets 拉取官方数据集。
        """
        logger.info("Loading JailbreakBench...")
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                col_name = "Goal" if "Goal" in df.columns else "goal"
                prompts = df[col_name].tolist()
            else:
                ds = load_dataset("JailbreakBench/JBB-Behaviors", split="train")
                prompts = ds["Goal"]
            return [[p] for p in prompts], [1] * len(prompts)
        except Exception as e:
            logger.error(f"❌ Failed to load JailbreakBench: {e}")
            return [], []

    @staticmethod
    def load_maliciousinstruct(filepath=str(MALICIOUSINSTRUCT_CSV)):
        """
        加载 MaliciousInstruct (100条高危指令)
        格式预期: CSV，包含 prompt 列。
        """
        logger.info(f"Loading MaliciousInstruct from {filepath}...")
        try:
            df = pd.read_csv(filepath)
            col = "prompt" if "prompt" in df.columns else ("Goal" if "Goal" in df.columns else None)
            if col is None:
                raise ValueError(f"No prompt-like column in maliciousinstruct csv. columns={list(df.columns)}")
            prompts = df[col].astype(str).tolist()
            return [[p] for p in prompts], [1] * len(prompts)
        except Exception as e:
            logger.error(f"❌ Failed to load MaliciousInstruct: {e}")
            return [], []

    # ==========================================
    # 2. Safety - 多轮/复杂攻击 (Multi-turn)
    # ==========================================
    @staticmethod
    def load_multibreak(filepath=str(MULTIBREAK_JSONL)):
        """
        加载 MultiBreak 多轮越狱数据集
        格式预期: JSONL，每行包含 'dialogue' 或 'turns' 列表 (List[str])。
        """
        logger.info(f"Loading MultiBreak from {filepath}...")
        histories, labels = [], []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    turns = data.get('turns', data.get('dialogue', []))
                    turns = _normalize_turns(turns)
                    if turns:
                        histories.append(turns)
                        labels.append(1) # 1 代表有害对话序列
            return histories, labels
        except Exception as e:
            logger.error(f"❌ Failed to load MultiBreak: {e}")
            return [], []

    @staticmethod
    def load_crescendo(filepath=str(CRESCENDO_JSON)):
        """
        加载 Crescendo 多轮诱导测试集
        格式预期: JSON 列表，每个元素包含多轮对话 prompt。
        """
        logger.info(f"Loading Crescendo from {filepath}...")
        histories, labels = [], []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # 假设 Crescendo 数据格式包含 'turns' 键
                    if 'turns' in item:
                        turns = _normalize_turns(item['turns'])
                        if turns:
                            histories.append(turns)
                            labels.append(1)
            return histories, labels
        except Exception as e:
            logger.error(f"❌ Failed to load Crescendo: {e}")
            return [], []

    # ==========================================
    # 3. Utility - 通用能力 (Lossless Proof)
    # ==========================================
    @staticmethod
    def load_mmlu(
        base_dir=str(MMLU_DIR), 
        subjects=['computer_security', 'college_mathematics', 'machine_learning', 'college_physics', 'jurisprudence'], 
        samples_per_subject=200
    ):
        """
        加载本地 MMLU 核心学科测试集 (适配你本地的离线文件夹结构)
        """
        logger.info(f"Loading local MMLU from {base_dir} for subjects: {subjects}...")
        histories, labels = [], []
        choices_map = ['A', 'B', 'C', 'D']
        
        for sub in subjects:
            subject_path = os.path.join(base_dir, sub)
            if not os.path.exists(subject_path):
                logger.warning(f"⚠️ Subject folder not found: {subject_path}")
                continue
                
            try:
                try:
                    ds = load_from_disk(subject_path)
                    ds = ds['test'] if 'test' in ds else ds['train']
                except Exception:
                    parquet_path = os.path.join(subject_path, "test-00000-of-00001.parquet")
                    if os.path.exists(parquet_path):
                        ds = load_dataset("parquet", data_files=parquet_path, split='train')
                    else:
                        data_files = [
                            os.path.join(subject_path, f)
                            for f in os.listdir(subject_path)
                            if f.endswith(('.parquet', '.csv', '.json'))
                        ]
                        if not data_files:
                            continue
                        ext = data_files[0].split('.')[-1]
                        ds = load_dataset(ext, data_files=data_files, split='train')

                # 固定随机种子采样，保证每次评测的数据一致
                if len(ds) > samples_per_subject:
                    indices = random.Random(42).sample(range(len(ds)), samples_per_subject)
                    ds = ds.select(indices)
                
                for item in ds:
                    prompt = (
                        f"Question: {item['question']}\n"
                        f"Options:\n"
                        f"A. {item['choices'][0]}\n"
                        f"B. {item['choices'][1]}\n"
                        f"C. {item['choices'][2]}\n"
                        f"D. {item['choices'][3]}\n"
                        f"Answer ONLY with the letter A, B, C, or D."
                    )
                    histories.append([prompt])
                    labels.append(choices_map[item['answer']])
                    
            except Exception as e:
                logger.error(f"❌ Error parsing MMLU subject {sub}: {e}")
                
        logger.info(f"✅ Loaded {len(histories)} MMLU questions.")
        return histories, labels

    @staticmethod
    def load_gsm8k(filepath=str(GSM8K_JSONL)):
        """
        加载 GSM8K 测试集 (1319题)，并强制启用 CoT (Chain-of-Thought)
        """
        logger.info("Loading GSM8K...")
        histories, labels = [], []
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        prompt = f"Question: {item['question']}\nAnswer: Let's think step by step."
                        histories.append([prompt])
                        exact_answer = item['answer'].split("#### ")[-1].strip()
                        labels.append(exact_answer)
                logger.info(f"✅ Loaded {len(histories)} GSM8K questions from local file.")
                return histories, labels

            dataset = load_dataset("gsm8k", "main", split="test")
            for item in dataset:
                # Prompt 设计：强迫模型输出计算过程
                prompt = f"Question: {item['question']}\nAnswer: Let's think step by step."
                histories.append([prompt])
                # 提取由 '####' 分隔的最终精确数字答案作为验证标签
                exact_answer = item['answer'].split("#### ")[-1].strip()
                labels.append(exact_answer)
            logger.info(f"✅ Loaded {len(histories)} GSM8K questions.")
            return histories, labels
        except Exception as e:
            logger.error(f"❌ Failed to load GSM8K: {e}")
            return [], []

    @staticmethod
    def load_mt_bench(filepath=str(MT_BENCH_JSONL)):
        """
        加载 MT-Bench 核心多轮对话 (80个场景)
        格式预期: FastChat 标准问题 jsonl，每行包含 'turns' 列表。
        """
        logger.info(f"Loading MT-Bench from {filepath}...")
        histories, labels = [], []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # MT-Bench 包含两轮 prompt: ["turn1 prompt", "turn2 prompt"]
                    turns = _normalize_turns(data.get('turns', []))
                    if turns:
                        histories.append(turns)
                        labels.append(0) # 0 代表安全/正常对话
            logger.info(f"✅ Loaded {len(histories)} MT-Bench multi-turn conversations.")
            return histories, labels
        except Exception as e:
            logger.error(f"❌ Failed to load MT-Bench: {e}")
            return [], []

    @staticmethod
    def load_humaneval(filepath=str(HUMANEVAL_JSONL)):
        """
        加载 HumanEval (164题)
        格式预期: jsonl，每行包含 task_id/prompt/test/entry_point 等字段。
        """
        logger.info(f"Loading HumanEval from {filepath}...")
        rows = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            logger.info(f"✅ Loaded {len(rows)} HumanEval tasks.")
            return rows
        except Exception as e:
            logger.error(f"❌ Failed to load HumanEval: {e}")
            return []

    @staticmethod
    def prepare_missing_datasets(base_dir=str(DATA_DIR)):
        """
        本地优先准备评测数据：优先复用已存在文件；缺失时自动下载并写入当前项目 data 目录。
        """
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)

        # 1) AdvBench: 如果不存在，优先由本地 harmful_behaviors.csv 或外部本地仓库转换
        adv_out = base / "advbench.csv"
        if not adv_out.exists():
            src = _first_existing([
                str(base / "harmful_behaviors.csv"),
                str(LOCAL_MIRROR_ADV_BENCH),
            ])
            if src:
                df = pd.read_csv(src)
                col = "Goal" if "Goal" in df.columns else ("goal" if "goal" in df.columns else None)
                if col is not None:
                    pd.DataFrame({"goal": df[col].astype(str)}).to_csv(adv_out, index=False)
                    logger.info(f"✅ Prepared AdvBench file: {adv_out}")

        # 2) JailbreakBench: 本地没有就下载并落盘为 csv
        jbb_out = base / "jailbreakbench.csv"
        if not jbb_out.exists():
            local_jbb = _first_existing([
                str(LOCAL_MIRROR_JBB),
            ])
            if local_jbb:
                src_df = pd.read_csv(local_jbb)
                col = "Goal" if "Goal" in src_df.columns else ("goal" if "goal" in src_df.columns else None)
                if col is not None:
                    pd.DataFrame({"Goal": src_df[col].astype(str)}).to_csv(jbb_out, index=False)
                    logger.info(f"✅ Copied JailbreakBench from local mirror to: {jbb_out}")

        if not jbb_out.exists():
            try:
                ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="train")
                pd.DataFrame({"Goal": ds["Goal"]}).to_csv(jbb_out, index=False)
                logger.info(f"✅ Downloaded JailbreakBench to: {jbb_out}")
            except Exception as e:
                logger.warning(f"⚠️ JailbreakBench download failed: {e}")

        # 3) GSM8K: 本地没有就下载并落盘 jsonl
        gsm_out = base / "gsm8k_test.jsonl"
        if not gsm_out.exists():
            try:
                ds = load_dataset("gsm8k", "main", split="test")
                with open(gsm_out, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps({"question": item["question"], "answer": item["answer"]}, ensure_ascii=False) + "\n")
                logger.info(f"✅ Downloaded GSM8K to: {gsm_out}")
            except Exception as e:
                logger.warning(f"⚠️ GSM8K download failed: {e}")

        # 3.5) MaliciousInstruct: 本地没有就下载并落盘 csv
        mal_out = base / "maliciousinstruct.csv"
        if not mal_out.exists():
            try:
                ds = load_dataset("walledai/MaliciousInstruct", split="train")
                pd.DataFrame({"prompt": ds["prompt"]}).to_csv(mal_out, index=False)
                logger.info(f"✅ Downloaded MaliciousInstruct to: {mal_out}")
            except Exception:
                try:
                    ds = load_dataset("LLMSafety/MaliciousInstruct", split="train")
                    pd.DataFrame({"prompt": ds["prompt"]}).to_csv(mal_out, index=False)
                    logger.info(f"✅ Downloaded MaliciousInstruct (fallback) to: {mal_out}")
                except Exception as e:
                    logger.warning(f"⚠️ MaliciousInstruct download failed: {e}")

        # 4) MT-Bench: 优先复用本地镜像，否则从 HF 下载
        mt_dir = base / "mt_bench"
        mt_dir.mkdir(parents=True, exist_ok=True)
        mt_out = mt_dir / "question.jsonl"
        if not mt_out.exists():
            local_mt = _first_existing([
                str(LOCAL_MIRROR_MT_BENCH),
            ])
            if local_mt:
                with open(local_mt, "r", encoding="utf-8") as src, open(mt_out, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                logger.info(f"✅ Copied MT-Bench from local mirror to: {mt_out}")
            else:
                try:
                    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                    with open(mt_out, "w", encoding="utf-8") as f:
                        for item in ds:
                            row = {"turns": item.get("turns", [])}
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    logger.info(f"✅ Downloaded MT-Bench prompts to: {mt_out}")
                except Exception as e:
                    logger.warning(f"⚠️ MT-Bench download failed: {e}")

        # 5) Crescendo: 优先复用本地镜像并转换为 json 列表格式
        crescendo_out = base / "crescendo.json"
        if not crescendo_out.exists():
            local_cres = _first_existing([
                str(EXTERNAL_CRESCENDO_PROMPTS),
                str(LOCAL_MIRROR_CRESCENDO),
            ])
            if local_cres:
                rows = []
                with open(local_cres, "r", encoding="utf-8") as f:
                    if str(local_cres).endswith(".json") and "crescendo_prompts" not in str(local_cres):
                        data = json.load(f)
                        for obj in data:
                            turns = _normalize_turns(obj.get("turns", []))
                            if turns:
                                rows.append({"turns": turns})
                    else:
                        for line in f:
                            obj = json.loads(line)
                            if "crescendo_sequence" in obj:
                                if obj.get("intent") != "harmful":
                                    continue
                                turns = _normalize_turns(obj.get("crescendo_sequence", []))
                            else:
                                turns = _normalize_turns(obj.get("turns", []))
                            if turns:
                                rows.append({"turns": turns})
                with open(crescendo_out, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ Prepared Crescendo file: {crescendo_out}")

        # 5.5) HumanEval: 本地没有就下载并落盘 jsonl
        humaneval_out = base / "humaneval.jsonl"
        if not humaneval_out.exists():
            try:
                ds = load_dataset("openai/openai_humaneval", split="test")
                with open(humaneval_out, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info(f"✅ Downloaded HumanEval to: {humaneval_out}")
            except Exception as e:
                logger.warning(f"⚠️ HumanEval download failed: {e}")

        # 6) MultiBreak: 尝试下载常见候选数据集，成功后保存为 jsonl
        multibreak_out = base / "multibreak.jsonl"
        if not multibreak_out.exists():
            try:
                ds = load_dataset(
                    "tom-gibbs/multi-turn_jailbreak_attack_datasets",
                    split="train",
                    streaming=True,
                )
                kept = 0
                with open(multibreak_out, "w", encoding="utf-8") as f:
                    for i, item in enumerate(ds):
                        mt = item.get("Multi-turn conversation")
                        if not mt:
                            continue
                        try:
                            parsed = ast.literal_eval(mt) if isinstance(mt, str) else mt
                        except Exception:
                            continue
                        turns = _normalize_turns(parsed)
                        if len(turns) >= 2:
                            row = {
                                "id": item.get("Example ID", f"tg_{i}"),
                                "goal": item.get("Goal", ""),
                                "turns": turns,
                            }
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            kept += 1
                if kept > 0:
                    logger.info(
                        f"✅ Downloaded and converted tom-gibbs multi-turn set to: {multibreak_out} (rows={kept})"
                    )
            except Exception as e:
                logger.warning(f"⚠️ tom-gibbs MultiBreak download failed: {e}")

        if not multibreak_out.exists():
            candidates = [
                ("PKU-Alignment/MultiJail", "train"),
                ("thu-coai/MultiJail", "train"),
            ]
            for name, split in candidates:
                try:
                    ds = load_dataset(name, split=split)
                    with open(multibreak_out, "w", encoding="utf-8") as f:
                        for item in ds:
                            turns = _normalize_turns(item.get("turns", item.get("dialogue", [])))
                            if turns:
                                f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
                    logger.info(f"✅ Downloaded multi-turn jailbreak set ({name}) to: {multibreak_out}")
                    break
                except Exception:
                    continue
            if not multibreak_out.exists():
                logger.warning("⚠️ MultiBreak-like dataset download failed automatically; please provide a direct URL or local file.")

        return {
            "advbench": str(adv_out),
            "jailbreakbench": str(jbb_out),
            "maliciousinstruct": str(mal_out),
            "gsm8k": str(gsm_out),
            "humaneval": str(humaneval_out),
            "mt_bench": str(mt_out),
            "crescendo": str(crescendo_out),
            "multibreak": str(multibreak_out),
        }

# ================= 测试模块 =================
if __name__ == "__main__":
    # 快速校验数据集加载是否正常 (不会真的触发 LLM 推理)
    print("--- Testing Benchmark Loader ---")

    prepared = BenchmarkLoader.prepare_missing_datasets()
    print("Prepared paths:", prepared)
    
    # 测试 MMLU (本地已存在)
    mmlu_h, mmlu_l = BenchmarkLoader.load_mmlu()
    if mmlu_h: print(f"MMLU Sample: {mmlu_h[0][0][:50]}... -> Label: {mmlu_l[0]}")
    
    # 测试 GSM8K (在线拉取)
    gsm_h, gsm_l = BenchmarkLoader.load_gsm8k()
    if gsm_h: print(f"GSM8K Sample: {gsm_h[0][0][:50]}... -> Label: {gsm_l[0]}")
    
    # 其他本地安全数据集如果没有放在指定路径，会打印 Error 日志并跳过，不会阻断程序