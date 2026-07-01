"""
scripts/check_degenerate.py

Scans annotation_responses.jsonl for degenerate (repetition-looping) responses
and reports which models and prompts are affected.

Usage:
    python3 scripts/check_degenerate.py
    python3 scripts/check_degenerate.py --input annotation_responses.jsonl --threshold 0.4
"""

import argparse
import json
from collections import defaultdict


def is_degenerate(text: str, repeat_threshold: float = 0.4, min_sentences: int = 4) -> bool:
    """
    Flags a response as degenerate if the ratio of unique sentences to total
    sentences is below repeat_threshold. Truncated responses (ending mid-word)
    are also flagged.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if len(sentences) < min_sentences:
        return False
    unique_ratio = len(set(sentences)) / len(sentences)
    return unique_ratio < repeat_threshold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="annotation_responses.jsonl")
    p.add_argument("--threshold", type=float, default=0.4)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Total records: {len(records)}")
    print(f"Degenerate threshold: unique sentence ratio < {args.threshold}\n")

    degenerate = [r for r in records if is_degenerate(r["response"], args.threshold)]

    print(f"Degenerate responses: {len(degenerate)} / {len(records)} ({100*len(degenerate)/len(records):.1f}%)\n")

    # By model
    by_model = defaultdict(list)
    for r in degenerate:
        by_model[r["model_id"]].append(r["prompt_id"])

    print("=" * 70)
    print(f"{'Model':<55} {'Count':>5}  Prompt IDs")
    print("=" * 70)
    for model_id, prompt_ids in sorted(by_model.items(), key=lambda x: -len(x[1])):
        print(f"{model_id:<55} {len(prompt_ids):>5}  {sorted(prompt_ids)}")

    # By stage
    print()
    by_stage = defaultdict(int)
    total_by_stage = defaultdict(int)
    for r in records:
        total_by_stage[r["stage"]] += 1
    for r in degenerate:
        by_stage[r["stage"]] += 1

    print("By stage:")
    for stage in ["base", "cpt", "sft", "dpo", "grpo", "gspo"]:
        n = by_stage.get(stage, 0)
        t = total_by_stage.get(stage, 0)
        pct = 100 * n / t if t else 0
        print(f"  {stage:<8} {n:>4} / {t:<4} ({pct:.1f}%)")

    # By family
    print()
    by_family = defaultdict(int)
    total_by_family = defaultdict(int)
    for r in records:
        total_by_family[r["family"]] += 1
    for r in degenerate:
        by_family[r["family"]] += 1

    print("By family:")
    for family in ["gemma", "llama", "qwen"]:
        n = by_family.get(family, 0)
        t = total_by_family.get(family, 0)
        pct = 100 * n / t if t else 0
        print(f"  {family:<8} {n:>4} / {t:<4} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
