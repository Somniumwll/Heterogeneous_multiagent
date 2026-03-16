

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gsm8k_multiagent.data.types import ExperimentResult


class ResultAnalyzer:
    """
    Computes per-method summary statistics from a dict of ``ExperimentResult``
    lists and produces formatted reports suitable for academic publication.
    """

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(results: Dict[str, List[ExperimentResult]]) -> Dict[str, Dict]:
        """
        Compute aggregate statistics for every method in *results*.

        Returns:
            Nested dict ``{method_name: {metric_name: value}}``.
        """
        summary: Dict[str, Dict] = {}

        for method, res_list in results.items():
            if not res_list:
                continue
            n = len(res_list)

            accuracy      = sum(r.is_correct    for r in res_list) / n
            avg_rounds    = float(np.mean([r.rounds_used  for r in res_list]))
            avg_entropy   = float(np.mean([r.final_entropy for r in res_list]))
            avg_time      = float(np.mean([r.total_time    for r in res_list]))
            rag_rate      = sum(r.used_rag       for r in res_list) / n

            delta_h = [
                r.entropy_trajectory[0] - r.entropy_trajectory[-1]
                for r in res_list if len(r.entropy_trajectory) > 1
            ]
            avg_delta_h = float(np.mean(delta_h)) if delta_h else 0.0

            # Framework–Execution specific metrics (zero for SW results)
            uf_rate = sum(1 for r in res_list if r.framework_understanding_failures) / n
            ed_rate = sum(1 for r in res_list if r.execution_deviations)             / n

            # Per-difficulty breakdown
            by_difficulty: Dict[str, Dict] = {}
            for diff in ("easy", "medium", "hard"):
                sub = [r for r in res_list if r.difficulty == diff]
                if sub:
                    by_difficulty[diff] = {
                        "count":    len(sub),
                        "accuracy": sum(r.is_correct for r in sub) / len(sub),
                    }

            summary[method] = {
                "n":            n,
                "accuracy":     accuracy,
                "avg_rounds":   avg_rounds,
                "avg_entropy":  avg_entropy,
                "avg_delta_h":  avg_delta_h,
                "avg_time":     avg_time,
                "rag_rate":     rag_rate,
                "uf_rate":      uf_rate,   # framework understanding failure rate
                "ed_rate":      ed_rate,   # execution deviation rate
                "by_difficulty": by_difficulty,
            }

        return summary

    # ------------------------------------------------------------------
    # Report printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(summary: Dict[str, Dict]) -> None:
        """Print a compact comparative report to *stdout*."""
        sep = "=" * 78
        print(f"\n{sep}")
        print("  Multi-Agent Collaboration on GSM8K – Comparative Experiment Report")
        print(sep)

        # Overall metrics table
        print(f"\n{'Method':<22} {'Acc':>6} {'Rounds':>7} {'H_final':>8} "
              f"{'ΔH':>7} {'RAG':>6} {'UF':>6} {'ED':>6}")
        print("─" * 75)
        for method, s in summary.items():
            print(
                f"{method:<22}"
                f" {s['accuracy']:>6.1%}"
                f" {s['avg_rounds']:>7.2f}"
                f" {s['avg_entropy']:>8.4f}"
                f" {s['avg_delta_h']:>7.4f}"
                f" {s['rag_rate']:>6.1%}"
                f" {s['uf_rate']:>6.1%}"
                f" {s['ed_rate']:>6.1%}"
            )
        print(f"\n  UF = framework understanding failure rate  "
              f"ED = execution deviation rate")

        # Per-difficulty breakdown
        print(f"\n{'Difficulty':<10} {'Method':<22} {'N':>4} {'Accuracy':>9}")
        print("─" * 50)
        for method, s in summary.items():
            for diff, ds in s.get("by_difficulty", {}).items():
                print(f"{diff:<10} {method:<22} {ds['count']:>4} {ds['accuracy']:>9.1%}")

        # Hypothesis checks
        keys = list(summary.keys())
        if len(keys) >= 3:
            acc = [summary[k]["accuracy"] for k in keys]
            print(f"\n  Hypothesis Tests  (Method order: {' > '.join(keys)})")
            print(f"  H1  guided > baseline   "
                  f"{'✓' if acc[1] > acc[0] else '✗'}  ({acc[1]:.1%} vs {acc[0]:.1%})")
            print(f"  H2  RAG ≥ guided        "
                  f"{'✓' if acc[2] >= acc[1] else '✗'}  ({acc[2]:.1%} vs {acc[1]:.1%})")
            print(f"  H3  RAG > baseline      "
                  f"{'✓' if acc[2] > acc[0] else '✗'}  ({acc[2]:.1%} vs {acc[0]:.1%})")

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(results: Dict[str, List[ExperimentResult]]) -> pd.DataFrame:
        """Flatten *results* into a tidy ``pandas.DataFrame``."""
        rows = []
        for res_list in results.values():
            for r in res_list:
                rows.append({
                    "problem_id":             r.problem_id,
                    "method":                 r.method,
                    "difficulty":             r.difficulty,
                    "is_correct":             r.is_correct,
                    "final_answer":           r.final_answer,
                    "ground_truth":           r.ground_truth,
                    "rounds_used":            r.rounds_used,
                    "total_time_s":           round(r.total_time, 3),
                    "final_entropy":          round(r.final_entropy, 4),
                    "entropy_improvement":    round(
                        (r.entropy_trajectory[0] - r.entropy_trajectory[-1])
                        if len(r.entropy_trajectory) > 1 else 0.0, 4
                    ),
                    "used_rag":               r.used_rag,
                    "understanding_failures": len(r.framework_understanding_failures),
                    "execution_deviations":   len(r.execution_deviations),
                    "problem_snippet":        (
                        r.problem[:80] + "…" if len(r.problem) > 80 else r.problem
                    ),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def save_csv(
        results: Dict[str, List[ExperimentResult]],
        path: Optional[str] = None,
    ) -> str:
        """Save a tidy CSV of all results and return the file path."""
        df   = ResultAnalyzer.to_dataframe(results)
        path = path or f"experiment_results_{int(time.time())}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"[ResultAnalyzer] Saved {len(df)} rows → {path}")
        return path
