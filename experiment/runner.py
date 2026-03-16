
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from gsm8k_multiagent.collaboration.fe_system import FECollaborationSystem
from gsm8k_multiagent.collaboration.sw_system import SWCollaborationSystem
from gsm8k_multiagent.data.loader import GSM8KLoader
from gsm8k_multiagent.data.types import ExperimentResult, Problem


class ExperimentRunner:

    _METHOD_NAMES = {
        "sw": ["sw_baseline", "sw_entropy", "sw_rag"],
        "fe": ["fe_baseline", "fe_entropy", "fe_rag"],
    }

    def __init__(self, paradigm: str = "sw"):
        """
        Args:
            paradigm: ``"sw"`` for Strong–Weak or ``"fe"`` for Framework–Execution.
        """
        if paradigm not in ("sw", "fe"):
            raise ValueError(f"Unknown paradigm '{paradigm}'. Choose 'sw' or 'fe'.")

        self._paradigm = paradigm
        self._loader   = GSM8KLoader()
        self._system   = (
            SWCollaborationSystem() if paradigm == "sw"
            else FECollaborationSystem()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        sample_size: int = 50,
        split: str = "test",
        seed: Optional[int] = None,
        max_rounds: int = 5,
    ) -> Dict[str, List[ExperimentResult]]:
    
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        problems = self._loader.load(sample_size=sample_size, split=split, seed=seed)
        method_names = self._METHOD_NAMES[self._paradigm]
        results: Dict[str, List[ExperimentResult]] = {m: [] for m in method_names}

        print(f"\n[Experiment] paradigm={self._paradigm.upper()}  "
              f"n={len(problems)}  max_rounds={max_rounds}\n")

        for i, problem in enumerate(tqdm(problems, desc="Problems")):
            print(f"\n  [{i + 1}/{len(problems)}] {problem.difficulty:6s}  "
                  f"{problem.question[:65]}…")
            try:
                r1, r2, r3 = self._solve_all(problem, max_rounds)
                results[method_names[0]].append(r1)
                results[method_names[1]].append(r2)
                results[method_names[2]].append(r3)
                self._print_row(r1, r2, r3)
            except Exception as exc:
                print(f"  [ERROR] {exc}")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solve_all(self, problem: Problem, max_rounds: int):
        """Run all three conditions and return (r1, r2, r3)."""
        r1 = self._system.baseline(problem, max_rounds=max_rounds)
        r2 = self._system.entropy_guided(problem, max_rounds=max_rounds)
        r3 = self._system.rag_enhanced(problem, max_rounds=max_rounds)
        return r1, r2, r3

    @staticmethod
    def _print_row(r1: ExperimentResult, r2: ExperimentResult, r3: ExperimentResult):
        rag = "Yes" if r3.used_rag else "No"
        print(f"    Baseline       {'✓' if r1.is_correct else '✗'}  "
              f"rounds={r1.rounds_used}  H={r1.final_entropy:.3f}")
        print(f"    Entropy-guided {'✓' if r2.is_correct else '✗'}  "
              f"rounds={r2.rounds_used}  H={r2.final_entropy:.3f}")
        print(f"    RAG-enhanced   {'✓' if r3.is_correct else '✗'}  "
              f"rounds={r3.rounds_used}  H={r3.final_entropy:.3f}  RAG={rag}")
