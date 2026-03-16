import random
import re
from typing import List, Optional, Tuple

import numpy as np

from gsm8k_multiagent.data.types import Problem


class GSM8KLoader:

    _OPERATION_KEYWORDS: List[str] = [
        'add', 'plus', 'sum', 'total', 'altogether', 'combined',
        'subtract', 'minus', 'less', 'difference', 'remaining', 'left',
        'multiply', 'times', 'each', 'per', 'double', 'triple', 'twice',
        'divide', 'split', 'share', 'average', 'half', 'quarter',
    ]
    _COMPLEX_CONCEPTS: List[str] = [
        'percent', '%', 'ratio', 'proportion', 'rate',
        'discount', 'profit', 'loss', 'interest',
    ]
    _DIFFICULTY_WEIGHTS: Tuple[float, float, float] = (0.30, 0.50, 0.20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        sample_size: int = 50,
        split: str = "test",
        seed: Optional[int] = None,
    ) -> List[Problem]:
      
        rng_state = self._save_rng(seed)
        try:
            problems = self._load_from_hub(split)
        except Exception as exc:
            print(f"[GSM8KLoader] HuggingFace load failed ({exc}). Using fallback samples.")
            problems = self._fallback_samples()
        result = self._stratified_sample(problems, sample_size)
        self._restore_rng(rng_state, seed)
        return result

    # ------------------------------------------------------------------
    # Difficulty scoring
    # ------------------------------------------------------------------

    def score_difficulty(self, question: str) -> float:
        """Compute a numerical difficulty score for one problem statement."""
        q = question.lower()
        num_count  = len(re.findall(r'\d+\.?\d*', question))
        op_count   = sum(1 for kw in self._OPERATION_KEYWORDS if kw in q)
        cx_count   = sum(1 for kw in self._COMPLEX_CONCEPTS  if kw in q)
        word_count = len(question.split())

        score = num_count * 0.3 + op_count * 0.4 + cx_count * 0.5
        if   word_count > 50: score += 1.0
        elif word_count > 30: score += 0.5
        return score

    def categorize_difficulty(self, score: float) -> str:
        """Map a numeric difficulty score to ``'easy'``, ``'medium'``, or ``'hard'``."""
        if score <= 2.0: return 'easy'
        if score <= 4.0: return 'medium'
        return 'hard'

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_hub(self, split: str) -> List[Problem]:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split=split)
        problems = []
        for idx, item in enumerate(dataset):
            raw = item['answer']
            answer = (
                raw.split('####')[-1].strip() if '####' in raw
                else (re.findall(r'-?\d+\.?\d*', raw) or [""])[-1]
            )
            score = self.score_difficulty(item['question'])
            problems.append(Problem(
                id=f"gsm8k_{idx + 1}",
                question=item['question'],
                answer=answer,
                difficulty=self.categorize_difficulty(score),
            ))
        print(f"[GSM8KLoader] Loaded {len(problems)} problems from HuggingFace ({split}).")
        return problems

    def _stratified_sample(self, problems: List[Problem], total: int) -> List[Problem]:
        """Sample *total* problems while preserving the target difficulty ratio."""
        buckets = {d: [p for p in problems if p.difficulty == d]
                   for d in ('easy', 'medium', 'hard')}
        easy_n  = int(total * self._DIFFICULTY_WEIGHTS[0])
        medium_n = int(total * self._DIFFICULTY_WEIGHTS[1])
        hard_n  = total - easy_n - medium_n

        sampled: List[Problem] = []
        for diff, n in [('easy', easy_n), ('medium', medium_n), ('hard', hard_n)]:
            pool = buckets[diff]
            sampled.extend(random.sample(pool, min(n, len(pool))))
        random.shuffle(sampled)

        counts = {d: sum(1 for p in sampled if p.difficulty == d) for d in ('easy', 'medium', 'hard')}
        print(f"[GSM8KLoader] Sampled {len(sampled)}: easy={counts['easy']}, "
              f"medium={counts['medium']}, hard={counts['hard']}")
        return sampled

    @staticmethod
    def _save_rng(seed: Optional[int]):
        if seed is None:
            return None
        state = (random.getstate(), np.random.get_state())
        random.seed(seed)
        np.random.seed(seed)
        return state

    @staticmethod
    def _restore_rng(state, seed: Optional[int]):
        if seed is not None and state is not None:
            random.setstate(state[0])
            np.random.set_state(state[1])

    @staticmethod
    def _fallback_samples() -> List[Problem]:
        return [
            Problem("gsm8k_1",
                    "Natalia sold clips to 48 of her friends in April, and then she sold "
                    "half as many clips in May. How many clips did Natalia sell altogether "
                    "in April and May?",
                    "72", "easy"),
            Problem("gsm8k_2",
                    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
                    "minutes of babysitting. How much did she earn?",
                    "10", "medium"),
            Problem("gsm8k_3",
                    "A store offers a 20% discount on all items. Sarah buys a jacket "
                    "originally priced at $80 and shoes originally priced at $60. She also "
                    "has a $10-off coupon. If sales tax is 8%, what is the final amount "
                    "Sarah pays?",
                    "120.96", "hard"),
        ]
