import math
import re
from collections import Counter
from typing import Dict, List

from gsm8k_multiagent.config import ENTROPY_TAU_1, ENTROPY_TAU_2


class InformationEntropyAnalyzer:

    #: Per-category uncertainty word lists and their signed weights
    _UNCERTAINTY_WEIGHTS: Dict[str, Dict] = {
        'high': {
            'words': ['confused', 'unsure', 'maybe', 'might', 'possibly',
                      'perhaps', 'guess', 'not sure', 'unclear'],
            'weight': 2.0,
        },
        'medium': {
            'words': ['think', 'probably', 'likely', 'seems', 'appears', 'could be'],
            'weight': 1.0,
        },
        'low': {  # confidence markers reduce entropy
            'words': ['sure', 'certain', 'definitely', 'clearly', 'obviously', 'exactly'],
            'weight': -1.0,
        },
    }

    _STRUCTURE_INDICATORS: List[str] = [
        'first', 'then', 'next', 'finally', 'step',
        'because', 'therefore', 'so',
    ]

    def __init__(self, tau_1: float = ENTROPY_TAU_1, tau_2: float = ENTROPY_TAU_2):
        """
        Args:
            tau_1: Entropy threshold separating light from moderate guidance.
            tau_2: Entropy threshold separating moderate from intensive guidance.
        """
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, response: str, problem: str, round_num: int = 1) -> float:

        words = response.lower().split()
        if not words:
            return 10.0

        vocab_h    = self._vocab_entropy(words)
        uncertainty = self._uncertainty_score(response.lower())
        structure_r = min(
            sum(1 for w in self._STRUCTURE_INDICATORS if w in response.lower()) * 0.5,
            2.0,
        )
        number_r   = self._number_relevance(problem, response)
        length_pen = 1.0 if len(words) < 10 else (0.5 if len(words) > 150 else 0.0)

        score = vocab_h + uncertainty + length_pen - structure_r - number_r
        return max(0.0, score)

    def guidance_level(self, entropy: float, round_num: int = 1) -> str:
        
        shift = min((round_num - 1) * 0.2, 0.6)
        t1 = max(self.tau_1 - shift, 1.0)
        t2 = max(self.tau_2 - shift, 2.0)

        if entropy <= t1:
            return "light_guidance"
        if entropy <= t2:
            return "moderate_guidance"
        return "intensive_guidance"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vocab_entropy(words: List[str]) -> float:
        freq = Counter(words)
        n = len(words)
        return -sum((c / n) * math.log2(c / n) for c in freq.values())

    def _uncertainty_score(self, text: str) -> float:
        score = 0.0
        for meta in self._UNCERTAINTY_WEIGHTS.values():
            count = sum(1 for w in meta['words'] if w in text)
            score += count * meta['weight']
        return score

    @staticmethod
    def _number_relevance(problem: str, response: str) -> float:
        p_nums = set(re.findall(r'\d+\.?\d*', problem))
        r_nums = set(re.findall(r'\d+\.?\d*', response))
        if not p_nums:
            return 0.0
        return len(p_nums & r_nums) / len(p_nums)
