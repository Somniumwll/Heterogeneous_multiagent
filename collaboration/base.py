from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

from gsm8k_multiagent.core.client import DualAgentClient
from gsm8k_multiagent.core.entropy import InformationEntropyAnalyzer
from gsm8k_multiagent.core.memory import RAGMemorySystem
from gsm8k_multiagent.data.types import ExperimentResult, Problem


class BaseCollaborationSystem(ABC):
   

    def __init__(
        self,
        client:  Optional[DualAgentClient]        = None,
        memory:  Optional[RAGMemorySystem]         = None,
        entropy: Optional[InformationEntropyAnalyzer] = None,
    ):
        self.client  = client  or DualAgentClient()
        self.memory  = memory  or RAGMemorySystem()
        self.entropy = entropy or InformationEntropyAnalyzer()

    # ------------------------------------------------------------------
    # Shared static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def extract_answer(text: str) -> str:
        """Return the last numeric token in *text* as the predicted answer."""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else ""

    @staticmethod
    def is_correct(predicted: str, ground_truth: str) -> bool:
        """Return ``True`` iff *predicted* ≈ *ground_truth* (tolerance 1e-6)."""
        try:
            return abs(float(predicted) - float(ground_truth)) < 1e-6
        except ValueError:
            return predicted.strip().lower() == ground_truth.strip().lower()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def baseline(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """Condition 1: no-guidance baseline."""

    @abstractmethod
    def entropy_guided(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """Condition 2: entropy-based adaptive guidance."""

    @abstractmethod
    def rag_enhanced(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """Condition 3: entropy guidance + RAG memory retrieval."""
