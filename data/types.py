"""
data/types.py – Shared dataclasses used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Problem:
    """A single GSM8K math word problem."""
    id: str
    question: str
    answer: str                  # Numeric ground-truth (string)
    difficulty: str              # 'easy' | 'medium' | 'hard'
    full_answer: str = ""        # Chain-of-thought gold solution (optional)


@dataclass
class ProblemMemory:
    """
    One entry in the RAG memory store.

    Captures the framework, solution steps, and quality metrics of a
    successfully solved problem so they can be retrieved for analogous
    future problems.
    """
    problem_text: str
    framework_approach: str      # High-level solution strategy
    solution_steps: str          # Detailed arithmetic steps
    final_answer: str
    difficulty: str
    success_entropy: float       # H(P_w) at the time of successful completion
    timestamp: float
    usage_count: int = 0


@dataclass
class ExperimentResult:
    """
    Complete record of one problem solved by one collaboration method.

    ``framework_understanding_failures`` and ``execution_deviations`` are
    populated only in the Framework–Execution paradigm; they remain empty
    lists in Strong–Weak experiments.
    """
    problem_id: str
    problem: str
    ground_truth: str
    method: str                              # e.g. "sw_entropy", "fe_rag"
    final_answer: str
    is_correct: bool
    rounds_used: int
    total_time: float                        # Wall-clock seconds
    entropy_trajectory: List[float]          # H(P_w) at each evaluation round
    final_entropy: float
    difficulty: str
    used_rag: bool = False
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)
    # Framework–Execution specific
    framework_understanding_failures: List[str] = field(default_factory=list)
    execution_deviations: List[str] = field(default_factory=list)
