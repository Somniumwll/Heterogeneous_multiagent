from __future__ import annotations

import argparse

from gsm8k_multiagent.collaboration.fe_system import FECollaborationSystem
from gsm8k_multiagent.collaboration.sw_system import SWCollaborationSystem
from gsm8k_multiagent.data.types import Problem
from gsm8k_multiagent.experiment.analyzer import ResultAnalyzer
from gsm8k_multiagent.experiment.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Experiment entry points
# ---------------------------------------------------------------------------

def run_experiment(
    paradigm: str = "sw",
    n: int = 50,
    seed: int = 42,
    max_rounds: int = 5,
) -> None:
    """Run the full comparative experiment for *paradigm* and save results."""
    runner  = ExperimentRunner(paradigm=paradigm)
    results = runner.run(sample_size=n, seed=seed, max_rounds=max_rounds)
    summary = ResultAnalyzer.summarize(results)
    ResultAnalyzer.print_report(summary)
    ResultAnalyzer.save_csv(results)


def run_single_problem(
    question: str,
    answer: str,
    difficulty: str = "medium",
    paradigm: str = "sw",
) -> None:
    """
    Solve one problem with all three conditions and print side-by-side results.

    Useful for rapid qualitative inspection of collaboration behaviour.
    """
    problem = Problem(id="demo_1", question=question, answer=answer, difficulty=difficulty)
    system  = SWCollaborationSystem() if paradigm == "sw" else FECollaborationSystem()

    print(f"\nProblem : {question}")
    print(f"Answer  : {answer}\n")

    for label, method in [
        ("Baseline",       system.baseline),
        ("Entropy-guided", system.entropy_guided),
        ("RAG-enhanced",   system.rag_enhanced),
    ]:
        r = method(problem)
        print(f"  {label:<16}  {'✓' if r.is_correct else '✗'}  "
              f"answer={r.final_answer}  rounds={r.rounds_used}  H={r.final_entropy:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K Multi-Agent Collaboration Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paradigm", choices=["sw", "fe"], default="sw",
        help="Collaboration paradigm: sw (Strong–Weak) or fe (Framework–Execution)",
    )
    parser.add_argument("--n",          type=int, default=50,  help="Number of problems")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed")
    parser.add_argument("--max-rounds", type=int, default=5,   help="Max rounds per problem")
    parser.add_argument(
        "--single", action="store_true",
        help="Run a built-in demo problem instead of the full experiment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.single:
        run_single_problem(
            question=(
                "A store offers a 20% discount on all items. Sarah buys a jacket "
                "originally priced at $80 and shoes originally priced at $60. She also "
                "has a $10-off coupon. If sales tax is 8%, what is the final amount "
                "Sarah pays?"
            ),
            answer="120.96",
            difficulty="hard",
            paradigm=args.paradigm,
        )
    else:
        run_experiment(
            paradigm=args.paradigm,
            n=args.n,
            seed=args.seed,
            max_rounds=args.max_rounds,
        )
