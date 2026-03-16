from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from gsm8k_multiagent.collaboration.base import BaseCollaborationSystem
from gsm8k_multiagent.data.types import ExperimentResult, Problem


class FECollaborationSystem(BaseCollaborationSystem):
  

    # ------------------------------------------------------------------
    # Public conditions
    # ------------------------------------------------------------------

    def baseline(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """
        Condition 1 – No-guidance FE baseline.
        Strong agent identifies framework deviations only; no remediation.
        """
        return self._run(problem, mode="baseline", max_rounds=max_rounds)

    def entropy_guided(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """
        Condition 2 – Entropy-guided FE collaboration.
        """
        return self._run(problem, mode="entropy", max_rounds=max_rounds)

    def rag_enhanced(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """
        Condition 3 – RAG-enhanced FE collaboration.
        """
        return self._run(problem, mode="rag", max_rounds=max_rounds)

    # ------------------------------------------------------------------
    # Core collaboration loop
    # ------------------------------------------------------------------

    def _run(self, problem: Problem, mode: str, max_rounds: int) -> ExperimentResult:
        t0       = time.time()
        log:     List[Dict] = []
        used_rag = False
        uf_log:  List[str] = []   # framework-understanding failures
        ed_log:  List[str] = []   # execution deviations
        feedback = ""

        # Round 1 – strong agent provides framework
        if mode == "rag":
            framework, used_rag = self._strong_provide_framework_rag(problem.question)
        else:
            framework = self._strong_provide_framework(problem.question)
        log.append({"round": 1, "agent": "strong",
                    "role": "provide_framework", "content": framework})

        # Round 2 – weak agent executes framework
        current = self._weak_execute(problem.question, framework)
        log.append({"round": 2, "agent": "weak",
                    "role": "execute_framework", "content": current})
        entropy_traj = [self.entropy.compute(current, problem.question)]

        # Rounds 3‥max_rounds – check understanding and guide re-execution
        for rnd in range(3, max_rounds + 1):
            h = self.entropy.compute(current, problem.question)
            entropy_traj.append(h)

            feedback = self._strong_check(
                problem.question, framework, current, h, rnd, mode
            )
            log.append({"round": rnd, "agent": "strong",
                        "role": "check", "content": feedback})

            # Record observed collaboration failures
            f_lower = feedback.lower()
            if "understanding deviation" in f_lower or "misunderstood" in f_lower:
                uf_log.append(f"Round {rnd}: framework understanding deviation")
            if "deviated" in f_lower or "not following" in f_lower:
                ed_log.append(f"Round {rnd}: execution deviation")

            if self._fully_correct(feedback):
                break

            if rnd < max_rounds:
                current = self._weak_re_execute(
                    problem.question, framework, current, feedback, mode
                )
                log.append({"round": rnd, "agent": "weak",
                            "role": "re_execute", "content": current})

        # Persist successful solutions
        final_answer = self.extract_answer(current)
        correct      = self.is_correct(final_answer, problem.answer)
        final_h      = entropy_traj[-1]

        if mode == "rag" and correct and final_h < 3.5:
            self.memory.add(
                problem=problem.question,
                framework=framework,
                solution=current,
                answer=final_answer,
                difficulty=problem.difficulty,
                entropy=final_h,
                rounds=sum(1 for e in log if e["agent"] == "weak"),
            )

        return ExperimentResult(
            problem_id       = problem.id,
            problem          = problem.question,
            ground_truth     = problem.answer,
            method           = f"fe_{mode}",
            final_answer     = final_answer,
            is_correct       = correct,
            rounds_used      = sum(1 for e in log if e["agent"] == "weak"),
            total_time       = time.time() - t0,
            entropy_trajectory = entropy_traj,
            final_entropy    = final_h,
            difficulty       = problem.difficulty,
            used_rag         = used_rag,
            conversation_log = log,
            framework_understanding_failures = uf_log,
            execution_deviations             = ed_log,
        )

    # ------------------------------------------------------------------
    # Agent call helpers
    # ------------------------------------------------------------------

    def _strong_provide_framework(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math expert. Provide a clear solution *framework* for the "
                    "problem: identify key quantities, state the solution strategy, and "
                    "outline the required calculation steps in order. "
                    "Do NOT compute the final numerical answer."
                ),
            },
            {"role": "user",
             "content": f"Problem: {question}\n\nProvide the solving framework."},
        ]
        return self.client.call(messages, agent="strong")

    def _strong_provide_framework_rag(self, question: str) -> Tuple[str, bool]:
        mem    = self.memory.retrieve(question, threshold=0.6)
        system = (
            "You are a math expert. Provide a solving framework without computing "
            "the final answer."
        )
        if mem:
            system += (
                f"\n\nRelevant past experience:\n"
                f"Similar problem: {mem.problem_text}\n"
                f"Successful framework: {mem.framework_approach[:300]}"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Problem: {question}\n\nProvide the solving framework."},
        ]
        return self.client.call(messages, agent="strong"), bool(mem)

    def _weak_execute(self, question: str, framework: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math assistant. Follow the given framework precisely "
                    "and compute the final numerical answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {question}\n\n"
                    f"Framework to follow:\n{framework}\n\n"
                    "Execute the framework with full calculations and state the final answer."
                ),
            },
        ]
        return self.client.call(messages, agent="weak")

    def _strong_check(
        self,
        question: str,
        framework: str,
        execution: str,
        entropy: float,
        rnd: int,
        mode: str,
    ) -> str:
        if mode == "baseline":
            prompt = (
                "Check whether the executor correctly followed the framework.\n\n"
                f"Problem: {question}\n"
                f"Framework: {framework}\n"
                f"Execution: {execution}\n\n"
                "Point out deviations only – do NOT provide corrections. "
                "If both understanding and execution are correct, reply exactly: "
                "'Framework understanding correct and execution is correct.'"
            )
        else:
            prompt = self._check_prompt(question, framework, execution, entropy, rnd, mode)

        return self.client.call([{"role": "user", "content": prompt}], agent="strong")

    def _check_prompt(
        self,
        question: str,
        framework: str,
        execution: str,
        entropy: float,
        rnd: int,
        mode: str,
    ) -> str:
        level = self.entropy.guidance_level(entropy, rnd)

        rag_block = ""
        if mode == "rag":
            mem = self.memory.retrieve(question, threshold=0.5)
            if mem:
                rag_block = (
                    "\n\n**Reference experience:**\n"
                    f"Framework: {mem.framework_approach[:200]}\n"
                    f"Solution: {mem.solution_steps[:200]}"
                )

        instructions = {
            "light_guidance": (
                "Performance is good. Confirm framework adherence, verify the arithmetic, "
                "and give brief positive reinforcement."
            ),
            "moderate_guidance": (
                "Identify the main deviation from the framework and provide a targeted hint. "
                "Leave the correction to the executor."
            ),
            "intensive_guidance": (
                "Provide a detailed analysis: enumerate every deviation, reconstruct the "
                "correct step-by-step execution of the framework, and state explicitly "
                "what the executor must do next."
            ),
        }

        return (
            f"[Round {rnd} | {level}]\n\n"
            f"Problem: {question}\n"
            f"Framework: {framework}\n"
            f"Execution: {execution}"
            f"{rag_block}\n\n"
            f"Review instruction: {instructions[level]}\n\n"
            "If both framework understanding and execution are correct, reply: "
            "'Framework understanding correct and execution is correct.'"
        )

    def _weak_re_execute(
        self,
        question: str,
        framework: str,
        current: str,
        feedback: str,
        mode: str,
    ) -> str:
        messages = [{
            "role": "user",
            "content": (
                f"Problem: {question}\n\n"
                f"Framework to follow:\n{framework}\n\n"
                f"Your previous execution:\n{current}\n\n"
                f"Reviewer feedback:\n{feedback}\n\n"
                "Revise your execution to follow the framework more accurately."
            ),
        }]
        return self.client.call(messages, agent="weak")

    @staticmethod
    def _fully_correct(review: str) -> bool:
        lower = review.lower()
        return (
            "framework understanding correct" in lower
            and ("execution is correct" in lower or "execution correct" in lower)
        )
