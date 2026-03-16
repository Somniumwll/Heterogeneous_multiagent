

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from gsm8k_multiagent.collaboration.base import BaseCollaborationSystem
from gsm8k_multiagent.data.types import ExperimentResult, Problem


class SWCollaborationSystem(BaseCollaborationSystem):

    # ------------------------------------------------------------------
    # Public conditions
    # ------------------------------------------------------------------

    def baseline(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """
        Condition 1 – No-guidance baseline.
        The strong agent points out errors without explaining corrections.
        """
        return self._run(problem, mode="baseline", max_rounds=max_rounds)

    def entropy_guided(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """
        Condition 2 – Entropy-guided collaboration.
        The strong agent selects guidance depth from {light, moderate, intensive}
        based on the current entropy H(P_w).
        """
        return self._run(problem, mode="entropy", max_rounds=max_rounds)

    def rag_enhanced(self, problem: Problem, max_rounds: int = 5) -> ExperimentResult:
        """
        Condition 3 – RAG-enhanced entropy guidance.
        A retrieved experience is prepended to the strong agent's prompt
        before applying entropy-based guidance.
        """
        return self._run(problem, mode="rag", max_rounds=max_rounds)

    # ------------------------------------------------------------------
    # Core collaboration loop
    # ------------------------------------------------------------------

    def _run(self, problem: Problem, mode: str, max_rounds: int) -> ExperimentResult:
        t0  = time.time()
        log: List[Dict] = []
        used_rag = False
        guidance = ""

        # Round 1 – weak agent initial attempt
        if mode == "rag":
            current, used_rag = self._weak_solve_rag(problem.question)
        else:
            current = self._weak_solve(problem.question)
        log.append({"round": 1, "agent": "weak", "role": "initial", "content": current})
        entropy_traj = [self.entropy.compute(current, problem.question)]

        # Rounds 2‥max_rounds – iterative review and revision
        for rnd in range(2, max_rounds + 1):
            h     = self.entropy.compute(current, problem.question)
            entropy_traj.append(h)

            if h < 1.5:          # converged – early exit
                break

            guidance = self._strong_review(problem.question, current, h, rnd, mode)
            log.append({"round": rnd, "agent": "strong", "role": "review",
                        "content": guidance, "entropy": h})

            if self._confirmed_correct(guidance):
                break

            if rnd < max_rounds:
                current = self._weak_revise(problem.question, current, guidance)
                log.append({"round": rnd, "agent": "weak",
                            "role": "revise", "content": current})

        # Persist successful, converged solutions to RAG memory
        final_answer = self.extract_answer(current)
        correct      = self.is_correct(final_answer, problem.answer)
        final_h      = entropy_traj[-1]

        if mode == "rag" and correct and final_h < 3.5:
            self.memory.add(
                problem=problem.question,
                framework=guidance,
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
            method           = f"sw_{mode}",
            final_answer     = final_answer,
            is_correct       = correct,
            rounds_used      = sum(1 for e in log if e["agent"] == "weak"),
            total_time       = time.time() - t0,
            entropy_trajectory = entropy_traj,
            final_entropy    = final_h,
            difficulty       = problem.difficulty,
            used_rag         = used_rag,
            conversation_log = log,
        )

    # ------------------------------------------------------------------
    # Agent call helpers
    # ------------------------------------------------------------------

    def _weak_solve(self, question: str) -> str:
        messages = [
            {"role": "system",
             "content": "You are a math assistant. Solve the problem step by step."},
            {"role": "user", "content": f"Problem: {question}"},
        ]
        return self.client.call(messages, agent="weak")

    def _weak_solve_rag(self, question: str) -> Tuple[str, bool]:
        mem    = self.memory.retrieve(question)
        system = "You are a math assistant. Solve the problem step by step."
        if mem:
            system = (
                "You are a math assistant. A solved reference problem is provided.\n\n"
                f"Reference problem: {mem.problem_text}\n"
                f"Reference solution: {mem.solution_steps}\n"
                f"Reference answer: {mem.final_answer}\n\n"
                "Use the reference only as a structural guide; recompute for the current values."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Problem: {question}"},
        ]
        return self.client.call(messages, agent="weak"), bool(mem)

    def _strong_review(
        self,
        question: str,
        solution: str,
        entropy: float,
        rnd: int,
        mode: str,
    ) -> str:
        if mode == "baseline":
            prompt = (
                "Review the following math solution. "
                "Identify errors only – do NOT provide the correct answer or method.\n\n"
                f"Problem: {question}\n"
                f"Solution: {solution}\n\n"
                "If correct, respond exactly: 'The solution is correct.'"
            )
        else:
            prompt = self._guidance_prompt(question, solution, entropy, rnd, mode)

        return self.client.call([{"role": "user", "content": prompt}], agent="strong")

    def _guidance_prompt(
        self,
        question: str,
        solution: str,
        entropy: float,
        rnd: int,
        mode: str,
    ) -> str:
        level = self.entropy.guidance_level(entropy, rnd)

        # Optional RAG context block
        rag_block = ""
        if mode == "rag":
            mem = self.memory.retrieve(question, threshold=0.5)
            if mem:
                rag_block = (
                    "\n\n**Relevant past experience (similar problem):**\n"
                    f"Framework: {mem.framework_approach[:200]}\n"
                    f"Solution excerpt: {mem.solution_steps[:200]}\n"
                )

        instructions = {
            "light_guidance": (
                "The student's answer is nearly correct. "
                "Confirm what is right and note any minor issues. "
                "Keep the tone encouraging."
            ),
            "moderate_guidance": (
                "Identify the main error and give a targeted hint toward the correct method. "
                "Do not reveal the final answer."
            ),
            "intensive_guidance": (
                "The student is significantly off-track. "
                "Break down the problem step by step, explain the correct reasoning framework, "
                "and specify the next concrete action the student should take."
            ),
        }

        return (
            f"[Round {rnd} | {level}]\n\n"
            f"Problem: {question}\n"
            f"Student solution: {solution}"
            f"{rag_block}\n\n"
            f"Guidance instruction: {instructions[level]}\n\n"
            "If the solution is completely correct, reply: 'The solution is correct.'"
        )

    def _weak_revise(self, question: str, current: str, guidance: str) -> str:
        messages = [{
            "role": "user",
            "content": (
                f"Problem: {question}\n\n"
                f"Your previous answer:\n{current}\n\n"
                f"Reviewer feedback:\n{guidance}\n\n"
                "Revise your solution based on the feedback."
            ),
        }]
        return self.client.call(messages, agent="weak")

    @staticmethod
    def _confirmed_correct(review: str) -> bool:
        lower = review.lower()
        return any(kw in lower for kw in
                   ["solution is correct", "answer is correct", "completely correct"])
