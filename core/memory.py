import json
import os
import time
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gsm8k_multiagent.data.types import ProblemMemory


class RAGMemorySystem:
  

    def __init__(self, memory_file: str = "gsm8k_memory.json", capacity: int = 50):
        """
        Args:
            memory_file: Path to the JSON persistence file.
            capacity:    Maximum number of memories to retain.
        """
        self._file     = memory_file
        self._capacity = capacity
        self.memories: List[ProblemMemory] = []
        self._vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self._vectors    = None
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        problem: str,
        framework: str,
        solution: str,
        answer: str,
        difficulty: str,
        entropy: float,
        rounds: int = 1,
    ) -> None:
       
        new_eff = self._effectiveness(entropy, rounds)
        idx = self._find_similar(problem, threshold=0.9)

        if idx >= 0:
            old_eff = self._effectiveness(self.memories[idx].success_entropy, 1)
            if new_eff > old_eff:
                self.memories[idx] = self._make(problem, framework, solution, answer, difficulty, entropy)
            else:
                self.memories[idx].usage_count += 1
        else:
            self.memories.append(
                self._make(problem, framework, solution, answer, difficulty, entropy)
            )

        if len(self.memories) > self._capacity:
            self.memories.sort(key=lambda m: self._effectiveness(m.success_entropy, 1), reverse=True)
            self.memories = self.memories[: self._capacity]

        self._rebuild_vectors()
        self._save()

    def retrieve(self, problem: str, threshold: float = 0.6) -> Optional[ProblemMemory]:
       
        idx = self._find_similar(problem, threshold)
        if idx < 0:
            return None
        self.memories[idx].usage_count += 1
        return self.memories[idx]

    def clear(self) -> None:
        """Delete all in-memory and on-disk data."""
        self.memories.clear()
        self._vectors = None
        if os.path.exists(self._file):
            os.remove(self._file)

    def __len__(self) -> int:
        return len(self.memories)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make(
        problem: str, framework: str, solution: str,
        answer: str, difficulty: str, entropy: float,
    ) -> ProblemMemory:
        return ProblemMemory(
            problem_text=problem,
            framework_approach=framework,
            solution_steps=solution,
            final_answer=answer,
            difficulty=difficulty,
            success_entropy=entropy,
            timestamp=time.time(),
        )

    @staticmethod
    def _effectiveness(entropy: float, rounds: int) -> float:
        return max(0.0, 5.0 - entropy) * 0.6 + max(0.0, 6.0 - rounds) * 0.4

    def _find_similar(self, problem: str, threshold: float) -> int:
        if not self.memories or self._vectors is None:
            return -1
        try:
            vec  = self._vectorizer.transform([problem])
            sims = cosine_similarity(vec, self._vectors)[0]
            best = int(np.argmax(sims))
            return best if sims[best] >= threshold else -1
        except Exception:
            return -1

    def _rebuild_vectors(self) -> None:
        if self.memories:
            try:
                self._vectors = self._vectorizer.fit_transform(
                    [m.problem_text for m in self.memories]
                )
            except Exception:
                self._vectors = None

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, 'r', encoding='utf-8') as f:
                self.memories = [ProblemMemory(**item) for item in json.load(f)]
            self._rebuild_vectors()
            print(f"[RAGMemory] Loaded {len(self.memories)} memories from '{self._file}'.")
        except Exception as exc:
            print(f"[RAGMemory] Load failed: {exc}")

    def _save(self) -> None:
        try:
            with open(self._file, 'w', encoding='utf-8') as f:
                json.dump([m.__dict__ for m in self.memories], f,
                          ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[RAGMemory] Save failed: {exc}")
