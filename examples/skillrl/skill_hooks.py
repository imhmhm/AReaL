"""Skill evolution controller (SkillRL pillar C) for AReaL.

Recursive skill evolution: when task success rates are low, collect failed
trajectories, ask an LLM (o3 via ``SkillUpdater``) to distil them into new
skills, and write them back into the shared ``SkillsOnlyMemory``.

AReaL mounting strategy (zero core changes):
- ``PPOTrainer.train(dynamic_filter_fn=...)`` calls ``should_accept_fn(traj)``
  once per trajectory with the workflow's output dict. We use this native hook
  both to (a) trigger evolution at the right cadence and (b) stay informed of
  trajectory outcomes.
- The *per-step* trajectory data (which ``concat_padded_tensors`` cannot carry
  as it only handles tensors) is recorded directly by the workflow into the
  controller's thread-safe buffer via :meth:`record_failure`.

This mirrors SkillRL's ``_update_skills_from_training`` / ``_validate`` logic,
retargeted from verl's ``ray_trainer.fit`` to AReaL's ``train`` + filter hook.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

from .memory import SkillUpdater, SkillsOnlyMemory

logger = logging.getLogger("SkillEvolution")


class SkillEvolutionController:
    """Thread-safe collector + evolution driver for the skill bank.

    A single instance is shared between:
    - the ``SkillEnvWorkflow`` (records failed trajectories), and
    - the ``should_accept_fn`` closure (triggers evolution).

    New skills are written only into the *training* workflow's memory — the
    eval workflow keeps its own memory instance — so validation scores are not
    inflated by skills tuned on validation data (same anti-leakage design as
    SkillRL).
    """

    def __init__(
        self,
        memory: SkillsOnlyMemory,
        skill_updater: SkillUpdater | None = None,
        update_threshold: float = 0.4,
        max_new_skills: int = 3,
        skill_update_freq: int = 5,
        save_dir: str = "./outputs",
        max_failures_to_analyze: int = 10,
    ):
        self.memory = memory
        self.update_threshold = update_threshold
        self.max_new_skills = max_new_skills
        self.skill_update_freq = skill_update_freq
        self.save_dir = save_dir
        self.max_failures_to_analyze = max_failures_to_analyze

        # Lazy-init SkillUpdater (needs AZURE_OPENAI_* env vars).
        if skill_updater is not None:
            self.skill_updater = skill_updater
        else:
            self.skill_updater = None  # built on first evolve() call

        # Thread-safe failed-trajectory buffer.
        self._lock = threading.Lock()
        self._failed: list[dict[str, Any]] = []
        self._seen_trajectories = 0
        self._last_evolve_at = 0
        self._global_step_estimate = 0
        self.update_history: list[dict] = []

    # ------------------------------------------------------------------ #
    # Called by the workflow (per trajectory)                             #
    # ------------------------------------------------------------------ #

    def record_failure(
        self,
        task: str,
        trajectory: list[dict[str, str]],
        task_type: str,
    ) -> None:
        """Record a failed trajectory for later analysis.

        ``trajectory`` is a list of ``{"action": str, "observation": str}``
        steps (SkillUpdater's expected format).
        """
        with self._lock:
            self._failed.append(
                {"task": task, "trajectory": trajectory, "task_type": task_type}
            )

    # ------------------------------------------------------------------ #
    # should_accept_fn factory (AReaL's dynamic_filter_fn)                 #
    # ------------------------------------------------------------------ #

    def make_should_accept_fn(self) -> Callable[[dict[str, Any]], bool]:
        """Build the ``should_accept_fn`` passed to ``trainer.train``.

        Returns ``True`` for every trajectory (we do NOT reject — failure
        collection happens in-workflow). Its side effect is to trigger
        :meth:`maybe_evolve` at the configured cadence.
        """

        def should_accept_fn(traj: dict[str, Any]) -> bool:
            with self._lock:
                self._seen_trajectories += 1
            # Trigger evolution every `skill_update_freq` *trajectories*.
            # (AReaL calls this once per accepted trajectory; we approximate
            #  the "every N training steps" cadence by trajectory count, which
            #  is monotonic and frame-rate independent.)
            self.maybe_evolve()
            return True

        return should_accept_fn

    # ------------------------------------------------------------------ #
    # Evolution driver                                                    #
    # ------------------------------------------------------------------ #

    def maybe_evolve(self) -> bool:
        """Trigger skill evolution if enough trajectories have accumulated.

        Returns True if an update was attempted (regardless of whether new
        skills were actually produced).
        """
        with self._lock:
            n = self._seen_trajectories
            if n - self._last_evolve_at < self.skill_update_freq:
                return False
            self._last_evolve_at = n
            failed = list(self._failed)
            self._failed.clear()

        if not failed:
            logger.info(
                "[SkillEvolution] cadence reached but no failed trajectories; skip"
            )
            return False

        # Compute success rate over the window since last evolve.
        # ``failed`` holds only failures; total window size == freq (trajectories).
        # Use the recorded failure count vs. cadence as a proxy for success rate.
        # If failure rate exceeds (1 - update_threshold), evolve.
        failure_rate = len(failed) / max(self.skill_update_freq, 1)
        success_rate = 1.0 - failure_rate
        if success_rate >= self.update_threshold:
            logger.info(
                f"[SkillEvolution] success_rate~{success_rate:.2f} >= "
                f"threshold {self.update_threshold}; skip update"
            )
            return False

        logger.info(
            f"[SkillEvolution] success_rate~{success_rate:.2f} < "
            f"{self.update_threshold}; analyzing {len(failed)} failures..."
        )
        return self._evolve(failed)

    def _evolve(self, failed: list[dict[str, Any]]) -> bool:
        """Run the SkillUpdater and write new skills into the shared memory."""
        try:
            if self.skill_updater is None:
                self.skill_updater = SkillUpdater(
                    max_new_skills_per_update=self.max_new_skills,
                )
            new_skills = self.skill_updater.analyze_failures(
                failed_trajectories=failed[: self.max_failures_to_analyze],
                current_skills=self.memory.skills,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[SkillEvolution] SkillUpdater error: {e}")
            return False

        if not new_skills:
            logger.info("[SkillEvolution] no new skills generated")
            return False

        added = self.memory.add_skills(new_skills, category="general")
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"updated_skills_{self._seen_trajectories}.json")
        self.memory.save_skills(save_path)

        self.update_history.append(
            {
                "n_seen": self._seen_trajectories,
                "n_failures_analyzed": len(failed[: self.max_failures_to_analyze]),
                "n_skills_added": added,
                "skill_ids": [s.get("skill_id") for s in new_skills[:added]],
                "save_path": save_path,
            }
        )
        logger.info(
            f"[SkillEvolution] added {added} new skills -> {save_path}"
        )
        return True

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seen_trajectories": self._seen_trajectories,
                "pending_failures": len(self._failed),
                "n_updates": len(self.update_history),
                "total_skills_added": sum(h["n_skills_added"] for h in self.update_history),
                "skill_count": self.memory.get_skill_count(),
            }
