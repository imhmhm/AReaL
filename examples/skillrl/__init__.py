"""SkillRL adaptation for AReaL.

Skill-augmented RL (recursive skill discovery + injection + evolution) ported
from https://github.com/aiming-lab/SkillRL onto the AReaL framework.

Design follows AReaL's examples convention (see ``examples/multi_turn_math/``):
all task-specific logic lives here and depends only on AReaL's stable public
API (``PPOTrainer``, ``RolloutWorkflow``, ``AsyncRewardWrapper``,
``get_custom_dataset``, ``load_expr_config``). Nothing is added to the
``areal/`` core package.

Modules
-------
- ``memory/``           : SkillsOnlyMemory + SkillUpdater (ported from SkillRL, verl-free)
- ``env_package/``      : gym envs + projection + prompts (ported from SkillRL)
- ``skill_env_workflow`` : SkillEnvWorkflow(RolloutWorkflow) — the env-driven multi-turn loop
- ``skill_hooks``       : SkillEvolutionController — recursive skill evolution (pillar C)
- ``reward_fn``         : trajectory-level reward (scalar)
- ``configs``           : SkillRLConfig(GRPOConfig)
"""

__all__ = []
