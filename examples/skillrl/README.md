# SkillRL on AReaL (Search task)

Skill-Augmented RL ported from [SkillRL](https://github.com/aiming-lab/SkillRL)
onto the AReaL framework. Self-contained per AReaL's examples convention
(see `examples/multi_turn_math/`): depends only on AReaL's stable public API,
adds nothing to the `areal/` core package.

> Design rationale: see `docs/SkillRL_迁移AReaL方案.md` in the SkillRL repo
> (three-layer dependency / single-trajectory async adaptation / examples
> boundary).

## What it does

- **Pillar B (skill injection)**: a `SkillsOnlyMemory` (template or embedding
  mode) retrieves Claude-style skills per task and injects them into the
  agent prompt every step via `format_for_prompt`.
- **Env-driven multi-turn rollout**: `SkillEnvWorkflow(RolloutWorkflow).arun_episode`
  drives a gym-style search env state machine, calling `engine.agenerate` per
  step. GRPO group / padding / async concurrency are handled by the framework
  (`GroupedRolloutWorkflow` + `concat_padded_tensors`), so the workflow only
  encodes a *single* trajectory.
- **Pillar C (skill evolution)**: a `SkillEvolutionController` collects failed
  trajectories (per-step `{action, observation}`) from the workflow and, at a
  configured cadence, drives `SkillUpdater.analyze_failures` (o3) to distil
  them into new `dyn_*` skills written back into the (train-only) shared
  `SkillsOnlyMemory`. Mounted via AReaL's native `trainer.train(dynamic_filter_fn=)`
  hook — **zero core changes**. Eval workflow keeps a separate memory instance
  so validation scores aren't inflated (anti-leakage, same as SkillRL).

## Layout

```
examples/skillrl/
  train.py                 # entry point: PPOTrainer.train(workflow="...SkillEnvWorkflow")
  configs.py               # SkillRLConfig(GRPOConfig) + env/skills_only_memory fields
  config_search.yaml       # NPU/vLLM GRPO config (template retrieval mode)
  skill_env_workflow.py    # SkillEnvWorkflow + SingleEnvAdapter (env-driven loop)
  reward_fn.py             # trajectory-level reward (env provides the score)
  dataset.py               # Search QA dataset loader (question/ground_truth/data_source)
  memory/                  # SkillsOnlyMemory + SkillUpdater (ported verbatim, verl-free)
  env_package/             # search env + skyrl_gym + prompts (ported, gym dep dropped)
  memory_data/search/      # Claude-style skill bank JSON
```

## Run

```bash
# Requires a Search retrieval backend at env.search.search_url (default
# http://127.0.0.1:8000/retrieve). Point train_dataset.path at a Search-R1
# style QA dataset (question/answer fields) in config_search.yaml.

python -m examples.skillrl.train --config examples/skillrl/config_search.yaml
```

## Verified (import + unit level)

- `SkillEnvWorkflow` is a `RolloutWorkflow` subclass; `SkillRLConfig` extends
  `GRPOConfig`.
- `SkillsOnlyMemory` loads the search skill bank (41 skills: 10 general + 20
  task-specific + 11 mistakes), `retrieve` + `format_for_prompt` work in
  template mode.
- `SingleEnvAdapter.reset/step/close` drive the underlying `SearchEnv`;
  `<answer>` termination yields the expected EM reward.
- `search_projection` extracts `<search>`/`<answer>` blocks and flags invalid
  (multi-tag) actions.
- Prompt building: init prompt (no skills) vs step prompt (skill-injected +
  history) both render correctly.
- **Skill evolution (pillar C)**: `SkillEvolutionController` —
  - `record_failure` → thread-safe buffer; `should_accept_fn` always returns
    `True` (旁路收集, never rejects trajectories).
  - Cadence trigger: every `skill_update_freq` trajectories, evolves only when
    recent success_rate `< update_threshold` (verified: success~1.0/0.75 → no
    evolve; success~0.25 → evolve).
  - End-to-end with a mock updater: 41 → 42 skills (1 `dyn_*` added), skill
    bank saved to JSON, `update_history` recorded.
  - Workflow + controller share the same `SkillsOnlyMemory` instance (verified
    identity), so new skills are immediately visible to subsequent rollouts.
- `train.py` assembly imports cleanly; train/eval get separate memory instances.

## Milestones

1. ✅ **Skill injection (pillar B)** on Search + template mode + GRPO + NPU/vLLM.
2. ✅ **Skill evolution (pillar C)**: `SkillEvolutionController` wired via
   `trainer.train(dynamic_filter_fn=)`. Failed trajectories recorded in-workflow;
   evolution triggered at cadence when success_rate < threshold; new skills
   written to the train-only memory + saved to disk. Eval memory kept separate.
3. ⬜ **Embedding mode** on NPU (`embedding_device="npu"`).
4. ⬜ **alfworld/webshop** envs.
5. ⬜ **GiGPO** step-level advantage (verify AReaL `compute_advantages` support).

> Note on pillar C cadence: AReaL's `should_accept_fn` is called once per
> accepted trajectory, so evolution is paced by trajectory count
> (`skill_update_freq`), approximating "every N train steps" (set
> `skill_update_freq ≈ batch_size` for ~1-evolve-per-step). For exact step
> alignment, the `global_step` could be read from `workflow_context` in a
> future refinement.
