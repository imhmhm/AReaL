# SkillRL 迁移到 AReaL 框架方案

> 目标：把 SkillRL 的 skill-augmented RL 算法迁到用户的 `AReaL` 框架（已原生支持 NPU）。
> 配套阅读：`docs/SkillRL_结构分析.md`、`docs/SkillRL_适配verl_v0.7.1_NPU方案.md`。

---

## 1. 结论先行

- **AReaL 是比 verl 更适合承载 SkillRL 的框架**：它的核心抽象 `RolloutWorkflow.arun_episode(engine, data)` 让 **workflow 完全接管多轮 rollout**，引擎只提供 `agenerate` 异步生成能力。这正好是 SkillRL 的 env-driven 风格，且比 verl-agent 的"driver 跑 env + 跨进程调 worker"更干净——**env 直接在 workflow 里调 engine.agenerate，没有 pad/unpad/跨进程往返**。
- **迁移本质**：把 SkillRL/verl-agent 的 env-driven loop 重写成一个 `RolloutWorkflow` 子类，env 嵌在里面；技能库逻辑（SkillRL 原创、与 verl 解耦）几乎零改搬入；技能进化钩子挂在 trainer 的 rollout→advantage 之间。
- **三大适配点**：① 把 env-driven loop 改写成 `async arun_episode(engine, data)`；② 技能进化钩子的挂载点从 verl 的 `ray_trainer.fit` 换到 AReaL 的 `rl_trainer.train`（需找到合适的钩子或加一个）；③ `*_success_rate` 这类 per-group 指标怎么从 workflow 透传到 trainer（AReaL 的 rollout_batch 字段约定）。

---

## 2. AReaL 核心抽象（迁移落点）

### 2.1 三件套

| 抽象 | 位置 | 契约 | SkillRL 对应 |
|---|---|---|---|
| **RolloutWorkflow** | `areal/api/workflow_api.py` | `async arun_episode(engine, data) → dict\|None` | SkillRL 的 `TrajectoryCollector.multi_turn_loop` |
| **InferenceEngine** | `areal/api/engine_api.py` | `async agenerate(ModelRequest) → ModelResponse` | verl 的 `actor_rollout_wg.generate_sequences`（但 async、单次生成、不跨进程） |
| **PPOTrainer** | `areal/trainer/rl_trainer.py` | `train()`：rollout→reward→compute_advantages→ppo_update；`group_size=config.gconfig.n_samples` | verl 的 `RayPPOTrainer.fit` |

### 2.2 数据流（与 verl 对比）

```
verl-agent:
  driver: envs.reset→build_text_obs→pad_dataproto→actor_rollout_wg.generate_sequences(Ray)
          →unpad→decode→envs.step   ← 每步跨进程往返一次

AReaL:
  workflow.arun_episode(engine, data):     ← 单条轨迹，在 async runtime 里
    env.reset()
    for step in range(max_steps):
        obs = build_text_obs(env_obs)      ← 技能注入在这
        resp = await engine.agenerate(req) ← 直接拿 token+logprob，无 pad/unpad
        action = decode(resp.output_tokens)
        env_obs, reward, done = env.step(action)
    return {input_ids, logprobs, loss_mask, versions, rewards, ...}
  trainer: prepare_batch 并发跑 group_size 个 episode → rollout_batch
         → compute_advantages(rollout_batch) → ppo_update
```

**关键简化**：AReaL 的 `arun_episode` 自己拼 `seq/logprobs/loss_mask`，引擎返回的 `resp.output_tokens/output_logprobs` 直接累积，无需 verl-agent 那套 `pad_dataproto_to_divisor`/`adjust_batch`/`to_list_of_dict`/`gather_rollout_data`。SkillRL 的 `rollout_loop.py`/`utils.py` 大部分**可以不要**，只保留 env/projection/memory 逻辑。

---

## 3. 代码归属与工作量

按 `docs/SkillRL_结构分析.md` 的三层：

| 层 | 来源 | AReaL 迁移动作 | 工作量 |
|---|---|---|---|
| ③ 技能库 | SkillRL 原创 | **零改直接搬**：`memory/skills_only_memory.py`、`memory/skill_updater.py`、`memory_data/*`、`skill_generation/*` | 极小 |
| ② env-driven | verl-agent | **重写**：envs/projection/prompts 搬入；`rollout_loop` 改写为 `RolloutWorkflow`；`reward_manager/episode.py` 退化为 reward 标量；`gigpo` 视需要 | 中 |
| ① 框架 | AReaL（非 verl） | 用 AReaL 原生 trainer/engine/config，**不碰 verl** | 0（用现成） |

---

## 4. 路线 B'（AReaL 专属）：写一个 SkillEnvWorkflow

### 4.1 新增文件结构（建议：全部放 `examples/skillrl/`，自包含）

> **设计纪律（对齐 AReaL 惯例，见 §4.1.0）**：任务专属逻辑（env 状态机、技能库、技能进化）**全部放 examples，不进 `areal/` 核心**。`areal/workflow/` 只收通用机制 workflow（rlvr/multi_turn/各 SDK 适配），任务专属进核心会违反扩展性原则。SkillEnvWorkflow/env/memory 都属任务专属，故全放 examples。只依赖 areal 稳定公开 API（`PPOTrainer`/`RolloutWorkflow`/`AsyncRewardWrapper`/`get_custom_dataset`/`ArealOpenAI`/`load_expr_config`），参考范本 `examples/multi_turn_math/`。

```
AReaL/
  examples/skillrl/                     ← ★ 全部任务代码放这，自包含（参考 multi_turn_math 范式）
    train.py                            ← 极简: PPOTrainer(config).train(workflow="...SkillEnvWorkflow", workflow_kwargs=...)
    configs.py                          ← SkillRLConfig(GRPOConfig) 子类 + env/skills_only_memory 字段（不改 areal/api/cli_args.py）
    skill_env_workflow.py               ← SkillEnvWorkflow(RolloutWorkflow) + SingleEnvAdapter（§4.2 骨架）
    reward_fn.py                        ← 轨迹级奖励（从 EpisodeRewardManager 退化为标量），字符串路径引用
    skill_hooks.py                      ← 技能进化 controller（§4.3，可选内联进 workflow）
    env_package/                        ← 从 SkillRL 搬（envs/projection/prompts），去 verl 依赖
        alfworld/, webshop/, search/, projection.py
    memory/                             ← 从 SkillRL 零改搬
        skills_only_memory.py, skill_updater.py, base.py
    memory_data/                        ← 技能库 JSON
    config_alfworld.yaml / config_search.yaml / ...
  areal/                                ← ★ 不动！只用现成核心，不新增任何文件
```

#### 4.1.0 AReaL examples 的复用边界（调研结论，决定本节布局）

实证 5 个 example（openclaw/tau2/search_agent/multi_turn_math/agent_workflow）的 import，结论：

| 复用的 areal 核心（稳定，不动） | examples 自包含（任务专属，搬过来） |
|---|---|
| `PPOTrainer`、`load_expr_config`、`GRPOConfig`/`PPOConfig` | 任务专属 `Config(PPOConfig)` 子类 |
| `RolloutWorkflow`(ABC)、`ArealOpenAI`（`experimental.openai`） | 任务专属 `XxxWorkflow(RolloutWorkflow)` + agent |
| `AsyncRewardWrapper`、`get_custom_dataset`、`load_hf_tokenizer`、`workflow_context`、`stats_tracker` | 任务专属 `reward_fn` + prompt + tool |

**边界原则**：`areal/workflow/` 只收**通用机制** workflow（rlvr/multi_turn/openai/anthropic/langchain/openai_agent/math_agent），即"与具体任务无关的可复用模板"。**任务专属逻辑（env 状态机、特定 reward、技能库）永不进核心，放 examples**。这样：① 不污染框架，下游 fork 改任务不动 `areal/`；② 零耦合可迁移，整个 `examples/skillrl/` 拷走即可跑；③ 多任务共存互不干扰。

**最佳范本 `examples/multi_turn_math/`**：整个任务一个 `gsm8k_rl_mt.py`（Workflow+agent+reward_fn+Config 子类+main）+ 一个 yaml，零散资源，纯依赖 areal 稳定 API。SkillRL 应照此自包含。

**反例对照 `openclaw`**：走 proxy 网关、外部 agent 连入、`train.py` 极简——**不适合 SkillRL 参考**，因 SkillRL 是自托管 env 状态机。


### 4.2 SkillEnvWorkflow 骨架（核心改写）

> **前置澄清（基于深度调研）**：AReaL 的并发与 group 由框架自动完成，**workflow 只需写"单条轨迹"逻辑**。具体机制见 §4.2.1。

对照 `areal/workflow/multi_turn.py` 的 `MultiTurnWorkflow`，把 SkillRL 的 `vanilla_multi_turn_loop` 翻译成 async 单轨迹版：

### 4.2.1 「批量向量化 → 单轨迹 async」机制深度调研

这是 AReaL 路线最关键的适配点，调研清楚后比预想简单。**核心结论：AReaL 的并发/group/padding 全在框架层自动完成，workflow 只需写单条轨迹逻辑；SkillRL 的向量化 env 需退化为"单 env 单轨迹"。**

**两端形态对比：**

| | SkillRL / verl-agent | AReaL |
|---|---|---|
| **env 形态** | 向量化 gym env：`SearchMultiProcessEnv` 一次持有 `env_num×group_n` 个子 env，`reset(kwargs:List[dict])`/`step(actions:List[str])` 批量操作 | 单轨迹：每个 `arun_episode` 一个 env 实例，`reset()`/`step()` 单条 |
| **group 来源** | `env.rollout.n`，在 `vanilla_multi_turn_loop` 里 `i % env.rollout.n` 手动分 uid | `config.gconfig.n_samples`，框架用 `GroupedRolloutWorkflow` 自动 `asyncio.gather` |
| **batch 装配** | driver 手动：`pad_dataproto_to_divisor`→`generate_sequences`(跨进程)→`unpad`→`gather_rollout_data`→`adjust_batch` copy/delete 对齐 DP | 框架自动：`prepare_batch` 返回 `list[dict]`，`concat_padded_tensors` 按 `attention_mask` 找 max_len 自动 padding |
| **reward** | `EpisodeRewardManager` 把 episode reward 放到最后有效 token | workflow 返回 `rewards` 标量（1D），框架对 1D 直接 concat 不 pad |

**AReaL 完整执行链路（已读源码确认）：**

```
trainer.train()  (rl_trainer.py:363)
  rollout_batch = self.actor.prepare_batch(
      dataloader, workflow, group_size=config.gconfig.n_samples, dynamic_bs=...)
    │ fsdp_engine/sglang_remote/... prepare_batch → 委托 RemoteInfEngine
    ▼
RemoteInfEngine.prepare_batch (remote_inf_engine.py:571)
  if group_size > 1:
      workflow = GroupedRolloutWorkflow(workflow, group_size)   ← ★ group 包装
  → WorkflowExecutor.prepare_batch
    │
    ▼
WorkflowExecutor.prepare_batch (workflow_executor.py:1262)
  data_generator = (对每个 dataloader item yield 一个 _RolloutTaskInput)
  results = dispatcher.active_submit_and_wait(data_generator, batch_size)
    │  BatchTaskDispatcher 维护输入队列/结果队列，控制 staleness，
    │  异步提交 + 等待，直到收齐 batch_size 条结果
    ▼
每个 task → _execute_workflow (workflow_executor.py:1029)
  traj = await pending_task.workflow.arun_episode(self.inference_engine, data)
     │  若被 GroupedRolloutWorkflow 包装：
     ▼
GroupedRolloutWorkflow.arun_episode (remote_inf_engine.py:75)
  results = await asyncio.gather(        ← ★ group_size 条并发跑同一 data
      *[self.workflow.arun_episode(engine, data) for _ in range(self.group_size)])
  valid = [r for r in results if r is not None]
  return concat_padded_tensors(valid)    ← ★ group 内自动 padding 拼接
     │
     ▼
should_accept_fn(traj) → True 接受 / False+None 拒绝（dynamic sampling）
  → _RolloutResult(trajectory=traj)
     │
     ▼
prepare_batch 返回 list[dict]（每条一条轨迹，变长）
  → trainer 里各 step：compute_values/compute_logp/compute_advantages/ppo_update
     这些方法的入参 rollout_batch 已是 padding 好的 dict[str, tensor]（concat_padded_tensors 产出）
```

**关键发现（修正初版误解）：**

1. **group 不用 workflow 管**——`GroupedRolloutWorkflow` 用 `asyncio.gather` 并发跑 group_size 次 `arun_episode(同一 data)`，再把结果 `concat_padded_tensors` 拼上。**同一个 data 被 group_size 个独立 env 实例并发跑**（不是 SkillRL 那种"向量化 env 一次跑 group_n 条"）。
2. **padding 不用 workflow 管**——`concat_padded_tensors`（`areal/utils/data.py:166`）遍历轨迹，按 `attention_mask` 找 max_len，对 `input_ids/logprobs/loss_mask/versions` 补 `pad_value`，对 `attention_mask` 补 0；**1D 张量（如 `rewards` 标量）直接 concat 不 pad**。这正是 `MultiTurnWorkflow` 返回 `{input_ids: [1,L], logprobs:[1,L], loss_mask:[1,L], versions:[1,L], rewards:[1] (标量 unsqueeze 后), attention_mask:[1,L]}` 的原因。
3. **变长天然支持**——每条轨迹长度不同（env 提前 done 即停），框架自动 padding 对齐，无需 SkillRL 的 `adjust_batch` copy/delete。
4. **dynamic sampling 原生**——`should_accept_fn` 返回 False + workflow 返回 None = 拒绝该轨迹，`dynamic_bs=True` 时继续生成直到收够 accepted。**对应 SkillRL 的 `filter_group_data`（DAPO 风格）**，可用 `should_accept_fn` 实现"同组 reward 全相同则拒绝"。
5. **每轨迹独立 env 实例**——`GroupedRolloutWorkflow` 的 `asyncio.gather` 里每次 `arun_episode` 会创建自己的 env；SkillRL 向量化 env 的"一次 reset 多条"语义被拆成"多条独立 reset"。

**对 SkillRL env 的具体改造（重点）：**

SkillRL 的 `SearchMultiProcessEnv` 等向量化 env（持有 N 个子 env、用 ThreadPoolExecutor+asyncio 批量 step）在 AReaL 下**整层向量化失去意义**——因为 `arun_episode` 只处理单条。改造有两种方案：

- **方案 X1（推荐，最小改动）**：写一个**单 env 适配器**，包住 SkillRL env 的单个子 env：
  ```python
  class SingleEnvAdapter:
      """把 SkillRL 向量化 env 的第 0 个子 env 暴露成单条接口。"""
      def __init__(self, build_vec_env, idx=0):
          self._build = build_vec_env   # 返回 SearchMultiProcessEnv(env_num=1, group_n=1)
      def reset(self, kwargs):           # kwargs: 单 dict（AReaL data 里是单条）
          env = self._build()            # env_num=1,group_n=1 → 内部只 1 个子 env
          self.env = env
          obs_list, info_list = env.reset([kwargs])   # 复用批量接口，传 1 条
          return obs_list[0], info_list[0]
      def step(self, action):
          obs_list, r_list, d_list, info_list = self.env.step([action])
          return obs_list[0], r_list[0], d_list[0], info_list[0]
      def success_evaluator(self, ...): ...   # 单条版
      def close(self): self.env.close()
  ```
  好处：**SkillRL 的 envs.py / projection.py / env_package 几乎零改**，只在外面包一层；向量化 env 的内部并发（ThreadPoolExecutor）退化为 size=1，无害。
- **方案 X2（更彻底）**：直接改写 `SearchMultiProcessEnv` 提供 `single_reset/single_step` 接口。改动面大，不推荐。

> 注：SkillRL env 的 `success_evaluator` 原签名是批量（`total_infos: List[List]`），单条版要相应改成处理单轨迹的 step 列表。这是 X1 适配器要补的少量逻辑。



```python
class SkillEnvWorkflow(RolloutWorkflow):
    def __init__(self, env_factory, projection_f, memory: SkillsOnlyMemory,
                 tokenizer, gconfig, max_steps):
        # env_factory: 返回 SingleEnvAdapter（包住 SkillRL 单 env），每次 arun_episode 调一次
        self.env_factory = env_factory
        self.projection_f = projection_f        # search_projection 等（原样用）
        self.memory = memory                    # SkillsOnlyMemory（零改搬入）
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.max_steps = max_steps

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        # ★ 单条轨迹！group_size 由框架用 GroupedRolloutWorkflow 并发跑多条，无需在此处理
        env = self.env_factory()                 # 单 env 实例
        kwargs = data.get("env_kwargs")
        obs, info = env.reset(kwargs)            # 单条 reset（SingleEnvAdapter）

        # ★ 技能检索（reset 时一次）：原样用 SkillsOnlyMemory
        task_desc = self._extract_task(obs)      # 单条 task
        retrieved = self.memory.retrieve(task_desc, top_k=...)

        seq, logprobs, loss_mask, versions = [], [], [], []
        episode_reward = 0.0
        done = False
        step_infos = []

        for step in range(self.max_steps):
            # ★ 技能注入：build_text_obs 里 format_for_prompt 拼进 *_TEMPLATE_WITH_MEMORY
            text_obs = self._build_text_obs(env, retrieved, init=(step == 0))
            input_ids = self._tokenize(text_obs)          # apply_chat_template → list[int]
            input_len = len(input_ids)

            req = ModelRequest(rid=uuid4().hex, input_ids=input_ids,
                                gconfig=self.gconfig.new(n_samples=1), tokenizer=self.tokenizer)
            resp = await engine.agenerate(req)             # ★ async，直接拿 token+logprob

            # 单条 decode + projection（projection_f 支持单元素 list）
            action_str = self.tokenizer.decode(resp.output_tokens)
            actions, valids = self.projection_f([action_str])
            next_obs, reward, done, step_info = env.step(actions[0])
            episode_reward += float(reward)                # 轨迹级奖励聚合
            step_infos.append(step_info)

            # 累积 token 级训练数据（照 multi_turn.py 的拼法；prompt 段 mask=0，response 段 mask=1）
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0]*input_len + resp.output_logprobs
            loss_mask += [0]*input_len + [1]*resp.output_len
            versions += [-1]*input_len + resp.output_versions

            if done:
                break

        # success_evaluator：单条版（适配器提供）
        success = env.success_evaluator(step_infos, episode_reward)  # {f"{task}_success_rate": float}
        env.close()

        res = dict(
            input_ids=torch.tensor(seq, dtype=torch.int32),
            logprobs=torch.tensor(logprobs, dtype=torch.float32),
            loss_mask=torch.tensor(loss_mask, dtype=torch.int32),
            versions=torch.tensor(versions, dtype=torch.int32),
            rewards=torch.tensor(episode_reward, dtype=torch.float32),  # 1D 标量，concat 时不 pad
            attention_mask=torch.ones(len(seq), dtype=torch.bool),        # 框架据此 padding
            # ★ 透传 skill 进化需要的 per-trajectory 指标（non-tensor，见 §4.3）
            success_rate=success,
            task_type=self.memory._detect_task_type(task_desc),
            prompt_str=task_desc,             # 给 _collect_failed_trajectories 用
        )
        return {k: v.unsqueeze(0) for k, v in res.items()}   # 每项 [1, ...]
```

**要点（基于调研修正）**：
- **单条语义**：`arun_episode` 只处理一条轨迹，所有 list/batch 操作改成单元素。`GroupedRolloutWorkflow` 会在 group 维度并发跑多条 + `concat_padded_tensors` 自动 padding，**workflow 完全不碰 group/padding**。
- `engine.agenerate` 取代 `actor_rollout_wg.generate_sequences`，async、无 pad、无跨进程往返。
- 技能检索/注入逻辑（`SkillsOnlyMemory.retrieve`/`format_for_prompt`/`_TEMPLATE_WITH_MEMORY`）**原样复用**，只是调用上下文从 verl `EnvironmentManager` 搬到 workflow 的 `_build_text_obs`。
- 返回 dict 形态必须含 `attention_mask`（`concat_padded_tensors` 依赖它找 max_len）且 `rewards` 为 1D（标量，避免被 pad）。non-tensor 字段（`success_rate` 等）需特殊处理见 §4.3。


### 4.3 技能进化钩子（核心适配点②）

SkillRL 在 verl `ray_trainer.fit` 里有两个钩子。AReaL 的 `rl_trainer.train` 结构不同（`prepare_batch`→`compute_advantages`→`ppo_update`），需重新定位：

| SkillRL 钩子 | 触发条件 | AReaL 挂载点（建议） |
|---|---|---|
| `_update_skills_from_training` | `*_success_rate < threshold` & 按 freq | `prepare_batch` 之后、`compute_advantages` 之前，加一个 trainer 钩子；或包在 `SkillEnvWorkflow` 外层 controller |
| `_update_skills_from_validation` | validation 后 | `eval` 路径（rl_trainer 里 `eval_workflow` 调用后） |

**实现选项**：
- **选项 1（推荐，低侵入）**：`SkillEnvWorkflow` 持有 `memory` + `skill_updater`，在 `arun_episode` 结束时把失败轨迹写入一个共享 buffer；trainer 侧加一个轻量 `SkillEvolutionHook`，按 `skill_update_freq`/`test_freq` 扫 buffer、调 `SkillUpdater.analyze_failures`、`memory.add_skills`。这样钩子逻辑（SkillRL 原创 ~200 行）几乎原样搬，只是触发判断从"读 verl batch.non_tensor_batch 的 success_rate"改成"读 workflow 透传的 success_rate"。
- **选项 2**：直接 monkey-patch / 继承 `PPOTrainer` 重写 `train`。更重，不推荐。

**防泄漏仍成立**：AReaL 的 train workflow 与 eval workflow 是两个实例（`workflow` vs `eval_workflow`），各自持 `memory` 实例——`add_skills` 只写 train 的 memory，eval 的不变，沿用 SkillRL 设计。

### 4.4 config（放 examples，字符串路径引用，对齐 multi_turn_math 惯例）

AReaL 用 `PPOConfig`（`areal/api/cli_args.py`）+ yaml。**关键：workflow/reward 用字符串路径引用（跟 `multi_turn_math` 完全一致），Config 用 `SkillRLConfig(GRPOConfig)` 子类加字段，不直接改 `areal/api/cli_args.py`**。

`examples/skillrl/config_alfworld.yaml`：
```yaml
# workflow 用字符串路径（框架按需 import），不放 areal.workflow 下
workflow: examples.skillrl.skill_env_workflow.SkillEnvWorkflow
eval_workflow: ${workflow}

workflow_kwargs:
  env_name: alfworld
  max_steps: 50
  # reward_fn 也用字符串路径（对齐 multi_turn_math 的 gsm8k_reward_fn）
  reward_fn: examples.skillrl.reward_fn.alfworld_reward_fn
  skills_only_memory:
    skills_json_path: examples/skillrl/memory_data/alfworld/claude_style_skills.json
    retrieval_mode: template          # NPU 先 template
    top_k: 6
    enable_dynamic_update: true
    update_skills_from_train: true
    update_threshold: 0.4
    max_new_skills: 3
    skill_update_freq: 5

gconfig:
  n_samples: 8                           # GRPO group（取代 env.rollout.n，框架用 GroupedRolloutWorkflow 自动并发）
# trainer.device 等 NPU 设置走 AReaL 原生
```

`examples/skillrl/configs.py`（Config 子类，加 env/skills 字段而不动核心）：
```python
from areal.api.cli_args import GRPOConfig
from dataclasses import dataclass, field

@dataclass
class SkillRLConfig(GRPOConfig):
    env: dict = field(default_factory=dict)             # env_name, max_steps 等
    skills_only_memory: dict = field(default_factory=dict)  # 技能库配置
```

> CLAUDE.md 提醒：修改 `areal/api/cli_args.py` 的 config 结构前要 "Ask First"。**通过 `SkillRLConfig` 子类 + workflow_kwargs 透传，避免动核心 config**（与 `multi_turn_math` 的 `MultiTurnGRPOConfig(GRPOConfig)` + `agent_run_args` 同款手法）。


---

## 5. 三框架对比（verl 0.3.1 / verl_v0.7.1 / AReaL）

| 维度 | SkillRL(0.3.1) | verl_v0.7.1 | AReaL |
|---|---|---|---|
| env-driven 多轮 | verl-agent 自有 | 需移植（绕 async 硬编码） | **原生**（workflow 接管） |
| 抽象 | driver loop + Ray worker | AgentLoopManager(tool-calling) | **RolloutWorkflow.arun_episode** |
| GRPO group | `env.rollout.n` 手动 | `env.rollout.n` | **`config.gconfig.n_samples`** 原生 |
| pad/unpad/对齐 | verl-agent 自有 | 同 | **不需要**（workflow 自拼） |
| GiGPO | 有 | 缺失需注册 | 查 AReaL adv 支持列表 |
| NPU | 无 | 原生 | **原生**（davinci/vlm_npu） |
| 技能库移植 | — | 第③层零改 | 第③层零改 |
| 钩子挂载 | `ray_trainer.fit` | 改写后 fit | `rl_trainer.train` |
| 改动文件数 | — | ~1(ray_trainer)+胶水 | 新增 workflow+memory+hooks |

**判断**：AReaL 的 workflow 抽象让 SkillRL 的 env-driven 风格**原生落地**，省掉 verl-agent 整套 pad/unpad/group 胶水（`rollout_loop.py`/`utils.py` 大半可删）。迁移的"重活"从 verl 路线的"改 ray_trainer 绕 async"变成"把 env loop 重写成 async arun_episode"——后者是写新代码而非改老代码，更可控。

---

## 6. 风险与陷阱

| # | 风险 | 应对（基于调研已细化） |
|---|---|---|
| 1 | AReaL `arun_episode` 是**单轨迹 async**，SkillRL env 是**批量向量化** | **方案 X1（推荐）**：写 `SingleEnvAdapter` 包住 `SearchMultiProcessEnv(env_num=1,group_n=1)`，把批量 `reset([kwargs])`/`step([action])` 暴露成单条。SkillRL envs.py/projection 几乎零改，向量化内部并发退化为 size=1 无害。`success_evaluator` 改单条版（处理单轨迹 step 列表）。 |
| 2 | 轨迹级奖励 vs token 级 loss_mask | 照 `multi_turn.py`：`rewards` 返回 1D 标量（`concat_padded_tensors` 对 1D 直接 concat 不 pad），`loss_mask` 标记可训练 token（env observation 段 mask=0，response 段 mask=1）。`EpisodeRewardManager` 退化为"返回 episode_reward 标量"。 |
| 3 | `*_success_rate` 透传 | workflow 返回 dict 里加 `success_rate`(dict)/`task_type`/`prompt_str` 等 non-tensor 字段。**调研发现** `concat_padded_tensors` 只处理 tensor 字段，non-tensor 字段需走 §4.3 的共享 buffer，不依赖框架聚合。 |
| 4 | 技能进化触发点无现成钩子 | 用选项 1：workflow 在 `arun_episode` 末尾写失败轨迹到共享 buffer；trainer 侧 `SkillEvolutionHook` 按 freq 扫 buffer 调 o3。 |
| 5 | GiGPO 是否需要 | 先用 GRPO（AReaL `group_size` 原生，`GroupedRolloutWorkflow` 并发）；GiGPO step-level advantage 需确认 AReaL `compute_advantages` 是否支持 step reward。 |
| 6 | NPU embedding 检索 | 同 verl 方案：`SkillsOnlyMemory` 加 device 透传；NPU 阶段强制 template。 |
| 7 | env 依赖重装 | AReaL 容器复用 SkillRL `setup.sh`；search 最轻先迁。 |
| 8 | `arun_episode` 返回 None = 拒绝轨迹 | **调研确认**对应 `should_accept_fn`+`dynamic_bs`；SkillRL `filter_group_data`（同组 reward 全相同则拒）可用 `should_accept_fn` 实现。 |

---

## 7. 落地顺序与实际完成情况

实际在 `AReaL` 的 `skill_rl` 分支（基于 `ascend-v1.0.1`）上完成，全部代码位于 `examples/skillrl/`（自包含，`areal/` 核心零改动）。

| # | 里程碑 | 状态 | 关键产出 |
|---|---|---|---|
| 1 | search 环境 + template 模式 + GRPO + NPU | ✅ 完成 | `SkillEnvWorkflow` + `SingleEnvAdapter`；技能注入链路端到端验证通过（技能库加载 41 条、检索/格式化、env↔engine 多轮、EM 奖励） |
| 2 | 技能进化钩子（支柱 C） | ✅ 完成 | `SkillEvolutionController`，经 `trainer.train(dynamic_filter_fn=)` 挂载；失败轨迹收集→按 cadence+threshold 触发 o3→写回 train memory+落盘；防泄漏（train/eval 独立 memory） |
| 3 | embedding 模式 NPU 适配 | ✅ 完成 | `embedding_device` 透传 + 双重 fallback（NPU→CPU、模型不可获取→template）；`torch.device('npu')`/`torch_npu`/`sentence_transformers` 验证可用 |
| 4 | alfworld/webshop 环境 | ✅ 完成 | 由用户在分支内完成 |
| 5 | GiGPO step-level advantage | ⏸️ 暂缓 | 见 §7.2 架构冲突分析；当前用 GRPO（group_size 归一化）已满足 |

AReaL 路线相比 verl_v0.7.1 路线，**省掉了改 ray_trainer 绕 async 这一最大风险点**；"env 批量向量化→单轨迹 async"的适配经调研收敛为「写一个 `SingleEnvAdapter` 包住 `env_num=1,group_n=1` 的向量化 env」，envs.py/projection 近乎零改，工作量可控。

### 7.1 实际适配产出（`examples/skillrl/`）

```
examples/skillrl/
  train.py                 # 装配: PPOTrainer.train(workflow=SkillEnvWorkflow, dynamic_filter_fn=...)
  configs.py               # SkillRLConfig(GRPOConfig) + env/skills_only_memory 字段
  config_search.yaml       # NPU/vLLM GRPO config (template 检索 + 进化开启)
  skill_env_workflow.py    # SkillEnvWorkflow(RolloutWorkflow) + SingleEnvAdapter
  skill_hooks.py           # SkillEvolutionController (支柱C, 经 dynamic_filter_fn 挂载)
  reward_fn.py             # 轨迹级奖励
  dataset.py               # 自包含 search QA loader
  memory/                  # SkillsOnlyMemory + SkillUpdater (零改搬入 + NPU fallback)
  env_package/             # search env + skyrl_gym + prompts (gym 依赖已去除)
  memory_data/search/      # 技能库 JSON (key 归一化为 task_specific_skills)
  README.md                # 用法 + 验证记录 + 里程碑
```

**已验证**（import + 单元级，均通过）：`SkillEnvWorkflow` 是 `RolloutWorkflow` 子类；技能库加载/检索/格式化；`SingleEnvAdapter` 驱动 `SearchEnv` 产出 EM 奖励；`search_projection` 解析；prompt 构建（init/技能注入/history）；进化控制器端到端（41→42 技能、触发条件、落盘、防泄漏共享 memory）。

### 7.2 GiGPO 暂缓的架构冲突分析（里程碑5）

调研后确认 GiGPO 在 AReaL 上存在**根本性架构冲突**，非简单实现问题：

**核心矛盾**：GiGPO 需要 **step-level 数据**（每步 `anchor_obs` 锚点 + `step_rewards` + `traj_uid`/`active_masks`，见 `gigpo/core_gigpo.py` 的 `compute_step_discounted_returns` 与 `compute_gigpo_outcome_advantage`），而 AReaL 是 **trajectory-level 拼接**（`concat_padded_tensors` 把多步拍平成一条序列，只有轨迹级标量 `rewards`）。

**两个具体障碍**：
1. **anchor_obs 传输**：GiGPO 的 step-level grouping（Eq.6 `build_step_group`）依赖每步 env 锚点状态（变长对象），`concat_padded_tensors` 只处理张量无法承载。这与里程碑2同类问题，但 anchor_obs 是 advantage 的**核心输入**，不能像 success_rate 那样走旁路。
2. **estimator 注入点**：AReaL `compute_advantages`（`areal/trainer/ppo/actor.py:129`）**硬编码 GAE**（`advantages_reversed` 循环），GRPO 是通过 `reward_norm` 的 group 归一化实现，**无 SkillRL 那种 `elif adv_estimator == GiGPO` 枚举分支**，无法直接注入 GiGPO。

**GiGPO vs GRPO 的增量**：`compute_gigpo_outcome_advantage` = `episode_advantages`（Eq.3，等价 GRPO）+ `step_advantage_w * step_advantages`（Eq.6-7，依赖 anchor_obs 分组）。即 GiGPO 相对 GRPO 的全部增量在 step-level 部分。

**决策**：当前用 GRPO（`config.gconfig.n_samples` 的 `GroupedRolloutWorkflow` group 归一化）已满足训练需求；GiGPO 暂缓。若未来确需 step-level 优势，需重构 workflow 返回 per-step 数据 + 自定义 advantage 函数绕过 GAE（工作量/风险高，可能触及 areal 核心），仅在 step-level 优势对任务确有收益时才值得。



---

## 8. 与 verl_v0.7.1 路线的取舍建议

- **若目标是"尽快在 NPU 上跑通 SkillRL"且 verl_v0.7.1 已就绪**：走 verl 路线（§5.3 改 ray_trainer + 搬 agent_system），改动集中在已知文件，风险点是 async 绕过。
- **若目标是"长期用 AReaL 作为主力 RL 框架"**：走 AReaL 路线，env-driven 与 AReaL 抽象天然契合，pad/group 胶水可删，更可持续；前期投入在写 `SkillEnvWorkflow`。
- **两路线共享**：第③层技能库代码、`skill_generation`、`memory_data` 完全通用，可先在任一框架验证技能逻辑，再决定框架。

---

## 9. 顶层设计结论：放 `examples/skillrl/` 而非 `areal/` 核心（回答"是否参考 examples 设计理念"）

**是，应参考 examples 设计理念，且 SkillRL 适配代码全部放 `examples/skillrl/`，不进 `areal/` 核心。**

### 9.1 AReaL 的分层契约（实证）

调研 5 个 example 的 import 后确认：AReaL 的扩展模式是 **"核心 `areal/` 只提供稳定机制 + examples 自包含任务"**。

- **`areal/` 提供稳定 API**（examples 只复用这些）：`PPOTrainer`、`RolloutWorkflow`(ABC)、`ArealOpenAI`、`AsyncRewardWrapper`、`get_custom_dataset`、`load_expr_config`、`GRPOConfig/PPOConfig`、`workflow_context`、`stats_tracker`。
- **`areal/workflow/` 只收通用机制 workflow**：`rlvr`/`multi_turn`/`openai`/`anthropic`/`langchain`/`openai_agent/math_agent`——即"与具体任务无关的可复用模板"。
- **examples 自包含任务逻辑**：env 状态机、特定 reward、prompt、tool、agent 类——全在各自目录。

### 9.2 为什么这样设计 = 扩展性（这正是 SkillRL 需要的）

SkillRL 是被大量下游工作进一步修改的基础。AReaL 的 examples 模式恰好提供这种扩展性：

1. **不污染框架**：下游 fork 改任务/技能时不动 `areal/` 核心，框架升级不破坏任务。
2. **零耦合可迁移**：整个 `examples/skillrl/` 目录拷走即可跑，只依赖 areal 稳定公开 API（无内部 API 耦合）。
3. **多任务/多实验共存**：alfworld/webshop/search 各起 config，互不干扰，共用一套 `PPOTrainer`。
4. **技能库天然适配**：SkillRL 的 memory/skill_updater 本就是"与 verl 解耦的第③层"，放进 examples 自包含，与 areal 的 examples 模式完全同构。

### 9.3 SkillRL 对应的分层映射

| SkillRL/verl-agent 原件 | AReaL 落点 | 说明 |
|---|---|---|
| `memory/skills_only_memory.py`、`skill_updater.py`（第③层，SkillRL 原创） | `examples/skillrl/memory/` | 零改搬入，自包含 |
| `env_package/*`（alfworld/webshop/search）、`projection.py`、`prompts/` | `examples/skillrl/env_package/` | 去 verl 依赖，配 `SingleEnvAdapter` |
| `multi_turn_rollout/rollout_loop.py`（verl-agent） | **删除**，改写为 `examples/skillrl/skill_env_workflow.py` 的 `SkillEnvWorkflow.arun_episode` | 胶水层被 areal 原生取代 |
| `reward_manager/episode.py` | `examples/skillrl/reward_fn.py`（标量化） | 轨迹级奖励，字符串路径引用 |
| `ray_trainer.py` 的 skill-update 钩子 | `examples/skillrl/skill_hooks.py` | trainer 外挂 controller |
| config | `examples/skillrl/configs.py`(`SkillRLConfig(GRPOConfig)`) + yaml | 不动 `areal/api/cli_args.py` |

### 9.4 设计纪律（对齐 multi_turn_math 范本）

照 `examples/multi_turn_math/`（最干净范本：一个 `.py` 含 Workflow+agent+reward_fn+Config 子类+main + 一个 yaml）：

1. **workflow 用字符串路径引用**：`trainer.train(workflow="examples.skillrl.skill_env_workflow.SkillEnvWorkflow", ...)`。
2. **reward_fn 用字符串路径引用**：`reward_fn="examples.skillrl.reward_fn.xxx"`。
3. **Config 子类化**：`SkillRLConfig(GRPOConfig)`，不直接改 `areal/api/cli_args.py`（CLAUDE.md 要求 Ask First）。
4. **不往 `areal/workflow/` 加 SkillEnvWorkflow**：它是任务专属（env 状态机+技能库），进核心违反扩展性原则。

**反例对照 `openclaw`**：走 proxy 网关、外部 agent 连入、`train.py` 极简——不适合 SkillRL 参考，因 SkillRL 是自托管 env 状态机。**正例范本 = `multi_turn_math`**。

