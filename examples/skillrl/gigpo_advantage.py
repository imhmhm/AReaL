"""GiGPO advantage computation for SkillRL on AReaL.

Faithful port of ``SkillRL/gigpo/core_gigpo.py`` (Eq. 3/5/6/7/8 of
arXiv:2505.10978), adapted to AReaL's tensor-only, positionally-grouped batch.

Adaptation deltas vs the original (math is identical; only the grouping keys
change, because AReaL carries no ``non_tensor_batch``):

- **Outer (episode) group = one prompt block.** AReaL's
  ``GroupedRolloutWorkflow`` runs ``n_samples`` trajectories of one prompt and
  ``concat_padded_tensors`` concatenates them into a contiguous
  ``[n_samples * max_steps, L]`` block; FFD whole-dict allocation keeps each
  block intact per DP rank. So the episode group is the contiguous block of
  ``group_size = n_samples * max_steps`` rows (the original keyed this by
  ``index``/uid; positional recovery is equivalent -- see
  ``SkillRL_样本组装与分组机制分析.md``).
- **Trajectory boundary within a block = every ``max_steps`` rows.** The
  original keyed this by ``traj_uid``; positional recovery is equivalent.
- **Inner (step) group = rows within a block sharing the same anchor.** The
  original clustered by ``anchor_obs`` string equality (``to_hashable``); we
  cluster by an int64 ``anchor_hash`` (blake2b of the obs text, 61-bit
  non-negative). int64 equality == string equality (collision negligible).
  Similarity mode (text-only ``SequenceMatcher``) is deferred to Phase 3 -- it
  cannot be expressed on the tensor-only path.

Return shape: unlike the original (which returns a token-level
``[bs, resp_len]`` tensor), this returns a **per-row scalar** ``[N]`` advantage
``A^E + ω·A^S``. The caller (``SkillFSDPPPOActor.compute_advantages``) writes it
into ``data["rewards"]`` and delegates to AReaL's GAE path
(``discount=gae_lambda=1``, ``values=0``), which broadcasts it uniformly to
every response token of the row -- exactly GiGPO's per-step outcome advantage
(GiGPO is outcome-based: no GAE, no value function).

Degradation guarantee (paper §3.3): when no anchor repeats within a prompt,
every step cluster has size 1 -> ``A^S = 0`` -> ``A = A^E`` = GRPO. So GiGPO's
lower bound is GRPO, automatically.
"""

from __future__ import annotations

import logging

import torch

_logger = logging.getLogger("gigpo_advantage")


# --------------------------------------------------------------------------- #
# Eq. 5: discounted return-to-go per trajectory                               #
# --------------------------------------------------------------------------- #
def compute_step_returns(
    step_rewards: torch.Tensor,
    group_size: int,
    max_steps: int,
    gamma: float,
) -> torch.Tensor:
    """Discounted return-to-go ``R_t = Σ_{k≥t} γ^{k−t} r_k`` (Eq. 5).

    ``step_rewards`` is the per-step env reward ``r_k`` for every row in the
    batch, laid out as contiguous prompt blocks of ``group_size`` rows, each
    block = ``n_samples`` contiguous trajectories of ``max_steps`` rows. We
    reshape to ``[num_blocks, n_samples, max_steps]`` and reverse-accumulate
    within each trajectory (traj j = dim 1).

    Padding rows (early-done trajectories padded to ``max_steps``) carry
    ``r_k = 0``, so they contribute zero discounted return -- ``R_t`` for real
    steps is unchanged, and padding rows get ``R_t = 0``.
    """
    N = step_rewards.shape[0]
    assert N % group_size == 0, f"N={N} not divisible by group_size={group_size}"
    num_blocks = N // group_size
    n_samples = group_size // max_steps
    assert (
        group_size == n_samples * max_steps
    ), f"group_size={group_size} != n_samples({n_samples}) * max_steps({max_steps})"

    sr = step_rewards.reshape(num_blocks, n_samples, max_steps).to(torch.float32)
    returns = torch.zeros_like(sr)
    running = torch.zeros(
        num_blocks, n_samples, device=sr.device, dtype=torch.float32
    )
    # Reverse recursion: R_t = r_t + γ * R_{t+1}. Equivalent to the original's
    # `running_return = traj_rewards[t] + gamma * running_return`.
    for t in reversed(range(max_steps)):
        running = sr[:, :, t] + gamma * running
        returns[:, :, t] = running
    return returns.reshape(N)


# --------------------------------------------------------------------------- #
# Generic per-group mean / (mean-std) normalization                           #
# --------------------------------------------------------------------------- #
def _group_norm(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    remove_std: bool,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Per-group normalization, matching core_gigpo's episode/step norm reward.

    - ``remove_std=True``  (``mode="mean_norm"``)    -> ``score − mean``
    - ``remove_std=False`` (``mode="mean_std_norm"``) -> ``(score − mean) / (std + ε)``

    ``group_ids`` must be contiguous ``0..G-1``. Size-1 groups yield 0 (their
    sole member equals the mean), matching the original's ``len==1`` branch
    (``mean=value, std=1`` -> ``score − score = 0``). ``std`` is unbiased
    (``torch.std`` default), matching the original.

    Vectorized via ``scatter_add`` (the original looped per group; this is the
    math-identical batched form -- repeats within a group do not change mean or
    std, so computing over all rows == over distinct trajectories).
    """
    scores = scores.to(torch.float32)
    group_ids = group_ids.long()
    num_groups = int(group_ids.max().item()) + 1

    ones = torch.ones_like(scores)
    counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)
    counts.scatter_add_(0, group_ids, ones)
    counts = counts.clamp(min=1.0)  # avoid div-by-zero (size-0 can't happen)

    sums = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)
    sums.scatter_add_(0, group_ids, scores)
    mean = sums / counts
    mean_per = mean[group_ids]

    if remove_std:
        return scores - mean_per

    sq_dev = (scores - mean_per).pow(2)
    sum_sq = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)
    sum_sq.scatter_add_(0, group_ids, sq_dev)
    # unbiased variance: sum_sq / (count - 1); size-1 -> denom clamped to 1, but
    # sq_dev is 0 there so std=0 -> (0)/(0+ε)=0 (== original's std=1 -> 0/1=0).
    var = sum_sq / (counts - 1.0).clamp(min=1.0)
    std_per = var[group_ids].clamp(min=0.0).sqrt()
    return (scores - mean_per) / (std_per + epsilon)


# --------------------------------------------------------------------------- #
# Eq. 3: episode advantage (group = prompt block)                             #
# --------------------------------------------------------------------------- #
def episode_advantage(
    rewards: torch.Tensor,
    group_size: int,
    remove_std: bool,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Episode-level advantage ``A^E(τ_i) = (R(τ_i) − mean_{prompt}) [/ std]``.

    The episode group is the contiguous prompt block (``group_size`` rows). The
    episode score is ``R(τ_i)`` -- the per-row ``rewards`` (every step-row of a
    trajectory carries the same trajectory outcome). This mirrors the original
    ``episode_norm_reward`` with ``compute_mean_std_cross_steps=True`` (default):
    every step-row is counted, which (since each trajectory's ``max_steps`` rows
    share one ``R``) is equivalent to normalizing over the ``n_samples``
    trajectories.
    """
    N = rewards.shape[0]
    assert N % group_size == 0, f"N={N} not divisible by group_size={group_size}"
    num_blocks = N // group_size
    # block_id[i] = which prompt block row i belongs to (contiguous groups).
    block_id = (
        torch.arange(num_blocks, device=rewards.device, dtype=torch.long)
        .repeat_interleave(group_size)
    )
    return _group_norm(rewards, block_id, remove_std, epsilon)


# --------------------------------------------------------------------------- #
# Eq. 6 + Eq. 7: step advantage (group = anchor cluster within a prompt)      #
# --------------------------------------------------------------------------- #
def build_step_group(
    anchor_hash: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Cluster rows by equal ``anchor_hash`` *within each prompt block* (Eq. 6).

    Returns a ``[N]`` int64 tensor of contiguous cluster ids (unique per
    (block, anchor)). Clustering is scoped to the block -- cross-prompt visits
    to the "same" anchor are never merged (matches the original's
    ``for idx in unique_indices`` partition). Exact-match only (int64 equality);
    similarity mode is Phase 3.
    """
    N = anchor_hash.shape[0]
    assert N % group_size == 0, f"N={N} not divisible by group_size={group_size}"
    num_blocks = N // group_size
    ids = torch.empty(N, dtype=torch.long, device=anchor_hash.device)
    offset = 0
    for b in range(num_blocks):
        lo = b * group_size
        hi = lo + group_size
        # unique returns the distinct anchors and `inv` mapping each row to its
        # cluster index within this block. Equal anchor_hash -> equal inv.
        _, inv = torch.unique(anchor_hash[lo:hi], return_inverse=True)
        ids[lo:hi] = offset + inv
        offset += int(inv.max().item()) + 1
    return ids


def step_advantage(
    step_returns: torch.Tensor,
    anchor_hash: torch.Tensor,
    group_size: int,
    remove_std: bool,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Step-level advantage ``A^S(a_t) = (R_t − mean_{anchor cluster}) [/ std]``.

    The step group is the anchor cluster (Eq. 6) scoped within each prompt
    block. Size-1 clusters (unique anchor, e.g. padding rows) -> ``A^S = 0``.
    """
    step_group_id = build_step_group(anchor_hash, group_size)
    return _group_norm(step_returns, step_group_id, remove_std, epsilon)


# --------------------------------------------------------------------------- #
# Eq. 8: joint advantage                                                      #
# --------------------------------------------------------------------------- #
def compute_gigpo_per_row_advantage(
    rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    anchor_hash: torch.Tensor,
    group_size: int,
    max_steps: int,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    gamma: float = 0.95,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """GiGPO per-row advantage ``A = A^E + ω · A^S`` (Eq. 8).

    Args:
        rewards: ``[N]`` episode outcome ``R(τ_i)`` per row (raw, pre-scaling;
            every step-row of a trajectory carries the same value).
        step_rewards: ``[N]`` per-step env reward ``r_k``.
        anchor_hash: ``[N]`` int64 anchor id per row (non-negative for real
            steps; padding rows use a unique negative sentinel so they form
            size-1 clusters).
        group_size: ``n_samples * max_steps`` (rows per contiguous prompt block).
        max_steps: env steps per trajectory.
        step_advantage_w: ``ω`` (default 1.0; all paper experiments use 1.0).
        mode: ``"mean_std_norm"`` (GRPO-style, with std) or ``"mean_norm"``
            (mean only). alfworld shipped config uses ``mean_std_norm``.
        gamma: Eq. 5 discount (alfworld shipped config uses 0.95).

    Returns:
        ``[N]`` per-row advantage. The caller writes it into ``data["rewards"]``
        and lets AReaL's GAE path (discount=gae_lambda=1) broadcast it to every
        response token of the row.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode!r}; expected mean_std_norm|mean_norm")

    # Eq. 5: discounted return-to-go per trajectory.
    step_returns = compute_step_returns(step_rewards, group_size, max_steps, gamma)
    # Eq. 3: episode advantage (group = prompt block).
    a_episode = episode_advantage(rewards, group_size, remove_std, epsilon)
    # Eq. 6: anchor clustering within each block (computed once; reused for Eq.7).
    step_group_id = build_step_group(anchor_hash, group_size)
    # Eq. 7: step advantage (group = anchor cluster within block).
    a_step = _group_norm(step_returns, step_group_id, remove_std, epsilon)

    # Diagnostic: mean step-group size. >1.0 means anchors repeat across the
    # prompt's trajectories -> A^S is active. ==1.0 means no repeats -> GiGPO
    # degenerated to GRPO (check that the n_samples rollouts share a game/task,
    # else cross-trajectory anchor clustering finds nothing). Mirrors
    # core_gigpo's "Avg size of step-level group" log.
    num_clusters = int(step_group_id.max().item()) + 1
    mean_size = step_group_id.numel() / max(num_clusters, 1)
    _logger.info(
        "[GiGPO] step groups: %d clusters, mean size %.2f over %d rows "
        "(>1.0 => A^S active; ==1.0 => degenerated to GRPO -- verify same-game "
        "grouping across the n_samples rollouts)",
        num_clusters, mean_size, step_group_id.numel(),
    )

    # Eq. 8: joint advantage.
    return a_episode + step_advantage_w * a_step
