"""Unit tests for examples/skillrl/gigpo_advantage.py.

Run:
    python -m examples.skillrl.test_gigpo_advantage
    # or
    pytest examples/skillrl/test_gigpo_advantage.py

Validates the faithful port of gigpo/core_gigpo.py (Eq. 3/5/6/7/8) against
hand-computed / property-based expectations on tiny positional batches
(n_samples=2, max_steps=3, 1-2 prompt blocks). The port adapts only the grouping
keys (positional blocks + int64 anchor equality); the math is identical.
"""

from __future__ import annotations

import math

import pytest
import torch

from examples.skillrl.gigpo_advantage import (
    build_step_group,
    compute_gigpo_per_row_advantage,
    compute_step_returns,
    episode_advantage,
    step_advantage,
)

# Layout constants for the tiny test batch.
N_SAMPLES = 2
MAX_STEPS = 3
GROUP_SIZE = N_SAMPLES * MAX_STEPS  # 6 rows per prompt block


def test_step_returns_discounted_to_go():
    """Eq. 5: R_t = sum_{k>=t} gamma^{k-t} r_k, per trajectory (block=1 prompt)."""
    # One block: traj0 wins at the last step (r=[0,0,10]); traj1 loses (r=[0,0,0]).
    step_rewards = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    gamma = 0.95
    returns = compute_step_returns(step_rewards, GROUP_SIZE, MAX_STEPS, gamma)
    # traj0: R_2=10, R_1=0.95*10=9.5, R_0=0.95*9.5=9.025 ; traj1: all 0.
    expected = torch.tensor([9.025, 9.5, 10.0, 0.0, 0.0, 0.0])
    assert torch.allclose(returns, expected, atol=1e-6)


def test_episode_advantage_mean_std():
    """Eq. 3 (mean_std_norm): (R - mean_block) / std_block over the prompt block.

    rewards: traj0 R=10 (won), traj1 R=0 (lost), each repeated max_steps=3 rows.
    """
    rewards = torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    a = episode_advantage(rewards, GROUP_SIZE, remove_std=False)
    mean = rewards.mean()
    std = rewards.std(unbiased=True)  # unbiased, matches core_gigpo torch.std
    expected = (rewards - mean) / (std + 1e-6)
    assert torch.allclose(a, expected, atol=1e-6)
    # Symmetric around 0 (one winner, one loser of equal magnitude).
    assert torch.allclose(a[:3], -a[3:], atol=1e-6)


def test_episode_advantage_mean_norm():
    """Eq. 3 (mean_norm): R - mean_block (no std division)."""
    rewards = torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    a = episode_advantage(rewards, GROUP_SIZE, remove_std=True)
    mean = rewards.mean()
    expected = rewards - mean
    assert torch.allclose(a, expected, atol=1e-6)


def test_step_advantage_degradation_unique_anchors():
    """Eq. 7: all-unique anchors -> every cluster size 1 -> A^S = 0 (GRPO lower bound)."""
    # Distinct anchors per row -> all size-1 clusters.
    anchor_hash = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    step_returns = torch.tensor([9.025, 9.5, 10.0, 0.0, 0.0, 0.0])
    a = step_advantage(step_returns, anchor_hash, GROUP_SIZE, remove_std=False)
    assert torch.allclose(a, torch.zeros(6), atol=1e-6)


def test_step_advantage_shared_anchor_clusters():
    """Eq. 6 + 7: rows sharing an anchor within a block cluster; A^S nonzero there.

    traj0 and traj1 both visit anchor 100 at step 0 -> cluster {9.025, 0.0}.
    All other anchors unique -> A^S = 0 there.
    """
    anchor_hash = torch.tensor([100, 200, 300, 100, 400, 500], dtype=torch.long)
    step_returns = torch.tensor([9.025, 9.5, 10.0, 0.0, 0.0, 0.0])
    a = step_advantage(step_returns, anchor_hash, GROUP_SIZE, remove_std=False)
    # Cluster {9.025, 0.0}: mean=4.5125. Unbiased std (N=2, matches core_gigpo's
    # torch.std default) = sqrt(2 * 4.5125^2 / (2-1)) = 4.5125 * sqrt(2) = 6.3808.
    mean = 4.5125
    std = 4.5125 * math.sqrt(2)  # unbiased std of 2 samples (NOT half-range = population std)
    expected_shared = torch.tensor(
        [(9.025 - mean) / (std + 1e-6), (0.0 - mean) / (std + 1e-6)]
    )
    assert torch.allclose(a[0], expected_shared[0], atol=1e-4)
    assert torch.allclose(a[3], expected_shared[1], atol=1e-4)
    # Non-shared (size-1) -> 0.
    assert torch.allclose(a[[1, 2, 4, 5]], torch.zeros(4), atol=1e-6)


def test_gigpo_combination_eq8():
    """Eq. 8: adv == A^E + ω * A^S, end-to-end."""
    rewards = torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    step_rewards = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    anchor_hash = torch.tensor([100, 200, 300, 100, 400, 500], dtype=torch.long)
    omega = 1.0
    gamma = 0.95

    adv = compute_gigpo_per_row_advantage(
        rewards, step_rewards, anchor_hash, GROUP_SIZE, MAX_STEPS,
        step_advantage_w=omega, mode="mean_std_norm", gamma=gamma,
    )
    step_returns = compute_step_returns(step_rewards, GROUP_SIZE, MAX_STEPS, gamma)
    a_e = episode_advantage(rewards, GROUP_SIZE, remove_std=False)
    a_s = step_advantage(step_returns, anchor_hash, GROUP_SIZE, remove_std=False)
    expected = a_e + omega * a_s
    assert torch.allclose(adv, expected, atol=1e-6)


def test_degradation_to_grpo():
    """When no anchor repeats, GiGPO == GRPO (A^S = 0)."""
    rewards = torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    step_rewards = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    anchor_hash = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)  # all unique
    adv = compute_gigpo_per_row_advantage(
        rewards, step_rewards, anchor_hash, GROUP_SIZE, MAX_STEPS,
        step_advantage_w=1.0, mode="mean_std_norm", gamma=0.95,
    )
    a_e = episode_advantage(rewards, GROUP_SIZE, remove_std=False)
    assert torch.allclose(adv, a_e, atol=1e-6)


def test_padding_rows_zero_step_advantage():
    """Padding rows (unique negative sentinel anchors) -> size-1 cluster -> A^S=0.

    Mirrors SkillEnvWorkflow._padding_row: traj0 wins at step 0 then is padded
    (2 padding rows). Real step anchor=100 (non-negative); padding anchors are
    unique negatives -> never cluster with real or each other.
    """
    # traj0: real step0 (r=10, anchor=100) + 2 padding (r=0, unique neg anchors).
    # traj1: 3 real steps, loses (r=0).
    rewards = torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    step_rewards = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    anchor_hash = torch.tensor([100, -11, -12, 700, 800, 900], dtype=torch.long)
    step_returns = compute_step_returns(step_rewards, GROUP_SIZE, MAX_STEPS, 0.95)
    # traj0 returns: R_0=10, R_1=0, R_2=0 ; traj1: all 0.
    assert torch.allclose(step_returns, torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-6)
    a_s = step_advantage(step_returns, anchor_hash, GROUP_SIZE, remove_std=False)
    # Padding rows (idx 1, 2) and all of traj1 are size-1 -> A^S=0. Real step0
    # (idx 0, anchor 100) is also size-1 here (traj1 has no anchor 100) -> 0.
    assert torch.allclose(a_s, torch.zeros(6), atol=1e-6)


def test_padding_clustered_with_real_only_when_shared():
    """A padding row's unique sentinel must NOT cluster with a real anchor.

    traj0 step0 anchor=100; traj1 step0 anchor=100 (shared -> cluster). traj0
    step1 is padding with sentinel -11. The -11 must be its own cluster even
    though it is negative -- confirming sign-based isolation is not relied upon
    (uniqueness is). build_step_group ids reflect this.
    """
    anchor_hash = torch.tensor([100, -11, 300, 100, 400, 500], dtype=torch.long)
    ids = build_step_group(anchor_hash, GROUP_SIZE)
    # rows 0 and 3 share anchor 100 -> same id; all others distinct.
    assert ids[0].item() == ids[3].item()
    assert len({ids[i].item() for i in (1, 2, 4, 5)}) == 4
    assert ids[1].item() != ids[0].item()


def test_multi_block_independent_grouping():
    """Two prompt blocks: episode/step grouping is per-block (no cross-block merge)."""
    # Block 0: traj0 wins (R=10), traj1 loses (R=0).
    # Block 1: traj0 loses (R=0), traj1 wins (R=10).
    rewards = torch.tensor(
        [10.0, 10.0, 10.0, 0.0, 0.0, 0.0,  # block 0
         0.0, 0.0, 0.0, 10.0, 10.0, 10.0]  # block 1
    )
    step_rewards = torch.tensor(
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
    )
    # Same anchor layout in both blocks; clustering must NOT merge across blocks.
    anchor_block = [100, 200, 300, 100, 400, 500]
    anchor_hash = torch.tensor(anchor_block + anchor_block, dtype=torch.long)
    N = 12
    assert N % GROUP_SIZE == 0

    adv = compute_gigpo_per_row_advantage(
        rewards, step_rewards, anchor_hash, GROUP_SIZE, MAX_STEPS,
        step_advantage_w=1.0, mode="mean_std_norm", gamma=0.95,
    )
    # Per-block A^E: block0 winner=traj0, block1 winner=traj1. Both blocks have
    # one 10 and one 0 -> identical A^E magnitudes, mirrored by trajectory.
    a_e = episode_advantage(rewards, GROUP_SIZE, remove_std=False)
    # Block 0 traj0 (rows 0:3) should equal block 1 traj1 (rows 9:12) in A^E
    # (both are the winner of an otherwise-identical 10/0 split).
    assert torch.allclose(a_e[0:3], a_e[9:12], atol=1e-6)
    # Step groups: anchor 100 appears at rows 0,3 (block0) and 9,12-1=... rows
    # 6,9 (block1). build_step_group must give block0's {0,3} and block1's {6,9}
    # DIFFERENT cluster ids (no cross-block merge).
    ids = build_step_group(anchor_hash, GROUP_SIZE)
    assert ids[0].item() == ids[3].item()        # same block -> same cluster
    assert ids[6].item() == ids[9].item()        # same block -> same cluster
    assert ids[0].item() != ids[6].item()        # different blocks -> different clusters


def test_unknown_mode_raises():
    rewards = torch.zeros(6)
    step_rewards = torch.zeros(6)
    anchor_hash = torch.arange(6, dtype=torch.long)
    with pytest.raises(ValueError):
        compute_gigpo_per_row_advantage(
            rewards, step_rewards, anchor_hash, GROUP_SIZE, MAX_STEPS, mode="bogus"
        )


def test_n_not_divisible_raises():
    with pytest.raises(AssertionError):
        compute_step_returns(torch.zeros(7), GROUP_SIZE, MAX_STEPS, 0.95)


if __name__ == "__main__":
    # Allow `python -m examples.skillrl.test_gigpo_advantage` without pytest.
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
