"""
instruction following GRPO training example.

This script demonstrates how to train a model using GRPO on instruction following datasets,
where the reward is computed based on whether the model follows specified
instruction constraints.
"""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    """
    Main training function.

    Args:
        args: Command line arguments containing:
            - tokenizer_path: Tokenizer path
            - model_path: Model path
            - train_dataset: Training dataset config (path, split, etc.)
            - valid_dataset: Validation dataset config (optional)
            - gconfig: Generation hyperparameters
            - Other GRPO training parameters
    """
    # Load config from arguments
    config, _ = load_expr_config(args, GRPOConfig)

    # Load tokenizer
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load training dataset
    train_dataset = get_custom_dataset(
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    # Load validation dataset if configured
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    # Workflow configuration
    workflow_kwargs = dict(
        reward_fn="areal.reward.instruction_following.if_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=config.gconfig.enable_thinking,
    )

    # Evaluation workflow configuration (with lower temperature)
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    # Create trainer and run
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
