"""把 agent_harness 的 yaml 场景转成 aReaL dataloader 可用的 jsonl.

用法:
    python gen_data.py --scenarios d:/Code/GitHub/vehicle_control/agent_harness/eval/scenarios --output /tmp/areal/vc_data/train.jsonl
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", default="d:/Code/GitHub/vehicle_control/agent_harness/eval/scenarios",
        help="yaml 场景目录路径",
    )
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    args = parser.parse_args()

    sys.path.insert(0, "d:/Code/GitHub/vehicle_control")
    from agent_harness.eval.scenario import load_scenarios

    scenarios = load_scenarios(args.scenarios)
    with open(args.output, "w", encoding="utf-8") as f:
        for sc in scenarios:
            data = sc.to_workflow_data()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"Generated {len(scenarios)} scenarios -> {args.output}")


if __name__ == "__main__":
    main()
