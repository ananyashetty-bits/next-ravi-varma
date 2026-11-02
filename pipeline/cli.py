import argparse
import subprocess
import sys

def run_batch(args):
    """Routes to your run_batch.py script"""
    cmd = ["python", "pipeline/run_batch.py", "--model", args.model, "--out", args.out]
    cmd.extend(["--seeds"] + args.seeds)
    
    if getattr(args, 'raw', None):
        cmd.extend(["--raw", args.raw])
        if getattr(args, 'condition', None):
            cmd.extend(["--condition", args.condition])
    elif getattr(args, 'expansions', None):
        cmd.extend(["--expansions", args.expansions])
        
    subprocess.run(cmd, check=True)

def run_expand(args):
    """Routes to the LLM expansion script (to be built)"""
    cmd = ["python", "pipeline/expand.py", "--input", args.input, "--out", args.out]
    if getattr(args, 'type', None):
        cmd.extend(["--type", args.type])
    subprocess.run(cmd, check=True)

def run_evaluate(args):
    """Routes to the metric evaluation script (to be built)"""
    cmd = ["python", "pipeline/eval.py", "--manifest", args.manifest, "--out", args.out]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Ravi Varma M3 Pipeline Master CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for Batch Generation
    batch_parser = subparsers.add_parser("batch", help="Run batch generation jobs")
    batch_parser.add_argument("--raw", help="Path to raw scenes JSON")
    batch_parser.add_argument("--expansions", help="Path to expansions JSON")
    batch_parser.add_argument("--condition", help="Condition name (for raw)")
    batch_parser.add_argument("--model", required=True, help="Model key (e.g., base, rrv_v1)")
    batch_parser.add_argument("--seeds", nargs='+', required=True, help="List of seeds")
    batch_parser.add_argument("--out", required=True, help="Output directory")

    # Subparser for Prompt Expansion
    expand_parser = subparsers.add_parser("expand", help="Run LLM prompt expansions")
    expand_parser.add_argument("--input", required=True, help="Input scenes JSON")
    expand_parser.add_argument("--type", help="Expansion type (e.g., llm_nokb)")
    expand_parser.add_argument("--out", required=True, help="Output JSON path")

    # Subparser for Evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Run CLIP/ImageHash evaluation")
    eval_parser.add_argument("--manifest", required=True, help="Path to generation manifest.json")
    eval_parser.add_argument("--out", required=True, help="Output metrics JSON path")

    args = parser.parse_args()

    if args.command == "batch":
        run_batch(args)
    elif args.command == "expand":
        run_expand(args)
    elif args.command == "evaluate":
        run_evaluate(args)

if __name__ == "__main__":
    main()