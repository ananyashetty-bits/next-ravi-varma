import argparse
import subprocess
import sys

def run_batch(args):
    """Routes to your run_batch.py script"""
    cmd = ["python", "pipeline/run_batch.py", "--model", args.model, "--out", args.out]
    cmd.extend(["--seeds"] + args.seeds)
    
    if args.raw:
        cmd.extend(["--raw", args.raw])
        if args.condition:
            cmd.extend(["--condition", args.condition])
    elif args.expansions:
        cmd.extend(["--expansions", args.expansions])
        
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

    args = parser.parse_args()

    if args.command == "batch":
        run_batch(args)

if __name__ == "__main__":
    main()