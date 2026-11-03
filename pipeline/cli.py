import argparse
import subprocess
import sys
import time

def run_batch(args):
    cmd = [sys.executable, "pipeline/run_batch.py", "--model", args.model, "--out", args.out]
    cmd.extend(["--seeds"] + [str(s) for s in args.seeds])
    if getattr(args, 'raw', None):
        cmd.extend(["--raw", args.raw])
        if getattr(args, 'condition', None):
            cmd.extend(["--condition", args.condition])
    elif getattr(args, 'expansions', None):
        cmd.extend(["--expansions", args.expansions])
    subprocess.run(cmd, check=True)

def run_expand(args):
    cmd = [sys.executable, "pipeline/expand.py", "--input", args.input, "--out", args.out]
    if getattr(args, 'type', None):
        cmd.extend(["--type", args.type])
    subprocess.run(cmd, check=True)

def run_evaluate(args):
    cmd = [sys.executable, "pipeline/eval.py", "--manifest", args.manifest, "--out", args.out]
    subprocess.run(cmd, check=True)

def run_generate_single(args):
    """Executes single text_to_image pipeline via CLI"""
    from pipeline.generate_api import text_to_image
    from common.tokens import n_tokens
    
    print(f"Running text_to_image for prompt: '{args.prompt}' with seed {args.seed}")
    start_time = time.time()
    
    image, exp, scene = text_to_image(args.prompt, seed=args.seed, mode=args.mode, model=args.model)
    
    elapsed = time.time() - start_time
    print(f"SceneSpec: {scene}")
    print(f"Expanded Prompt: {exp['positive']}")
    print(f"Token Count: {n_tokens(exp['positive'])}")
    print(f"Generation completed in {elapsed:.2f} seconds.")
    
    image.save(args.out)
    print(f"Saved generated image to {args.out}")

def main():
    parser = argparse.ArgumentParser(description="Ravi Varma M3 Pipeline Master CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for Single Generation (Direct text-to-image)
    gen_parser = subparsers.add_parser("generate", help="Run single text-to-image generation")
    gen_parser.add_argument("prompt", help="Text description of the scene")
    gen_parser.add_argument("--seed", type=int, default=1, help="Random seed")
    gen_parser.add_argument("--mode", default="llm_kb", help="Expansion mode")
    gen_parser.add_argument("--model", default="rrv_v1", help="Model key")
    gen_parser.add_argument("--out", required=True, help="Output image path (.png)")

    # Subparser for Batch Generation
    batch_parser = subparsers.add_parser("batch", help="Run batch generation jobs")
    batch_parser.add_argument("--raw", help="Path to raw scenes JSON")
    batch_parser.add_argument("--expansions", help="Path to expansions JSON")
    batch_parser.add_argument("--condition", help="Condition name (for raw)")
    batch_parser.add_argument("--model", required=True, help="Model key")
    batch_parser.add_argument("--seeds", nargs='+', required=True, help="List of seeds")
    batch_parser.add_argument("--out", required=True, help="Output directory")

    # Subparser for Prompt Expansion
    expand_parser = subparsers.add_parser("expand", help="Run LLM prompt expansions")
    expand_parser.add_argument("--input", required=True, help="Input scenes JSON")
    expand_parser.add_argument("--type", help="Expansion type")
    expand_parser.add_argument("--out", required=True, help="Output JSON path")

    # Subparser for Evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Run CLIP/ImageHash evaluation")
    eval_parser.add_argument("--manifest", required=True, help="Path to generation manifest.json")
    eval_parser.add_argument("--out", required=True, help="Output metrics JSON path")

    args = parser.parse_args()

    if args.command == "generate":
        run_generate_single(args)
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "expand":
        run_expand(args)
    elif args.command == "evaluate":
        run_evaluate(args)

if __name__ == "__main__":
    main()