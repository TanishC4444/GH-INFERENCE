"""The command line client never loads model weights on your computer."""
import argparse
import json

from . import Agent, Limits, MODELS, list_quantizations, search_models


def token_limit(value):
    if value == "auto":
        return None
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a positive integer or auto")
    if count < 1:
        raise argparse.ArgumentTypeError("Use a positive integer or auto")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="Search public GGUF repositories")
    models.add_argument("query", nargs="?", default=None)
    variants = commands.add_parser("variants", help="List every published GGUF variant")
    variants.add_argument("model")
    variants.add_argument("--revision", default="main")
    run = commands.add_parser("run", help="Send a prompt to a GitHub worker")
    run.add_argument("prompt")
    run.add_argument("--model", default="qwen0.5b")
    run.add_argument("--quantization")
    run.add_argument("--filename")
    run.add_argument("--revision", default="main")
    run.add_argument("--context", type=int, default=None)
    run.add_argument("--max-tokens", type=token_limit, default=512)
    run.add_argument("--track-time", action="store_true")
    run.add_argument("--experimental", action="store_true")
    stress = commands.add_parser("stress", help="Run an all-to-all remote agent benchmark")
    stress.add_argument("--models", nargs="+", default=["qwen0.5b", "llama", "qwen", "tinyllama"])
    stress.add_argument("--agents", type=int, default=20)
    stress.add_argument("--rounds", type=int, default=3)
    stress.add_argument("--max-tokens", type=token_limit, default=64)
    stress.add_argument("--estimated-worker-watts", type=float)
    stress.add_argument("--output", default="stress-results.json")
    stress.add_argument("--topology", choices=["all", "ring"], default="all")
    discuss = commands.add_parser("discuss", help="Run agent rounds on one resident model")
    discuss.add_argument("prompt")
    discuss.add_argument("--model", default="qwen0.5b")
    discuss.add_argument("--agents", type=int, default=20)
    discuss.add_argument("--rounds", type=int, default=3)
    discuss.add_argument("--max-tokens", type=token_limit, default=128)
    discuss.add_argument("--topology", choices=["all", "ring"], default="all")
    args = parser.parse_args()
    if args.command == "models":
        print(json.dumps(search_models(args.query) if args.query is not None else list(MODELS), indent=2))
    elif args.command == "variants":
        print(json.dumps(list_quantizations(args.model, args.revision), indent=2))
    elif args.command == "discuss":
        agent = Agent(model=args.model, max_tokens=args.max_tokens, track_time=True,
                      limits=Limits(max_prompts=args.agents * args.rounds))
        print(json.dumps(agent.discuss(args.prompt, agents=args.agents, rounds=args.rounds,
                                       topology=args.topology), indent=2))
    elif args.command == "stress":
        from . import stress_test
        report = stress_test(models=args.models, agents=args.agents, rounds=args.rounds,
                             max_tokens=args.max_tokens, topology=args.topology,
                             estimated_worker_watts=args.estimated_worker_watts,
                             output=args.output)
        print(json.dumps(report["summary"], indent=2))
    else:
        factory = Agent.experimental if args.experimental else Agent
        agent = factory(model=args.model, quantization=args.quantization,
                        filename=args.filename, revision=args.revision,
                        n_ctx=args.context, max_tokens=args.max_tokens, track_time=args.track_time)
        print(agent(args.prompt))
        if args.track_time:
            import sys
            from dataclasses import asdict
            print(json.dumps(asdict(agent.last_result.timings)), file=sys.stderr)


if __name__ == "__main__":
    main()
