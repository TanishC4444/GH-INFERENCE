"""The command line client never loads model weights on your computer."""
import argparse
import json

from . import Agent, MODELS, list_quantizations, search_models


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
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--track-time", action="store_true")
    run.add_argument("--experimental", action="store_true")
    args = parser.parse_args()
    if args.command == "models":
        print(json.dumps(search_models(args.query) if args.query is not None else list(MODELS), indent=2))
    elif args.command == "variants":
        print(json.dumps(list_quantizations(args.model, args.revision), indent=2))
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
