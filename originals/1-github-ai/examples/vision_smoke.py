"""Run real VLM inference on GitHub: GH_AI_TOKEN=... python examples/vision_smoke.py.

Supply a matched GGUF repository, weight file and projector from its publisher.
A nonempty response checks execution only; inspect the answer for visual accuracy.
"""
import argparse
from gh_inference import Agent, add_vision_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("filename")
    parser.add_argument("projector")
    parser.add_argument("image", help="Public HTTPS image URL")
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--chat-format", choices=["moondream2", "llava-1-5"], default="moondream2")
    args = parser.parse_args()
    model = add_vision_model(args.repository, filename=args.filename,
                             projector_filename=args.projector, revision=args.revision,
                             chat_format=args.chat_format)
    agent = Agent(model=model, remember=False, n_ctx=args.n_ctx, max_tokens=128, temperature=0, track_time=True)
    answer = agent.run("Describe the objects and their colors in this image.", images=[args.image])
    if not answer.strip():
        raise RuntimeError("Vision model returned an empty answer")
    print(answer)
    print(agent.last_job.actions_url)
    print(agent.last_result.usage)


if __name__ == "__main__":
    main()
