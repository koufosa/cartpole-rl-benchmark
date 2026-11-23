"""Command-line interface for training, evaluation, plots and demo."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
from .agents import PhysicsAgent, RandomAgent
from .env import make_env, play_episode
from .evaluation import evaluate_all
from .plots import plot_evaluation, plot_training
from .training import train


def build_parser():
    parser = argparse.ArgumentParser(description="CartPole policy benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train neural policy and save model, CSV and plot")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--episodes-per-epoch", type=int, default=40)
    t.add_argument("--percentile", type=float, default=70)
    t.add_argument("--epsilon-start", type=float, default=0.9)
    t.add_argument("--epsilon-end", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output", default="outputs")
    e = sub.add_parser("evaluate", help="Evaluate baselines and saved policy")
    e.add_argument("--model", default="outputs/policy.joblib")
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=10000)
    e.add_argument("--output", default="outputs")
    d = sub.add_parser("demo", help="Render one episode (requires a display)")
    d.add_argument("--agent", choices=["random", "physics", "neural"], default="neural")
    d.add_argument("--model", default="outputs/policy.joblib")
    d.add_argument("--seed", type=int, default=123)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        _, history = train(args.output, epochs=args.epochs,
                           episodes_per_epoch=args.episodes_per_epoch,
                           percentile=args.percentile, epsilon_start=args.epsilon_start,
                           epsilon_end=args.epsilon_end, seed=args.seed)
        plot_training(history, Path(args.output) / "training_curve.png")
        print(f"Saved model and training artifacts to {args.output}/")
    elif args.command == "evaluate":
        policy = joblib.load(args.model)  # Only load model files you created/trust.
        summary, details = evaluate_all(policy, args.output,
                                        n_episodes=args.episodes, seed=args.seed)
        plot_evaluation(details, Path(args.output) / "evaluation_boxplot.png")
        for row in summary:
            print(f"{row['agent']:15s}: mean={row['mean_reward']:6.1f} "
                  f"+/- {row['std_reward']:5.1f}, >=475={row['success_rate_475']:.1%}")
    elif args.command == "demo":
        agents = {"random": RandomAgent(seed=args.seed), "physics": PhysicsAgent()}
        agent = agents.get(args.agent)
        greedy = args.agent == "neural"
        if greedy:
            agent = joblib.load(args.model)  # Only load model files you created/trust.
        env = make_env(render_mode="human")
        try:
            result = play_episode(env, agent, seed=args.seed, greedy=greedy)
            print(f"{args.agent} reward: {result.reward}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
