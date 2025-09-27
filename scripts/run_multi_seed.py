import argparse
from drags.experiments import run_multi_seed_experiment, aggregate_results
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=128); p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44]); args=p.parse_args()
    df_raw = run_multi_seed_experiment(seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size)
    aggregate_results(df_raw)
