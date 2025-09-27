import argparse
from drags.experiments import run_experiment
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=128); args=p.parse_args()
    run_experiment(epochs=args.epochs, batch_size=args.batch_size)
