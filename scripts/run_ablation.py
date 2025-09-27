import argparse
from drags.experiments import run_ablation_study
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--topk", type=float, nargs="+", default=[0.05,0.10,0.15])
    p.add_argument("--layers", type=str, nargs="+", default=["layer1","layer1+layer2"]); p.add_argument("--epochs", type=int, default=2); args=p.parse_args()
    layers_list=[tuple(s.split("+")) for s in args.layers]
    run_ablation_study(topk_list=args.topk, layers_list=layers_list, epochs=args.epochs)
