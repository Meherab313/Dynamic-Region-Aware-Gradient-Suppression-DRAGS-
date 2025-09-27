import argparse, torchvision
from drags.models import build_resnet18
from drags.viz import drags_heatmap_pdf
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--num-images", type=int, default=5); args=p.parse_args()
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    model = build_resnet18(num_classes=10, use_drags=True, drags_layers=("layer1","layer2"))
    drags_heatmap_pdf(model, dataset, num_images=args.num_images)
