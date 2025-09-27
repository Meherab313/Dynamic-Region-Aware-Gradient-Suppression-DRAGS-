from torchvision.models import resnet18
from .layers import DRAGSLayerWrapper
from .utils import device
def build_resnet18(num_classes=10, use_drags=False, drags_layers=("layer1","layer2"), pool_ks=3, topk_ratio=0.1):
    model = resnet18(weights=None, num_classes=num_classes)
    if use_drags:
        for name in drags_layers:
            module = getattr(model, name)
            setattr(model, name, DRAGSLayerWrapper(module, pool_ks=pool_ks, topk_ratio=topk_ratio))
    return model.to(device)
