import numpy as np, torch, torchvision.transforms as T
from PIL import Image, ImageFilter
from .utils import device
def add_gaussian_noise(img, std=0.1):
    arr = np.array(img)/255.0; noise = np.random.normal(0, std, arr.shape); arr = np.clip(arr+noise,0,1)
    return Image.fromarray((arr*255).astype(np.uint8))
def add_motion_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=2))
def add_random_occlusion(img, size=8):
    arr = np.array(img); H,W,_ = arr.shape; x=np.random.randint(0,W-size); y=np.random.randint(0,H-size); arr[y:y+size, x:x+size]=0; return Image.fromarray(arr)
def eval_with_corruptions(model, test_set, batch_size=128):
    import torchvision.transforms as T
    corruptions = {"clean": lambda x:x, "gaussian": add_gaussian_noise, "motion_blur": add_motion_blur, "occlusion": add_random_occlusion}
    normalize = T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)); to_tensor = T.ToTensor()
    results = {}
    for cname, cfunc in corruptions.items():
        data = [(normalize(to_tensor(cfunc(img))), label) for img,label in test_set]
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        correct=0; total=0; model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs); _,pred = outputs.max(1); total += targets.size(0); correct += pred.eq(targets).sum().item()
        results[cname] = 100.0*correct/total
    return results
