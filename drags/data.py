import torch, torchvision, torchvision.transforms as T
def get_cifar10(batch_size=128):
    normalize = T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
