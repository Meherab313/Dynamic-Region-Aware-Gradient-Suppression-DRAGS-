import time, torch, torch.nn as nn, torch.optim as optim
from .utils import device
def train_model(model, train_loader, test_loader, epochs=10, lr=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    start = time.time()
    for epoch in range(epochs):
        model.train(); running_loss=0; correct=0; total=0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets); loss.backward(); optimizer.step()
            running_loss += loss.item()*inputs.size(0); _,pred = outputs.max(1); total += targets.size(0); correct += pred.eq(targets).sum().item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/total:.4f} - Train Acc: {100.0*correct/total:.2f}%")
    return time.time()-start
