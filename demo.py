import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# 加载预训练ResNet50模型
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 修改最后一层全连接层以适应CIFAR-10的10个类别
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)


# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()

    epoch_time = time.time() - start_time
    acc = 100. * correct / total
    print(f'Epoch: {epoch} | Time: {epoch_time:.2f}s | Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {acc:.2f}%')


# 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'Test Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {acc:.2f}%')
    return acc


# 训练并测试模型
num_epochs = 50
best_acc = 0

for epoch in range(1, num_epochs + 1):
    train(epoch)
    current_acc = test()

    # 保存最佳模型
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), 'resnet50_cifar10.pth')

print(f'Best Test Accuracy: {best_acc:.2f}%')