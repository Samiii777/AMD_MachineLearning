import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = self._make_stage(64, 64, 3)
        self.stage3 = self._make_stage(64, 128, 4, stride=2)
        self.stage4 = self._make_stage(128, 256, 6, stride=2)
        self.stage5 = self._make_stage(256, 512, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(Residual(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Residual(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avg_pool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

# Set up the data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Set up multi-GPU training
num_gpus = torch.cuda.device_count()
batch_size = 256 * num_gpus

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2 * num_gpus)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2 * num_gpus)

# Initialize the model, loss function, and optimizer
model = ResNet50()

if num_gpus > 1:
    print(f"Using {num_gpus} GPUs!")
    model = nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

initial_lr = 0.1
lr = initial_lr * num_gpus
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Train the model
for epoch in range(37):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Print progress every 10 batches
        if (i + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, loss = {running_loss / (i+1):.3f}')
    
    # Print epoch loss
    print(f'Epoch {epoch+1}, loss = {running_loss / (i+1):.3f}')

    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testset)
    print(f'Test loss: {test_loss/len(testloader):.3f}, accuracy: {accuracy*100:.2f}%')

# Save the model
if num_gpus > 1:
    torch.save(model.module.state_dict(), 'resnet50_cifar10.pth')
else:
    torch.save(model.state_dict(), 'resnet50_cifar10.pth')
print('Model saved as resnet50_cifar10.pth')
