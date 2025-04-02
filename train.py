import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gzip
from torchvision import transforms
import src.mnist_reader as mnist_reader

# 1. 手动实现 mnist_reader 功能
def load_mnist(path, kind='train'):
    """加载MNIST格式数据（兼容Oracle-MNIST）"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    
    return images, labels

# 2. 自定义Dataset类（修正版）
class OracleMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 将numpy数组直接转换为PIL图像
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        else:
            # 如果没有transform，直接转换为tensor并归一化
            image = transforms.functional.to_tensor(image)
            image = transforms.functional.normalize(image, [0.5], [0.5])
        
        return image, label

# 3. 定义CNN模型
class OracleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(OracleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 4. 训练与评估函数
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses, train_accs = [], []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    return train_losses, train_accs

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    print(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    return test_loss, test_acc

# 5. 主程序
def main():
    # 加载数据
    try:
        import mnist_reader
        x_train, y_train = mnist_reader.load_data('data/oracle', kind='train')
        x_test, y_test = mnist_reader.load_data('data/oracle', kind='t10k')
    except ImportError:
        print("mnist_reader not found, using local loader")
        x_train, y_train = load_mnist('data/oracle', kind='train')
        x_test, y_test = load_mnist('data/oracle', kind='t10k')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 创建Dataset和DataLoader
    train_dataset = OracleMNISTDataset(x_train, y_train, transform=transform)
    test_dataset = OracleMNISTDataset(x_test, y_test, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    model = OracleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练与评估
    print("Starting training...")
    train_losses, train_accs = train_model(
        model, train_loader, criterion, optimizer, epochs=10
    )
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    
    # 保存模型
    torch.save(model.state_dict(), 'oracle_cnn.pth')
    print("Model saved to oracle_cnn.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()