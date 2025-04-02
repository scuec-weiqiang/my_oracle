import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import mnist_reader as mnist_reader
from model import OracleMNISTDataset,OracleCNN

# 训练函数
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


# 评估函数
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


# 主程序
def main():
    # 加载数据
    x_train, y_train = mnist_reader.load_data('data/oracle', kind='train')
    x_test, y_test = mnist_reader.load_data('data/oracle', kind='t10k')
    
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

    epochs=10

    # 训练与评估
    print("Starting training...")
    train_losses, train_accs = train_model(
        model, train_loader, criterion, optimizer, epochs=epochs
    )
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color='blue')
    plt.axhline(y=test_loss, color='red', linestyle='dashed', label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # 画 accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label="Train Acc", color='blue')
    plt.axhline(y=test_acc, color='red', linestyle='dashed', label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curve")

    # 显示窗口
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), 'oracle_cnn.pth')
    print("Model saved to oracle_cnn.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()