import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 自定义Dataset类（修正版）
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

# 定义CNN模型
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