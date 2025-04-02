
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import src.mnist_reader as mnist_reader
from train import OracleMNISTDataset, OracleCNN
from torch.utils.data import DataLoader


# 1. 加载训练好的模型
def load_model(model_path, num_classes=10):
    model = OracleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 2. 设置中文字体
def set_chinese_font():
    """强制指定文泉驿微米黑字体路径"""
    try:
        font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # 您的实际路径
        chinese_font = FontProperties(fname=font_path)
        return chinese_font
    except:
        print("字体加载失败，将回退到英文显示")
        return None

# 3. 可视化测试结果（带中文标签）
def visualize_predictions(model, test_loader, device, num_samples=6):
    # 初始化字体
    chinese_font = set_chinese_font()
    
    # 甲骨文标签映射
    oracle_map = {
        0: "大", 1: "日", 2: "月", 3: "牛",
        4: "下", 5: "田", 6: "不", 7: "矢",
        8: "时", 9: "木"
    }
    
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].cpu().squeeze()
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        
        # 显示图片
        axes[i].imshow(img, cmap='gray')
        
        # 构建标题
        title = f"真实: {oracle_map[true_label]}\n预测: {oracle_map[pred_label]}"
        color = 'red' if true_label != pred_label else 'black'
        
        # 应用中文字体或回退到英文
        if chinese_font:
            axes[i].set_title(title, color=color, fontproperties=chinese_font)
        else:
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('oracle_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model("oracle_cnn.pth").to(device)
    
    # 加载测试数据
    x_test, y_test = mnist_reader.load_data('data/oracle', kind='t10k')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = OracleMNISTDataset(x_test, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)
    
    # 可视化结果
    visualize_predictions(model, test_loader, device)