import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# 配置参数
num_classes = 101
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# 1. 数据预处理和加载
# 定义数据增强和预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 加载整个数据集（所有图像在一个文件夹data下）
data_dir = "E:\engineering-practice-and-innovation-project-ii\Final\data"
full_dataset = ImageFolder(root=data_dir, transform=transform)

# 设置训练集和验证集比例
train_size = int(0.8 * len(full_dataset))  # 80% 用于训练
val_size = len(full_dataset) - train_size  # 剩下 20% 用于验证

# 随机划分数据集
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

if __name__ == "__main__":
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=400,  # 可调整为 128 等更大的值
        shuffle=True,
        num_workers=20,  # 可尝试 8 到 12
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=400,
        shuffle=False,
        num_workers=20,
        pin_memory=True,
    )

    # 检查划分结果
    print(f"训练集大小：{len(train_loader.dataset)}")
    print(f"验证集大小：{len(val_loader.dataset)}")

    # 2. 加载 ResNet 模型并微调
    # 使用预训练的 ResNet-18
    model = models.resnet18(pretrained=True)

    # 修改全连接层，适应新的分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 将模型移动到 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("使用 GPU 加速")
    else:
        device = torch.device("cpu")
        print("GPU 不可用，使用 CPU")

    model = model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练模型

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 训练模式
                dataloader = train_loader
            else:
                model.eval()  # 验证模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清空梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播和优化
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        print("训练完成！")

        # 5. 导出模型为 ONNX 格式
        dummy_input = torch.randn(
            1, 3, 224, 224, device=device
        )  # 与模型输入相匹配的张量
        onnx_file_path = "resnet18.onnx"
        torch.onnx.export(
            model,  # 要导出的模型
            dummy_input,  # 示例输入张量
            onnx_file_path,  # 输出的 ONNX 文件名
            export_params=True,  # 导出所有参数
            opset_version=11,  # ONNX 操作集版本
            do_constant_folding=True,  # 启用常量折叠优化
            input_names=["input"],  # 输入张量的名称
            output_names=["output"],  # 输出张量的名称
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },  # 动态批量大小
        )

        print(f"模型已成功导出为 {onnx_file_path}")
