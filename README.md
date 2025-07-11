# CNN 模型性能评测框架

本项目是一个轻量级的计算机视觉模型评测框架，旨在通过标准化的流程，测试和对比不同卷积神经网络 (CNN) 模型在图像分类任务上的性能。框架使用了 PyTorch 构建，并在一个模拟的 ImageNet 子集上进行了测试。

## 项目亮点

- **模块化设计**: 将数据预处理、模型加载、评测循环和结果可视化等功能解耦，易于扩展和维护。
- **模型对比**: 内置了对经典模型 (如 ResNet) 和自定义简单模型的性能对比，包括准确率和推理时间。
- **可复现性**: 通过固定随机种子和明确的依赖列表 (`requirements.txt`)，保证了实验结果的可复现性。
- **数据模拟**: 包含了自建虚拟数据集的脚本，解决了在没有完整数据集情况下进行原型开发和测试的问题。

## 功能列表

- [x] 使用 `torchvision` 加载预训练的 CNN 模型。
- [x] 实现了一个可自定义的简单 CNN 模型作为对照组。
- [x] 标准化的图像预处理流程 (Resize, CenterCrop, ToTensor, Normalize)。
- [x] 计算模型在测试集上的 Top-1 准确率。
- [x] 测量并比较不同模型的单张图片平均推理时间。
- [x] 自动生成虚拟图像数据集用于快速测试。

## 技术栈

- **语言**: Python
- **核心库**: PyTorch, NumPy, Matplotlib
- **工具**: Git, Jupyter Notebook (可选)

## 如何运行

1.  **克隆仓库**
    ```bash
    git clone https://github.com/wusier/CNN_Model_Benchmark
    cd CNN_Model_Benchmark
    ```

2.  **创建虚拟环境并安装依赖**
    ```bash
    # 建议使用虚拟环境
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate

    # 安装依赖
    pip install -r requirements.txt
    ```

3.  **运行评测脚本**
    ```bash
    python [你的主脚本文件名].py
    ```
    脚本将自动检查设备 (优先使用CUDA)，加载模型，运行测试，并打印出最终的准确率和推理时间对比结果。

## 未来可扩展方向

- [ ] 增加 Top-5 准确率的计算。
- [ ] 集成更多的预训练模型 (如 VGG, EfficientNet)。
- [ ] 使用 TensorBoard 进行训练过程和结果的可视化。
