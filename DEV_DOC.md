# 面部表情识别（FER）项目 - TensorFlow 移植版开发文档

## 1. 项目概述

本项目是将 [usef-kh/fer](https://github.com/usef-kh/fer) 仓库的 PyTorch 实现移植到 TensorFlow 框架的版本。原项目在 FER2013 数据集上使用 VGG 网络架构实现面部表情识别，达到了 73.28% 的单网络分类准确率（state-of-the-art）。

### 1.1 任务描述
- 输入：48x48 灰度人脸图像
- 输出：7 类情绪分类（Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral）
- 数据集：FER2013（Kaggle）

## 2. 项目结构

```
ML_FinalWork0.3/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # 数据加载与预处理工具函数
│   └── fer2013.py          # FER2013 数据集加载、TenCrop、数据管道
├── models/
│   ├── __init__.py
│   └── vgg.py              # VGG 网络模型（TensorFlow/Keras 实现）
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py       # 模型检查点保存与恢复
│   ├── hparams.py          # 超参数配置
│   ├── logger.py           # 训练日志记录与可视化
│   └── setup_network.py    # 网络初始化
├── train.py                # 训练入口脚本
├── evaluate.py             # 评估入口脚本
├── requirements.txt        # Python 依赖
└── DEV_DOC.md              # 本文档
```

## 3. 移植对照表

| 模块 | PyTorch 原版 | TensorFlow 移植版 | 说明 |
|------|-------------|------------------|------|
| 模型定义 | `torch.nn.Module` | `tf.keras.Model` | 子类化 API |
| 卷积层 | `nn.Conv2d` | `layers.Conv2D` | channels_last 格式 |
| BatchNorm | `nn.BatchNorm2d` | `layers.BatchNormalization` | training 参数控制 |
| 优化器 | `torch.optim.SGD` | `tf.keras.optimizers.SGD` | Nesterov 动量 |
| 学习率调度 | `ReduceLROnPlateau` | 手动实现 patience 逻辑 | 等效行为 |
| 混合精度 | `torch.cuda.amp` | TF 原生支持（可选） | 默认关闭 |
| 数据加载 | `DataLoader` | `tf.data.Dataset` | prefetch 优化 |
| 损失函数 | `nn.CrossEntropyLoss` | `SparseCategoricalCrossentropy` | from_logits=True |
| 检查点 | `torch.save` | `model.save_weights` + JSON | 权重+日志分离 |

## 4. 模型架构

### VGG 网络（主模型）

```
输入: (batch, 40, 40, 1) — 灰度图像经 TenCrop 裁剪后

Conv2D(64, 3x3) + BN + ReLU
Conv2D(64, 3x3) + BN + ReLU
MaxPool(2x2)

Conv2D(128, 3x3) + BN + ReLU
Conv2D(128, 3x3) + BN + ReLU
MaxPool(2x2)

Conv2D(256, 3x3) + BN + ReLU
Conv2D(256, 3x3) + BN + ReLU
MaxPool(2x2)

Conv2D(512, 3x3) + BN + ReLU
Conv2D(512, 3x3) + BN + ReLU
MaxPool(2x2)

Flatten → Dense(4096) + Dropout → Dense(4096) + Dropout → Dense(7)
```

### 关键设计决策
- 使用 `tf.keras.Model` 子类化 API（而非 Sequential/Functional），保持与原 PyTorch 代码结构一致
- BatchNormalization 通过 `training` 参数区分训练/推理模式
- 输出为 logits（未经 softmax），损失函数使用 `from_logits=True`

## 5. 数据处理

### TenCrop 策略
与原版一致，对每张图像生成 10 个裁剪：
- 4 个角落裁剪 (40x40)
- 1 个中心裁剪 (40x40)
- 以上 5 个裁剪各自的水平翻转

推理时对 10 个裁剪的输出取平均作为最终预测。

### 数据增强（训练时）
- 随机水平翻转

### 归一化
- 像素值除以 255.0（与原版 mu=0, st=255 一致）

## 6. 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| optimizer | SGD | momentum=0.9, nesterov=True |
| learning_rate | 0.01 | 初始学习率 |
| weight_decay | 0.0001 | L2 正则化（TF 中通过 kernel_regularizer 实现） |
| scheduler | ReduceLROnPlateau | patience=5, factor=0.75 |
| batch_size | 64 | 训练批大小 |
| epochs | 300 | 最大训练轮数 |
| dropout | 0.1 | 全连接层 Dropout 率 |

## 7. 使用方法

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据准备
下载 FER2013 数据集并放置在以下路径：
```
datasets/fer2013/fer2013.csv
```

### 训练
```bash
python train.py network=vgg name=my_experiment
```

可选参数：`bs=128`, `lr=0.01`, `n_epochs=100`, `drop=0.2`

### 评估
```bash
python evaluate.py network=vgg name=my_experiment restore_epoch=100
```

## 8. 与原版的主要差异

1. **数据格式**：TensorFlow 使用 channels_last (NHWC)，PyTorch 使用 channels_first (NCHW)
2. **TenCrop 实现**：使用 NumPy 预计算所有裁剪，存储为 (N, 10, 40, 40, 1) 张量
3. **学习率调度**：手动实现 ReduceLROnPlateau 逻辑（监控验证集准确率）
4. **混合精度训练**：原版使用 GradScaler，TF 版本暂未启用（可通过 mixed_precision policy 开启）
5. **检查点格式**：使用 TF SavedModel 权重格式 + JSON 日志，替代 PyTorch 的 .pt 文件

## 9. 后续优化方向

- 启用 TensorFlow 混合精度训练 (`mixed_precision.set_global_policy('mixed_float16')`)
- 添加 TensorBoard 回调进行训练监控
- 实现 EfficientNet 模型的 TF 版本（使用 `tf.keras.applications`）
- 添加数据增强层（`tf.keras.layers.RandomFlip`, `RandomRotation` 等）
- 支持 TFRecord 格式以提升大规模数据加载性能
