import os
import torch
from enum import Enum

class LossType(Enum):
    MSE = 'mse'  # 均方误差
    MAE = 'mae'  # 平均绝对误差
    SMOOTH_L1 = 'smooth_l1'  # 平滑L1损失
    WEIGHTED_MSE = 'weighted_mse'  # 加权MSE损失

DATA_CONFIG = {
    'VIS_DIR': r'path/to/vis/images',  # 可见光图像目录
    'IR_DIR': r'path/to/ir/images',    # 红外图像目录
    'JSON_PATH': r'path/to/delta.json',  # 包含变换参数的JSON文件路径
    'TRAIN_PERCENTAGE': 0.7,  # 训练集占比
    'MODEL_DIR': './checkpoints',  # 模型保存目录
    'IMAGE_SIZE': (128, 128),  # 输入图像尺寸 - 可选择(128, 128)或(256, 256)等
    'NORMALIZE_VIS_MEAN': 0.5,  # 可见光图像归一化均值
    'NORMALIZE_VIS_STD': 0.5,   # 可见光图像归一化标准差
    'NORMALIZE_IR_MEAN': 0.5,   # 红外图像归一化均值
    'NORMALIZE_IR_STD': 0.5,    # 红外图像归一化标准差
}

MODEL_CONFIG = {
    'BATCH_SIZE': 32,  # 批量大小
    'SIAMESE_SHARE_WEIGHTS': True,  # 是否共享权重
    'CONV_CHANNELS': [64, 128, 256, 512],  # 卷积层通道数
    'FC_DIMS': [256, 128, 64, 4],  # 全连接层维度
    'DROPOUT_RATE': 0.5,  # Dropout比率
}

TRAIN_CONFIG = {
    'NUM_EPOCHS': 300,  # 训练轮数
    'LEARNING_RATE': 0.001,  # 学习率
    'WEIGHT_DECAY': 1e-5,  # 权重衰减
    'BATCH_SIZE': 64,
    'LOSS_TYPE': LossType.WEIGHTED_MSE,  # 损失函数类型
    'LOSS_WEIGHTS': {  # 各参数损失权重
        'theta': 1.0,  # 旋转角度损失权重
        's': 1.0,      # 缩放因子损失权重
        'dx': 1.0,     # x方向平移损失权重
        'dy': 1.0      # y方向平移损失权重
    },
    'THETA_RANGE': (-30, 30),  # 旋转角度范围(度)
    'SCALE_RANGE': (0.6, 1.4),  # 缩放因子范围
    'SHIFT_RANGE': (-0.3, 0.3),  # 平移范围(相对于图像尺寸的百分比)
    'SAVE_FREQ': 5,  # 每多少轮保存一次模型
    'MAX_MODELS': 20,  # 最多保存多少个模型(不包括best和last)
    'SAVE_BEST': True,  # 是否保存最佳模型
}

INFERENCE_CONFIG = {
    'MODEL_PATH': r'path/to/model.pth',  # 用于推理的模型路径
    'INFERENCE_DIR': './inference',  # 推理结果保存目录
    'FUSION_MODE': 'average',  # 融合模式: 'average', 'weighted'等
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RANDOM_SEED = 42 