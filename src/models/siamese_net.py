import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import MODEL_CONFIG, DATA_CONFIG

class FeatureExtractor(nn.Module):
    """特征提取器，CNN结构，用于孪生网络的两个分支"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # 获取配置的通道数
        channels = MODEL_CONFIG['CONV_CHANNELS']
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.conv1_2 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channels[1])
        self.conv2_2 = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(channels[2])
        self.conv3_2 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(channels[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(channels[3])
        self.conv4_2 = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(channels[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        return x

class RegressionHead(nn.Module):
    """回归头，将特征映射到四个变换参数"""
    def __init__(self, input_features):
        super(RegressionHead, self).__init__()
        
        fc_dims = MODEL_CONFIG['FC_DIMS']
        dropout_rate = MODEL_CONFIG['DROPOUT_RATE']
        
        # 融合特征处理（降维）
        self.conv5_1 = nn.Conv2d(input_features, fc_dims[0] * 2, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(fc_dims[0] * 2)
        self.conv5_2 = nn.Conv2d(fc_dims[0] * 2, fc_dims[0], kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(fc_dims[0])
        
        # 全连接层序列
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(fc_dims[0], fc_dims[1])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_dims[1], fc_dims[2])
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(fc_dims[2], fc_dims[3])  # 输出4个参数
    
    def forward(self, x):
        # 融合特征处理
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        
        # 全局平均池化
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # 线性输出，不使用激活函数
        
        return x

class SiameseNet(nn.Module):
    """孪生网络，用于图像配准参数估计"""
    def __init__(self):
        super(SiameseNet, self).__init__()
        
        # 特征提取器（权重共享）
        self.feature_extractor = FeatureExtractor()
        
        # 融合后的特征维度 = 两个分支的通道数相加
        feature_dim = MODEL_CONFIG['CONV_CHANNELS'][3] * 2
        
        # 回归头
        self.regression_head = RegressionHead(feature_dim)
    
    def forward(self, ir_img, vis_img):
        # 使用同一个特征提取器处理两张图片
        ir_features = self.feature_extractor(ir_img)
        vis_features = self.feature_extractor(vis_img)
        
        # 特征融合（沿着通道维度拼接）
        fused_features = torch.cat([ir_features, vis_features], dim=1)
        
        # 回归头处理
        transform_params = self.regression_head(fused_features)
        
        # 输出四个变换参数：旋转角度θ、缩放因子s、平移向量(Δx, Δy)
        return transform_params
        
    def get_parameter_count(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 