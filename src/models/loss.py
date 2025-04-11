import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import TRAIN_CONFIG, LossType

class WeightedMSELoss(nn.Module):
    """加权均方误差损失函数"""
    def __init__(self, weights=None):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights if weights is not None else TRAIN_CONFIG['LOSS_WEIGHTS']
    
    def forward(self, pred, target):
        theta_loss = F.mse_loss(pred[:, 0], target[:, 0])
        scale_loss = F.mse_loss(pred[:, 1], target[:, 1])
        dx_loss = F.mse_loss(pred[:, 2], target[:, 2])
        dy_loss = F.mse_loss(pred[:, 3], target[:, 3])
        
        # 应用权重
        weighted_loss = (
            self.weights['theta'] * theta_loss +
            self.weights['s'] * scale_loss +
            self.weights['dx'] * dx_loss +
            self.weights['dy'] * dy_loss
        )
        
        return weighted_loss

class WeightedMAELoss(nn.Module):
    """加权平均绝对误差损失函数"""
    def __init__(self, weights=None):
        super(WeightedMAELoss, self).__init__()
        self.weights = weights if weights is not None else TRAIN_CONFIG['LOSS_WEIGHTS']
    
    def forward(self, pred, target):
        theta_loss = F.l1_loss(pred[:, 0], target[:, 0])
        scale_loss = F.l1_loss(pred[:, 1], target[:, 1])
        dx_loss = F.l1_loss(pred[:, 2], target[:, 2])
        dy_loss = F.l1_loss(pred[:, 3], target[:, 3])
        
        weighted_loss = (
            self.weights['theta'] * theta_loss +
            self.weights['s'] * scale_loss +
            self.weights['dx'] * dx_loss +
            self.weights['dy'] * dy_loss
        )
        
        return weighted_loss

class WeightedSmoothL1Loss(nn.Module):
    """加权平滑L1损失函数"""
    def __init__(self, weights=None, beta=1.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self.weights = weights if weights is not None else TRAIN_CONFIG['LOSS_WEIGHTS']
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, pred, target):
        theta_loss = self.smooth_l1(pred[:, 0], target[:, 0])
        scale_loss = self.smooth_l1(pred[:, 1], target[:, 1])
        dx_loss = self.smooth_l1(pred[:, 2], target[:, 2])
        dy_loss = self.smooth_l1(pred[:, 3], target[:, 3])
        
        weighted_loss = (
            self.weights['theta'] * theta_loss +
            self.weights['s'] * scale_loss +
            self.weights['dx'] * dx_loss +
            self.weights['dy'] * dy_loss
        )
        
        return weighted_loss

def get_loss_function(loss_type=None):
    """获取指定类型的损失函数
    
    Args:
        loss_type (LossType, optional): 损失函数类型，默认为配置文件中指定的类型
        
    Returns:
        nn.Module: 损失函数对象
    """
    if loss_type is None:
        loss_type = TRAIN_CONFIG['LOSS_TYPE']
    
    if isinstance(loss_type, str):
        try:
            loss_type = LossType(loss_type)
        except ValueError:
            raise ValueError(f"未知的损失函数类型: {loss_type}")
    
    if loss_type == LossType.MSE:
        return nn.MSELoss()
    elif loss_type == LossType.MAE:
        return nn.L1Loss()
    elif loss_type == LossType.SMOOTH_L1:
        return nn.SmoothL1Loss()
    elif loss_type == LossType.WEIGHTED_MSE:
        return WeightedMSELoss()
    else:
        raise ValueError(f"未支持的损失函数类型: {loss_type}")

def normalize_params(params, denormalize=False):
    """归一化/反归一化变换参数
    
    Args:
        params (torch.Tensor): 形状为[batch_size, 4]的张量，表示四个变换参数
        denormalize (bool): 如果为True，执行反归一化；如果为False，执行归一化
        
    Returns:
        torch.Tensor: 归一化/反归一化后的参数
    """
    theta_range = TRAIN_CONFIG['THETA_RANGE']
    scale_range = TRAIN_CONFIG['SCALE_RANGE']
    shift_range = TRAIN_CONFIG['SHIFT_RANGE']
    
    result = params.clone()
    
    if denormalize:
        # 从[-1, 1]范围反归一化到实际范围
        result[:, 0] = result[:, 0] * (theta_range[1] - theta_range[0]) / 2 + (theta_range[0] + theta_range[1]) / 2  # theta
        result[:, 1] = result[:, 1] * (scale_range[1] - scale_range[0]) / 2 + (scale_range[0] + scale_range[1]) / 2  # scale
        result[:, 2] = result[:, 2] * (shift_range[1] - shift_range[0]) / 2 + (shift_range[0] + shift_range[1]) / 2  # dx
        result[:, 3] = result[:, 3] * (shift_range[1] - shift_range[0]) / 2 + (shift_range[0] + shift_range[1]) / 2  # dy
    else:
        # 从实际范围归一化到[-1, 1]范围
        result[:, 0] = 2 * (result[:, 0] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1  # theta
        result[:, 1] = 2 * (result[:, 1] - scale_range[0]) / (scale_range[1] - scale_range[0]) - 1  # scale
        result[:, 2] = 2 * (result[:, 2] - shift_range[0]) / (shift_range[1] - shift_range[0]) - 1  # dx
        result[:, 3] = 2 * (result[:, 3] - shift_range[0]) / (shift_range[1] - shift_range[0]) - 1  # dy
    
    return result 