import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

def setup_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def create_fusion_visualization(ir_img: np.ndarray, 
                              aligned_ir_img: np.ndarray,
                              vis_img: np.ndarray,
                              fused_img: np.ndarray,
                              transform_params: Dict[str, float],
                              save_path: str,
                              title: Optional[str] = None) -> None:
    """创建融合可视化图像"""
    setup_matplotlib_chinese()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    axes[0, 0].imshow(ir_img, cmap='gray')
    axes[0, 0].set_title('原始红外图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(aligned_ir_img, cmap='gray')
    axes[0, 1].set_title('对齐后的红外图像')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(vis_img, cmap='gray')
    axes[1, 0].set_title('可见光图像')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fused_img)
    axes[1, 1].set_title('融合图像')
    axes[1, 1].axis('off')
    
    param_text = f"旋转: {transform_params['theta']:.2f}°, 缩放: {transform_params['scale']:.2f}, " \
                 f"水平偏移: {transform_params['dx']*100:.2f}%, 垂直偏移: {transform_params['dy']*100:.2f}%"
    
    plt.figtext(0.5, 0.01, param_text, ha='center', fontsize=12)
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def create_comparison_visualization(pred_params: Dict[str, float],
                                  true_params: Dict[str, float],
                                  ir_img: np.ndarray,
                                  true_aligned_ir: np.ndarray,
                                  pred_aligned_ir: np.ndarray,
                                  vis_img: np.ndarray,
                                  save_path: str) -> None:
    """创建预测和真实参数对比的可视化图像"""
    setup_matplotlib_chinese()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    axes[0, 0].imshow(ir_img, cmap='gray')
    axes[0, 0].set_title('原始红外图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vis_img, cmap='gray')
    axes[0, 1].set_title('可见光图像')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(true_aligned_ir, cmap='gray')
    axes[1, 0].set_title('真实参数对齐的红外图像')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_aligned_ir, cmap='gray')
    axes[1, 1].set_title('预测参数对齐的红外图像')
    axes[1, 1].axis('off')
    
    true_text = f"真实参数 - 旋转: {true_params['theta']:.2f}°, 缩放: {true_params['scale']:.2f}, " \
               f"水平偏移: {true_params['dx']*100:.2f}%, 垂直偏移: {true_params['dy']*100:.2f}%"
    
    pred_text = f"预测参数 - 旋转: {pred_params['theta']:.2f}°, 缩放: {pred_params['scale']:.2f}, " \
               f"水平偏移: {pred_params['dx']*100:.2f}%, 垂直偏移: {pred_params['dy']*100:.2f}%"
    
    plt.figtext(0.5, 0.05, true_text, ha='center', fontsize=12)
    plt.figtext(0.5, 0.02, pred_text, ha='center', fontsize=12)
    
    plt.suptitle('预测参数与真实参数对比', fontsize=16)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def save_side_by_side_images(images: Dict[str, np.ndarray], 
                           save_path: str, 
                           titles: Optional[Dict[str, str]] = None) -> None:
    """保存多张图像并排展示"""
    setup_matplotlib_chinese()
    
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    i = 0
    for key, img in images.items():
        row, col = i // cols, i % cols
        
        if len(img.shape) == 2 or img.shape[2] == 1:
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(img)
        
        if titles and key in titles:
            axes[row, col].set_title(titles[key])
        else:
            axes[row, col].set_title(key)
        
        axes[row, col].axis('off')
        i += 1
    
    for j in range(i, rows * cols):
        row, col = j // cols, j % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig) 