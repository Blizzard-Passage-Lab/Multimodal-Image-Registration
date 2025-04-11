import cv2
import numpy as np
from typing import Tuple, Optional

def fuse_images(ir_img: np.ndarray, vis_img: np.ndarray, 
               mode: str = 'average', weights: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if ir_img.shape != vis_img.shape:
        raise ValueError("红外图像和可见光图像的尺寸必须相同")

    if len(ir_img.shape) > 2 and ir_img.shape[2] > 1:
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    if len(vis_img.shape) > 2 and vis_img.shape[2] > 1:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        
    if ir_img.max() <= 1.0:
        ir_img = (ir_img * 255).astype(np.uint8)
    if vis_img.max() <= 1.0:
        vis_img = (vis_img * 255).astype(np.uint8)
    
    if mode == 'average':
        fused = (ir_img.astype(np.float32) + vis_img.astype(np.float32)) / 2
        fused = fused.astype(np.uint8)
        fused_rgb = cv2.cvtColor(fused, cv2.COLOR_GRAY2RGB)
    
    elif mode == 'weighted':
        if weights is None:
            weights = (0.4, 0.6)
        
        fused = (ir_img.astype(np.float32) * weights[0] + 
                vis_img.astype(np.float32) * weights[1])
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        fused_rgb = cv2.cvtColor(fused, cv2.COLOR_GRAY2RGB)
    
    elif mode == 'false_color':
        fused_rgb = np.zeros((ir_img.shape[0], ir_img.shape[1], 3), dtype=np.uint8)
        fused_rgb[:, :, 0] = ir_img
        fused_rgb[:, :, 1] = vis_img
        fused_rgb[:, :, 2] = vis_img
    
    elif mode == 'layered':
        fused_rgb = np.zeros((ir_img.shape[0], ir_img.shape[1], 3), dtype=np.uint8)
        fused_rgb[:, :, 0] = ir_img
        fused_rgb[:, :, 1] = vis_img
        fused_rgb[:, :, 2] = ((ir_img.astype(np.float32) + vis_img.astype(np.float32)) / 2).astype(np.uint8)
    
    else:
        raise ValueError(f"不支持的融合模式: {mode}")
    
    return fused_rgb

def apply_color_map(img: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    return cv2.applyColorMap(img, colormap)

def alpha_blend(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError("两个图像的尺寸必须相同")
    
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    return blended 