import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Union, List

def apply_transform(image: np.ndarray, 
                   theta: float = 0.0, 
                   scale: float = 1.0, 
                   dx: float = 0.0, 
                   dy: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    
    tx = dx * width
    ty = dy * height
    
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, theta, scale)
    
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    
    transformed_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return transformed_image

def apply_inverse_transform(image: np.ndarray, 
                          theta: float = 0.0, 
                          scale: float = 1.0, 
                          dx: float = 0.0, 
                          dy: float = 0.0) -> np.ndarray:
    inv_theta = -theta
    inv_scale = 1.0 / scale if scale != 0 else 1.0
    
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    matrix = cv2.getRotationMatrix2D(center, theta, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    
    inv_matrix = cv2.invertAffineTransform(matrix)
    
    inverse_transformed_image = cv2.warpAffine(image, inv_matrix, (width, height), 
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return inverse_transformed_image

def params_to_transformation_matrix(theta: float, scale: float, dx: float, dy: float, 
                                  image_size: Tuple[int, int]) -> np.ndarray:
    height, width = image_size
    center = (width / 2, height / 2)
    
    matrix = cv2.getRotationMatrix2D(center, theta, scale)
    
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    
    return matrix

def transformation_matrix_to_params(matrix: np.ndarray, 
                                   image_size: Tuple[int, int]) -> Dict[str, float]:
    height, width = image_size
    
    scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale = (scale_x + scale_y) / 2
    
    theta_rad = np.arctan2(matrix[0, 1], matrix[0, 0])
    theta = np.degrees(theta_rad)
    
    center = (width / 2, height / 2)
    pure_rotation_scale = cv2.getRotationMatrix2D(center, theta, scale)
    
    tx = matrix[0, 2] - pure_rotation_scale[0, 2]
    ty = matrix[1, 2] - pure_rotation_scale[1, 2]
    
    dx = tx / width
    dy = ty / height
    
    return {
        'theta': theta,
        'scale': scale,
        'dx': dx,
        'dy': dy
    }

def transform_batch_tensor_to_params(batch_tensor: torch.Tensor) -> List[Dict[str, float]]:
    batch_np = batch_tensor.detach().cpu().numpy()
    
    params_list = []
    for i in range(batch_np.shape[0]):
        params = {
            'theta': float(batch_np[i, 0]),
            'scale': float(batch_np[i, 1]),
            'dx': float(batch_np[i, 2]),
            'dy': float(batch_np[i, 3])
        }
        params_list.append(params)
    
    return params_list 