import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

from .config import DATA_CONFIG
from .utils.data_utils import load_json_data, get_transform_params
from .models.loss import normalize_params

class IRVISDataset(Dataset):
    def __init__(self, 
                image_ids: List[str], 
                vis_dir: str, 
                ir_dir: str, 
                json_path: str,
                transform=None):
        self.image_ids = image_ids
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.transform = transform
        self.image_size = DATA_CONFIG['IMAGE_SIZE']
        
        self.json_data = load_json_data(json_path)
        
        self.vis_mean = DATA_CONFIG['NORMALIZE_VIS_MEAN']
        self.vis_std = DATA_CONFIG['NORMALIZE_VIS_STD']
        self.ir_mean = DATA_CONFIG['NORMALIZE_IR_MEAN']
        self.ir_std = DATA_CONFIG['NORMALIZE_IR_STD']
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        vis_path = os.path.join(self.vis_dir, f"{image_id}.jpg")
        vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
        
        ir_path = os.path.join(self.ir_dir, f"{image_id}.jpg")
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        
        vis_img = cv2.resize(vis_img, self.image_size)
        ir_img = cv2.resize(ir_img, self.image_size)
        
        params = get_transform_params(self.json_data, image_id)
        
        transform_tensor = torch.tensor([
            params['rotation'],
            params['scale'],
            params['horizontalPercent'] / 100.0,
            params['verticalPercent'] / 100.0
        ], dtype=torch.float32)
        
        normalized_transform = normalize_params(transform_tensor.unsqueeze(0)).squeeze(0)
        
        if self.transform:
            augmented = self.transform(image=vis_img, image2=ir_img)
            vis_img = augmented['image']
            ir_img = augmented['image2']
        
        vis_tensor = torch.tensor(vis_img, dtype=torch.float32).unsqueeze(0) / 255.0
        ir_tensor = torch.tensor(ir_img, dtype=torch.float32).unsqueeze(0) / 255.0
        
        vis_tensor = (vis_tensor - self.vis_mean) / self.vis_std
        ir_tensor = (ir_tensor - self.ir_mean) / self.ir_std
        
        return {
            'image_id': image_id,
            'vis_img': vis_tensor,
            'ir_img': ir_tensor,
            'transform_params': normalized_transform
        }

def create_dataloaders(train_ids: List[str], 
                      valid_ids: List[str], 
                      batch_size: int = None,
                      vis_dir: str = None,
                      ir_dir: str = None,
                      json_path: str = None,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    batch_size = batch_size or DATA_CONFIG['BATCH_SIZE']
    vis_dir = vis_dir or DATA_CONFIG['VIS_DIR']
    ir_dir = ir_dir or DATA_CONFIG['IR_DIR']
    json_path = json_path or DATA_CONFIG['JSON_PATH']
    
    train_dataset = IRVISDataset(
        image_ids=train_ids,
        vis_dir=vis_dir,
        ir_dir=ir_dir, 
        json_path=json_path
    )
    
    valid_dataset = IRVISDataset(
        image_ids=valid_ids,
        vis_dir=vis_dir,
        ir_dir=ir_dir,
        json_path=json_path
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader 