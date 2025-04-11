import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from ..config import DATA_CONFIG, RANDOM_SEED
from typing import Dict, List, Tuple, Optional

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_json_data(json_path: str) -> Dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_transform_params(json_data: Dict, image_id: str) -> Optional[Dict]:
    for item in json_data.get('images', []):
        if item.get('id') == image_id:
            return {
                'scale': item.get('scale', 1.0),
                'horizontalPercent': item.get('horizontalPercent', 0.0),
                'verticalPercent': item.get('verticalPercent', 0.0),
                'rotation': item.get('rotation', 0.0)
            }
    return None

def find_valid_data_pairs(vis_dir: str, ir_dir: str, json_path: str) -> List[str]:
    json_data = load_json_data(json_path)
    valid_ids = []
    
    vis_files = [f for f in os.listdir(vis_dir) if f.endswith('.jpg')]
    
    print("正在查找有效的数据对...")
    for vis_file in tqdm(vis_files, desc="检查数据对"):
        image_id = os.path.splitext(vis_file)[0]
        
        ir_file = os.path.join(ir_dir, f"{image_id}.jpg")
        if not os.path.exists(ir_file):
            continue
        
        transform_params = get_transform_params(json_data, image_id)
        if transform_params is None:
            continue
        
        valid_ids.append(image_id)
    
    print(f"找到 {len(valid_ids)} 个有效数据对")
    return valid_ids

def prepare_datasets(vis_dir: str = None, ir_dir: str = None, json_path: str = None, 
                    train_percentage: float = None) -> Tuple[List[str], List[str]]:
    vis_dir = vis_dir or DATA_CONFIG['VIS_DIR']
    ir_dir = ir_dir or DATA_CONFIG['IR_DIR']
    json_path = json_path or DATA_CONFIG['JSON_PATH']
    train_percentage = train_percentage or DATA_CONFIG['TRAIN_PERCENTAGE']
    
    set_seed()
    
    valid_ids = find_valid_data_pairs(vis_dir, ir_dir, json_path)
    
    random.shuffle(valid_ids)
    
    train_size = int(len(valid_ids) * train_percentage)
    
    train_ids = valid_ids[:train_size]
    valid_ids = valid_ids[train_size:]
    
    return train_ids, valid_ids

def save_dataset_split(train_ids: List[str], valid_ids: List[str], 
                      model_dir: str = None) -> None:
    model_dir = model_dir or DATA_CONFIG['MODEL_DIR']
    
    os.makedirs(model_dir, exist_ok=True)
    
    train_path = os.path.join(model_dir, 'train.txt')
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_ids))
    
    valid_path = os.path.join(model_dir, 'valid.txt')
    with open(valid_path, 'w') as f:
        f.write('\n'.join(valid_ids))
    
    print(f"数据集划分已保存: 训练集 {len(train_ids)} 个, 验证集 {len(valid_ids)} 个")
    print(f"文件保存路径: {train_path} 和 {valid_path}")

def load_dataset_split(model_dir: str = None) -> Tuple[List[str], List[str]]:
    model_dir = model_dir or DATA_CONFIG['MODEL_DIR']
    
    train_path = os.path.join(model_dir, 'train.txt')
    if not os.path.exists(train_path):
        return [], []
    
    valid_path = os.path.join(model_dir, 'valid.txt')
    if not os.path.exists(valid_path):
        return [], []
    
    with open(train_path, 'r') as f:
        train_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(valid_path, 'r') as f:
        valid_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    return train_ids, valid_ids 