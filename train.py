import os
import torch
import argparse
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import TRAIN_CONFIG, DATA_CONFIG, DEVICE, RANDOM_SEED
from src.models.siamese_net import SiameseNet
from src.models.loss import get_loss_function
from src.utils.data_utils import set_seed, prepare_datasets, save_dataset_split, load_dataset_split
from src.dataset import create_dataloaders
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='训练CNN图像配准模型')
    
    parser.add_argument('--vis_dir', type=str, default=DATA_CONFIG['VIS_DIR'],
                       help='可见光图像目录')
    parser.add_argument('--ir_dir', type=str, default=DATA_CONFIG['IR_DIR'],
                       help='红外图像目录')
    parser.add_argument('--json_path', type=str, default=DATA_CONFIG['JSON_PATH'],
                       help='变换参数JSON文件路径')
    parser.add_argument('--model_dir', type=str, default=DATA_CONFIG['MODEL_DIR'],
                       help='模型保存目录')
    
    parser.add_argument('--train_percentage', type=float, default=DATA_CONFIG['TRAIN_PERCENTAGE'],
                       help='训练集占比')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['BATCH_SIZE'],
                       help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=TRAIN_CONFIG['NUM_EPOCHS'],
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=TRAIN_CONFIG['LEARNING_RATE'],
                       help='学习率')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='随机种子')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"使用设备: {DEVICE}")
    
    train_ids_path = os.path.join(args.model_dir, 'train.txt')
    valid_ids_path = os.path.join(args.model_dir, 'valid.txt')
    
    if os.path.exists(train_ids_path) and os.path.exists(valid_ids_path):
        print("使用已有的数据集划分")
        train_ids, valid_ids = load_dataset_split(args.model_dir)
    else:
        print("准备数据集并划分训练集和验证集")
        train_ids, valid_ids = prepare_datasets(
            vis_dir=args.vis_dir,
            ir_dir=args.ir_dir,
            json_path=args.json_path,
            train_percentage=args.train_percentage
        )
        save_dataset_split(train_ids, valid_ids, args.model_dir)
    
    train_loader, valid_loader = create_dataloaders(
        train_ids=train_ids,
        valid_ids=valid_ids,
        batch_size=args.batch_size,
        vis_dir=args.vis_dir,
        ir_dir=args.ir_dir,
        json_path=args.json_path
    )
    
    model = SiameseNet()
    print(f"模型参数数量: {model.get_parameter_count():,}")
    
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=TRAIN_CONFIG['WEIGHT_DECAY']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    loss_fn = get_loss_function()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=DEVICE
    )
    
    trainer.train(resume_from=args.resume)
    
    print("训练完成!")

if __name__ == '__main__':
    main() 