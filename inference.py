import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

from src.config import DATA_CONFIG, INFERENCE_CONFIG, DEVICE
from src.models.siamese_net import SiameseNet
from src.models.loss import normalize_params
from src.utils.data_utils import load_json_data, get_transform_params
from src.utils.transform_utils import apply_inverse_transform
from src.utils.rgb import fuse_images
from src.utils.vis_utils import create_fusion_visualization, setup_matplotlib_chinese

def parse_args():
    parser = argparse.ArgumentParser(description='使用CNN模型进行图像配准')
    
    parser.add_argument('--vis_dir', type=str, default=DATA_CONFIG['VIS_DIR'],
                        help='可见光图像目录')
    parser.add_argument('--ir_dir', type=str, default=DATA_CONFIG['IR_DIR'],
                        help='红外图像目录')
    parser.add_argument('--json_path', type=str, default=DATA_CONFIG['JSON_PATH'],
                        help='变换参数JSON文件路径')
    
    parser.add_argument('--model_path', type=str, default=INFERENCE_CONFIG['MODEL_PATH'],
                        help='模型路径')
    parser.add_argument('--output_dir', type=str, default=INFERENCE_CONFIG['INFERENCE_DIR'],
                        help='输出目录')
    
    parser.add_argument('--image_id', type=str, default=None,
                        help='指定图像ID进行推理')
    parser.add_argument('--fusion_mode', type=str, default=INFERENCE_CONFIG['FUSION_MODE'],
                        choices=['average', 'weighted', 'false_color', 'layered'],
                        help='融合模式')
    
    return parser.parse_args()

def load_model(model_path: str) -> SiameseNet:
    model = SiameseNet()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(DEVICE)
    
    print(f"已加载模型: {model_path}")
    print(f"模型参数数量: {model.get_parameter_count():,}")
    
    return model

def preprocess_image(image_path: str, image_size: tuple = None) -> torch.Tensor:
    image_size = image_size or DATA_CONFIG['IMAGE_SIZE']
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, image_size)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
    tensor = (tensor - DATA_CONFIG['NORMALIZE_IR_MEAN']) / DATA_CONFIG['NORMALIZE_IR_STD']
    tensor = tensor.unsqueeze(0)
    
    return tensor

def predict_transform_params(model: SiameseNet, 
                           ir_img: torch.Tensor, 
                           vis_img: torch.Tensor) -> Dict[str, float]:
    ir_img = ir_img.to(DEVICE)
    vis_img = vis_img.to(DEVICE)
    
    with torch.no_grad():
        output = model(ir_img, vis_img)
    
    denorm_output = normalize_params(output, denormalize=True)
    
    params = {
        'theta': float(denorm_output[0, 0].cpu().numpy()),
        'scale': float(denorm_output[0, 1].cpu().numpy()),
        'dx': float(denorm_output[0, 2].cpu().numpy()),
        'dy': float(denorm_output[0, 3].cpu().numpy())
    }
    
    return params

def process_image(image_id: str, 
                 model: SiameseNet, 
                 args: argparse.Namespace,
                 json_data: Optional[Dict] = None) -> Dict:
    ir_path = os.path.join(args.ir_dir, f"{image_id}.jpg")
    vis_path = os.path.join(args.vis_dir, f"{image_id}.jpg")
    
    if not os.path.exists(ir_path) or not os.path.exists(vis_path):
        print(f"错误: 找不到图像 {image_id}")
        return None
    
    ir_img_orig = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    vis_img_orig = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    
    ir_tensor = preprocess_image(ir_path)
    vis_tensor = preprocess_image(vis_path)
    
    pred_params = predict_transform_params(model, ir_tensor, vis_tensor)
    
    true_params = None
    if json_data:
        json_params = get_transform_params(json_data, image_id)
        if json_params:
            true_params = {
                'theta': json_params['rotation'],
                'scale': json_params['scale'],
                'dx': json_params['horizontalPercent'] / 100.0,
                'dy': json_params['verticalPercent'] / 100.0
            }
    
    aligned_ir = apply_inverse_transform(
        ir_img_orig,
        theta=pred_params['theta'],
        scale=pred_params['scale'],
        dx=pred_params['dx'],
        dy=pred_params['dy']
    )
    
    fused_img = fuse_images(aligned_ir, vis_img_orig, mode=args.fusion_mode)
    
    os.makedirs(args.output_dir, exist_ok=True)
    img_output_dir = os.path.join(args.output_dir, image_id)
    os.makedirs(img_output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(img_output_dir, 'ir_original.jpg'), ir_img_orig)
    cv2.imwrite(os.path.join(img_output_dir, 'ir_aligned.jpg'), aligned_ir)
    cv2.imwrite(os.path.join(img_output_dir, 'vis.jpg'), vis_img_orig)
    cv2.imwrite(os.path.join(img_output_dir, 'fused.jpg'), fused_img)
    
    create_fusion_visualization(
        ir_img=ir_img_orig,
        aligned_ir_img=aligned_ir,
        vis_img=vis_img_orig,
        fused_img=fused_img,
        transform_params=pred_params,
        save_path=os.path.join(img_output_dir, 'fusion.jpg'),
        title=f"图像ID: {image_id}"
    )
    
    result = {
        'image_id': image_id,
        'predicted_params': pred_params,
        'true_params': true_params
    }
    
    with open(os.path.join(img_output_dir, 'params.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def main():
    args = parse_args()
    model = load_model(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    json_data = None
    if args.json_path and os.path.exists(args.json_path):
        json_data = load_json_data(args.json_path)
    
    setup_matplotlib_chinese()
    
    results = []
    
    if args.image_id:
        result = process_image(args.image_id, model, args, json_data)
        if result:
            results.append(result)
    else:
        image_ids = []
        for filename in os.listdir(args.vis_dir):
            if filename.endswith('.jpg'):
                image_id = os.path.splitext(filename)[0]
                image_ids.append(image_id)
        
        print(f"找到 {len(image_ids)} 个图像")
        
        for image_id in tqdm(image_ids, desc="处理图像"):
            result = process_image(image_id, model, args, json_data)
            if result:
                results.append(result)
    
    if results:
        summary_path = os.path.join(args.output_dir, 'results.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"已处理 {len(results)} 个图像")
        print(f"结果已保存到: {summary_path}")

if __name__ == "__main__":
    main() 