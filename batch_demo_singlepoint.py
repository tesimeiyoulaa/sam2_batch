import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ----------------------------
# 配置
# ----------------------------
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

input_image_dir = "/home/zrh/sam2/offroad"      # 输入图片目录
output_mask_dir = "/home/zrh/sam2/offroad/sam_mask"      # 输出 mask 目录
os.makedirs(output_mask_dir, exist_ok=True)

# ----------------------------
# 加载模型和 Mask Generator
# ----------------------------
sam2 = build_sam2(model_cfg, sam2_checkpoint)
mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    box_nms_thresh=0.8,
    min_mask_region_area=100.0,
    use_m2m=True,
)

# ----------------------------
# 辅助函数：显示和保存 masks
# ----------------------------
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, (0,0,0,1), 1)
    ax.imshow(img)

def save_masks_separately(anns, image_name, output_dir):
    """保存每个 mask 为单独的文件"""
    os.makedirs(output_dir, exist_ok=True)
    for i, ann in enumerate(anns):
        mask = ann['segmentation'].astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_path = os.path.join(output_dir, f"{image_name}_mask_{i:03d}.png")
        mask_img.save(mask_path)

def save_masks_combined(anns, image_name, output_dir):
    """保存所有 masks 合并在一张图上（不同颜色）"""
    if len(anns) == 0:
        return
    
    # 创建彩色 mask 叠加图
    height, width = anns[0]['segmentation'].shape
    combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for ann in anns:
        mask = ann['segmentation']
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        combined_mask[mask] = color
    
    combined_img = Image.fromarray(combined_mask)
    combined_path = os.path.join(output_dir, f"{image_name}_combined_masks.png")
    combined_img.save(combined_path)

# ----------------------------
# 批量处理图片
# ----------------------------
for filename in os.listdir(input_image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_image_dir, filename)
        image_name = os.path.splitext(filename)[0]
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        print(f"Processing {filename}...")
        
        # 生成 masks
        masks = mask_generator_2.generate(image)
        
        # 保存单独的 masks
        save_masks_separately(masks, image_name, output_mask_dir)
        
        # 保存合并的 masks
        save_masks_combined(masks, image_name, output_mask_dir)
        
        # 可选：生成可视化图
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        vis_path = os.path.join(output_mask_dir, f"{image_name}_visualization.png")
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"✅ Finished processing {filename}, generated {len(masks)} masks")
