import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----------------------------
# 配置
# ----------------------------
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

input_image_dir = "batchimage/3/V/"      # 输入图片目录
output_mask_dir = "/home/zrh/sam2/batchimage/sam_mask/3"      # 输出 mask 目录
os.makedirs(output_mask_dir, exist_ok=True)

# ----------------------------
# 加载模型
# ----------------------------
predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))

# ----------------------------
# 批量处理图片（限制最大边为 512）
# ----------------------------
def resize_image_keep_aspect(image, max_size=512):
    """将图像等比例缩放，使得长边不超过 max_size"""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image  # 不需要缩放
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS))
    return resized

for filename in os.listdir(input_image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_image_dir, filename)
        image_name = os.path.splitext(filename)[0]
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        # 限制分辨率：长边不超过 512
        image = resize_image_keep_aspect(image, max_size=512)
        
        print(f"Processing {filename} (resized to {image.shape[1]}x{image.shape[0]})...")
        
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
