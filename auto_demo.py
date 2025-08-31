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
# 批量处理
# ----------------------------
for filename in os.listdir(input_image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # 设置图像
        predictor.set_image(image)

        # 设置提示（这里以图像中心点为例，你可以替换为你的提示）
        h, w, _ = image.shape
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])

        # 预测
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        # 保存 mask
        mask = masks[0].astype(np.uint8) * 255  # 转成 0/255 的二值图像
        mask_image = Image.fromarray(mask)
        output_path = os.path.join(
            output_mask_dir, os.path.splitext(filename)[0] + ".png"
        )
        mask_image.save(output_path)

        print(f"✅ Processed and saved mask for {filename}")
