"""
预处理模块
"""
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
from data import TARGET_WIDTH, TARGET_HEIGHT


def preprocess_for_model(input_image):
    """
    与训练阶段保持一致的推理预处理：灰度 -> resize -> CLAHE
    
    Args:
        input_image: 输入图像（可以是路径或 numpy 数组）
    
    Returns:
        (img_resized, img_clahe): 调整大小后的图像和 CLAHE 增强后的图像
    """
    if isinstance(input_image, str):
        img = Image.open(input_image)
        try:
            exif = img.getexif()
            if exif:
                orientation = exif.get(0x0112, 1)
                if orientation == 2:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90)
                elif orientation == 7:
                    img = img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90)
        except Exception:
            pass
        input_image = np.array(img)
    
    if input_image.ndim == 2:
        img_gray = input_image
    elif input_image.shape[2] == 3:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGBA2GRAY)
    
    img_resized = cv2.resize(
        img_gray,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    img_clahe = exposure.equalize_adapthist(img_resized, clip_limit=0.02).astype(np.float32)
    return img_resized, img_clahe
