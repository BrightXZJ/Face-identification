# utils/image_utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_chinese_text(img, text, position, font_size=24, color=(0, 255, 0), thickness=2):
    """
    在 OpenCV 图片上绘制中文文本（解决 OpenCV 不支持中文的问题）
    参数:
        img: OpenCV 图片 (BGR)
        text: 要绘制的文本（中文）
        position: (x, y) 坐标
        font_size: 字体大小
        color: (B, G, R) 颜色元组
        thickness: 线条粗细
    返回:
        绘制后的图片
    """
    # 转为 PIL 格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体（如果系统没有中文字体，回退到默认字体）
    try:
        # 常见中文字体路径（Windows）
        font_paths = [
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "/System/Library/Fonts/PingFang.ttc", # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # RGB 转 PIL 顺序
    
    # 转回 OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)