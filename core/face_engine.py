# core/face_engine.py
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import MODELS_DIR, DETECTION_UPSCALE
import os
from PIL import Image

class FaceEngine:
    """
    人脸引擎：负责检测人脸 + 提取512维特征向量
    """
    def __init__(self, device='cpu'):
        self.device = device
        
        # 1. 初始化 MTCNN 人脸检测器
        self.detector = MTCNN(
            keep_all=True,
            device=device,
            selection_method='largest',
            post_process=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.8],
            factor=0.709,
            image_size=160
        )
        
        # 2. 初始化 InceptionResnetV1 识别器
        self.recognizer = InceptionResnetV1(
            pretrained='vggface2',
            device=device
        ).eval()
        
        print("✅ FaceEngine 初始化完成，模型已加载")
    
    def detect_faces(self, frame):
        """
        检测图片中所有人脸，返回裁剪后的人脸张量列表和对应的边界框
        参数:
            frame: OpenCV 格式的图片 (BGR, numpy.ndarray)
        返回:
            faces: 列表，每个元素为 torch.Tensor，shape=(3, 160, 160)，已归一化到 [-1,1]
            boxes: 列表，每个元素为 [x1, y1, x2, y2] (整数坐标)
            probs: 列表，每个元素为该人脸的检测置信度
        """
        # OpenCV 是 BGR，MTCNN 需要 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测人脸，返回 (boxes, probs, landmarks)
        boxes, probs, landmarks = self.detector.detect(rgb_frame, landmarks=True)
        
        if boxes is None:
            return [], [], []
        
        # 确保 boxes 是 numpy 数组，并转换为整数
        boxes = np.array(boxes).astype(int)
        probs = np.array(probs).tolist()
        
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # 确保坐标不越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # 裁剪人脸区域 (BGR)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            
            # ---- 手动预处理：转为 RGB，resize 到 160x160，归一化到 [-1,1] ----
            # 转为 RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            # resize 到 160x160 (使用 PIL 或 cv2)
            face_resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            # 转换为 float32 并归一化到 [0,1]
            face_float = face_resized.astype(np.float32) / 255.0
            # 归一化到 [-1, 1]
            face_normalized = (face_float - 0.5) / 0.5
            # 转换为 (C, H, W) 格式
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).float()
            faces.append(face_tensor)
        
        # boxes 转为列表
        boxes_list = boxes.tolist()
        return faces, boxes_list, probs
    
    def get_embedding(self, face_tensor):
        """
        输入一张人脸张量，提取 512 维特征向量
        参数:
            face_tensor: torch.Tensor, shape=(3, 160, 160)，范围 [-1, 1]
        返回:
            embedding: numpy.ndarray, shape=(512,)，已归一化（模长为1）
        """
        if face_tensor is None or face_tensor.shape[0] != 3:
            return None
        
        # 添加 batch 维度: (3,160,160) -> (1,3,160,160)
        batch_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.recognizer(batch_tensor)
        
        embedding_np = embedding.cpu().numpy().flatten()
        # L2 归一化
        norm = np.linalg.norm(embedding_np)
        if norm > 1e-8:
            embedding_np = embedding_np / norm
        return embedding_np