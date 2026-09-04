# test_engine_v2.py
import cv2
import numpy as np
from core.face_engine import FaceEngine

print("1. 初始化 FaceEngine...")
engine = FaceEngine()

print("2. 尝试打开摄像头 (ID=0)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头 ID=0，尝试 ID=1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 摄像头 ID=1 也无法打开，请检查物理连接或权限。")
        exit()
else:
    print("✅ 摄像头打开成功")

print("3. 读取一帧...")
ret, frame = cap.read()
if not ret:
    print("❌ 读取帧失败")
    cap.release()
    exit()
else:
    print(f"✅ 读取帧成功，尺寸: {frame.shape}")

print("4. 执行人脸检测...")
faces, boxes, probs = engine.detect_faces(frame)
if faces:
    print(f"✅ 检测到 {len(faces)} 张人脸")
    emb = engine.get_embedding(faces[0])
    print(f"✅ 特征向量维度: {emb.shape}, 模长: {np.linalg.norm(emb):.4f}")
else:
    print("⚠️ 未检测到人脸，请确保光照充足且人脸正对摄像头")

cap.release()
print("测试完成")