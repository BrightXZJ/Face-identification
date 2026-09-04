# core/db_manager.py
import os
import pickle
import shutil
import cv2
import numpy as np
from config import FACE_DB_DIR, CACHE_DIR
from core.face_engine import FaceEngine

class DatabaseManager:
    """
    人脸数据库管理器：支持增删改查，自动维护特征缓存
    """
    def __init__(self):
        self.db_path = FACE_DB_DIR          # data/face_db/
        self.cache_path = os.path.join(CACHE_DIR, 'embeddings.pkl')  # 缓存文件
        self.engine = FaceEngine()          # 依赖特征提取引擎
        self.embeddings = {}                # 字典: {name: np.ndarray(512,)}
        self.load_database()                # 初始化时自动加载
        
    def load_database(self):
        """
        加载数据库：优先读取缓存，若缓存不存在或校验失败，则扫描文件夹重建
        """
        # 确保数据库文件夹存在
        os.makedirs(self.db_path, exist_ok=True)
        
        # 尝试读取缓存
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"✅ 从缓存加载了 {len(self.embeddings)} 个人的人脸特征")
                return
            except Exception as e:
                print(f"⚠️ 缓存加载失败 ({e})，将重新构建...")
        
        # 缓存不存在或损坏，全量扫描重建
        self._rebuild_cache()
    
    def _rebuild_cache(self):
        """
        扫描 face_db 文件夹，为每个人提取特征，重建缓存
        """
        self.embeddings = {}
        # 遍历 face_db 下的所有子文件夹（每个人一个文件夹）
        if not os.path.exists(self.db_path):
            return
        
        for person_name in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            # 查找该文件夹下的第一张图片（支持 jpg, png, jpeg）
            img_files = [f for f in os.listdir(person_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not img_files:
                continue
            
            # 取第一张图片作为基准
            img_path = os.path.join(person_dir, img_files[0])
            embedding = self._extract_from_image_path(img_path)
            
            if embedding is not None:
                self.embeddings[person_name] = embedding
                print(f"✅ 加载成功: {person_name}")
        
        # 保存缓存
        self._save_cache()
        print(f"✅ 重建缓存完成，共 {len(self.embeddings)} 人")
    
    def _extract_from_image_path(self, image_path):
        """
        内部方法：从图片路径提取特征向量
        """
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        
        faces, boxes, probs = self.engine.detect_faces(frame)
        if not faces:
            return None
        
        # 取检测到的第一张脸（置信度最高的）
        best_face = faces[0]
        return self.engine.get_embedding(best_face)
    
    def _save_cache(self):
        """保存缓存到文件"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def add_person(self, name, image_path):
        """
        新增人员：保存图片到该人名目录，提取特征，更新缓存
        参数:
            name: 姓名（也将作为文件夹名）
            image_path: 临时图片路径（如前端上传的图片，或摄像头抓拍）
        返回:
            bool: 是否成功
        """
        # 1. 创建该人员的文件夹
        person_dir = os.path.join(self.db_path, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # 2. 复制/移动图片到目标目录（命名为 base.jpg）
        base_img_path = os.path.join(person_dir, 'base.jpg')
        shutil.copy2(image_path, base_img_path)
        
        # 3. 提取特征
        embedding = self._extract_from_image_path(base_img_path)
        if embedding is None:
            # 若提取失败，删除文件夹（回滚）
            shutil.rmtree(person_dir)
            return False
        
        # 4. 更新内存字典并保存缓存
        self.embeddings[name] = embedding
        self._save_cache()
        print(f"✅ 成功添加人员: {name}")
        return True
    
    def delete_person(self, name):
        """
        删除人员：删除文件夹，从缓存移除
        返回:
            bool: 是否成功
        """
        if name not in self.embeddings:
            return False
        
        # 1. 删除文件夹
        person_dir = os.path.join(self.db_path, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        
        # 2. 从缓存删除
        del self.embeddings[name]
        self._save_cache()
        print(f"✅ 成功删除人员: {name}")
        return True
    
    def update_person(self, name, image_path):
        """
        更新人员基准照片（先删后加）
        """
        self.delete_person(name)
        return self.add_person(name, image_path)
    
    def get_all_names(self):
        """返回所有人员姓名列表"""
        return list(self.embeddings.keys())
    
    def get_embedding_by_name(self, name):
        """根据姓名获取特征向量"""
        return self.embeddings.get(name, None)
    
    def get_all_embeddings(self):
        """获取所有人名与特征向量的字典（供比对模块使用）"""
        return self.embeddings