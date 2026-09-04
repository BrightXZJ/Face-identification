# core/face_matcher.py
import numpy as np
from config import SIMILARITY_THRESHOLD

class FaceMatcher:
    """
    人脸匹配器：计算相似度，判定身份
    """
    def __init__(self, db_manager, threshold=None):
        """
        参数:
            db_manager: DatabaseManager 实例
            threshold: 相似度阈值，不传则使用 config 中的默认值
        """
        self.db = db_manager
        self.threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        计算两个向量的余弦相似度
        假设 vec1, vec2 已经是 L2 归一化的，则余弦相似度 = np.dot(vec1, vec2)
        为了安全，这里手动计算
        """
        # 防止除零
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def match(self, query_embedding):
        """
        在数据库中匹配最相似的人员
        参数:
            query_embedding: numpy.ndarray, 512维待测特征向量
        返回:
            name: 匹配的姓名 或 "Unknown"
            score: 最高相似度分数 (0~1)
        """
        if query_embedding is None:
            return "Unknown", 0.0
        
        best_name = "Unknown"
        best_score = 0.0
        
        # 遍历数据库中所有人
        for name, db_embedding in self.db.get_all_embeddings().items():
            score = self.cosine_similarity(query_embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_name = name
        
        # 判断是否超过阈值
        if best_score < self.threshold:
            return "Unknown", best_score
        
        return best_name, best_score
    
    def match_batch(self, query_embeddings_list):
        """
        批量匹配（用于一张图中有多张人脸的情况）
        返回: [(name, score), ...]
        """
        results = []
        for emb in query_embeddings_list:
            results.append(self.match(emb))
        return results