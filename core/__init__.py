# core/__init__.py
"""
core包：封装所有核心算法与业务逻辑
"""
from .face_engine import FaceEngine
from .camera_handler import CameraHandler
from .db_manager import DatabaseManager
from .face_matcher import FaceMatcher

# 为了方便，定义 __all__ 指定 from core import * 时导入哪些
__all__ = ['FaceEngine', 'CameraHandler', 'DatabaseManager', 'FaceMatcher']