import os

# ---------- 路径配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据存储路径
DATA_DIR = os.path.join(BASE_DIR, "data")
FACE_DB_DIR = os.path.join(DATA_DIR, "face_db")          # 基准照片存放处
CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")   # 特征缓存
LOGS_DIR = os.path.join(DATA_DIR, "logs")                # 日志
TEMP_DIR = os.path.join(DATA_DIR, "temp")                # 临时抓拍

# 模型权重路径（insightface会自动下载，我们指定存放位置）
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------- 摄像头配置 ----------
CAMERA_ID = 0          # 0为笔记本内置摄像头，1为外置USB摄像头（根据实际修改）
FRAME_WIDTH = 640      # 采集宽度（降低分辨率可加速）
FRAME_HEIGHT = 480

# ---------- 算法参数 ----------
SIMILARITY_THRESHOLD = 0.65   # 余弦相似度阈值（高于此值判定为同一人）
FRAME_SKIP = 2                # 性能优化：每2帧推理1次（减小可提高实时性，增大可降低CPU）

# ---------- 人脸检测参数 ----------
DETECTION_UPSCALE = 1.0       # 检测上采样倍数（1.0最快，1.5~2.0可提高小脸检出率）

# ---------- 自动创建必要的文件夹 ----------
def create_directories():
    """自动创建项目所需的文件夹结构"""
    dirs = [DATA_DIR, FACE_DB_DIR, CACHE_DIR, LOGS_DIR, TEMP_DIR, MODELS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # 创建一个 .gitkeep 文件，确保空文件夹被git跟踪（非必须）
        keep_file = os.path.join(d, ".gitkeep")
        if not os.path.exists(keep_file):
            with open(keep_file, 'w') as f:
                pass

# 初始化时自动创建文件夹
create_directories()

# 打印配置信息，确认路径正确
if __name__ == "__main__":
    print(f"✅ 项目根目录: {BASE_DIR}")
    print(f"✅ 人脸标准库: {FACE_DB_DIR}")
    print(f"✅ 缓存目录: {CACHE_DIR}")
    print("✅ 所有文件夹已创建完毕！")