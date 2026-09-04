# 人脸识别交互系统

conda activate face_rec

python app.py


## 项目结构

FaceRecognition/
├── app.py # Flask 主入口，启动 Web 服务
├── register_face.py # 人脸录入工具（摄像头现场拍照入库）
├── config.py # 全局配置（阈值、摄像头ID、路径等）
├── requirements.txt # Python 依赖清单
│
├── core/ # 核心算法模块
│ ├── init.py
│ ├── face_engine.py # 模型加载、检测人脸、提取特征
│ ├── db_manager.py # 数据库管理（增删改查 + 特征缓存）
│ ├── face_matcher.py # 余弦相似度计算与阈值判定
│ └── camera_handler.py # 摄像头采集与帧处理
│
├── utils/ # 辅助工具
│ └── image_utils.py # 解决 OpenCV 中文乱码（PIL 绘制）
│
├── data/ # 数据存储（自动生成）
│ ├── face_db/ # 基准照片库（按姓名分文件夹存放）
│ ├── embeddings_cache/ # 特征向量缓存文件 (.pkl)
│ └── logs/ # 运行日志（预留）
│
├── templates/ # 前端模板
│ └── index.html # 主页面
│
└── static/ # 静态资源（CSS/JS，预留）