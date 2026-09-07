# 人脸识别交互系统

一个基于 Flask、OpenCV、MTCNN 和 FaceNet 的本地人脸识别系统。系统通过摄像头检测人脸，使用 InceptionResnetV1 提取 512 维特征向量，再通过余弦相似度与本地人脸库进行匹配。

## 功能

- 实时人脸识别：显示摄像头视频流，并用绿色框标记已匹配人员、红色框标记未知人员。
- 浏览器拍照录入：通过浏览器摄像头拍照，填写姓名后加入人脸库。
- 人脸库管理：查看已注册人员并删除人员。
- 特征缓存：首次扫描照片库后生成 `data/embeddings_cache/embeddings.pkl`，后续启动优先从缓存加载。
- 独立录入工具：提供基于 OpenCV 窗口的 `register_face.py`，适合不使用网页录入时操作。

## 运行环境

- Windows、macOS 或 Linux
- Python 3.8 及以上
- 可用摄像头
- 能够安装 PyTorch 模型依赖的环境

网页录入需要浏览器允许当前页面访问摄像头；实时识别使用运行 Flask 服务所在机器的摄像头。

## 安装

### 1. 创建并激活 Conda 环境

```bash
conda create -n face_rec python=3.10
conda activate face_rec
```

### 2. 安装项目依赖

```bash
pip install -r requirements.txt
```

当前代码的 `core/face_engine.py` 直接使用 `facenet_pytorch` 和 `torch`。如果安装基础依赖后提示缺少这些模块，请额外执行：

```bash
pip install torch torchvision facenet-pytorch
```

PyTorch 也可以根据机器是否支持 CUDA，按照其官方安装命令选择对应版本。项目当前默认使用 CPU：

```python
FaceEngine(device='cpu')
```

首次创建 `FaceEngine` 时，`InceptionResnetV1(pretrained='vggface2')` 可能会自动下载预训练模型。请确保网络可用，并允许模型保存到项目的 `models/` 目录或相应的 PyTorch 缓存目录。

## 启动网页系统

在项目根目录执行：

```bash
conda activate face_rec
python app.py
```

启动后打开：

```text
http://127.0.0.1:5000
```

主页提供三个入口：

| 页面 | 地址 | 说明 |
| --- | --- | --- |
| 主页 | `/` | 功能导航 |
| 人脸识别 | `/recognition` | 查看实时识别视频流 |
| 添加人脸 | `/add_face` | 浏览器拍照并录入人员 |
| 人脸库管理 | `/manage` | 查看和删除已注册人员 |

## 使用流程

### 录入人员

1. 打开“添加人脸”页面。
2. 允许浏览器使用摄像头。
3. 点击“拍照”，确认画面中包含清晰的人脸。
4. 输入姓名并点击“录入”。
5. 系统检测图片中的第一张人脸，提取特征并保存到 `data/face_db/<姓名>/base.jpg`。

姓名会经过 `secure_filename` 清理。相同姓名不能重复录入；如果图片中没有检测到人脸，录入会失败。

### 开始识别

打开“人脸识别”页面。系统每处理 `FRAME_SKIP` 帧进行一次检测，对每张检测到的人脸提取特征并与人脸库匹配。

- 绿色框：相似度达到阈值，显示姓名和分数。
- 红色框：相似度低于阈值，显示 `Unknown` 和分数。

### 使用命令行录入工具

```bash
python register_face.py
```

在 OpenCV 窗口中：

- 按空格键保存当前人脸。
- 按 `q` 退出。

该工具会将截取的人脸保存为 `data/face_db/<姓名>/base.jpg`。网页录入和该工具不要同时占用同一个摄像头。

## HTTP API

### 获取人员列表

```http
GET /api/persons
```

返回示例：

```json
{"persons": ["Deng_Weixuan", "Xu_Zhijian"]}
```

### 添加人员

```http
POST /api/add_person
Content-Type: multipart/form-data
```

表单字段：

- `name`：人员姓名，必填。
- `image`：图片文件，必填。

### 删除人员

```http
POST /api/delete_person
Content-Type: application/json
```

请求体：

```json
{"name": "Deng_Weixuan"}
```

### 获取实时视频流

```http
GET /video_feed
```

接口返回 MJPEG 视频流，通常由 `/recognition` 页面中的 `<img>` 元素加载。

## 配置

项目配置集中在 `config.py`：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `CAMERA_ID` | `0` | OpenCV 摄像头编号，外接摄像头通常需要改为 `1` 或其他编号 |
| `FRAME_WIDTH` | `640` | 摄像头采集宽度 |
| `FRAME_HEIGHT` | `480` | 摄像头采集高度 |
| `SIMILARITY_THRESHOLD` | `0.65` | 余弦相似度识别阈值，越高越严格 |
| `FRAME_SKIP` | `2` | 每隔多少帧执行一次人脸推理 |
| `DETECTION_UPSCALE` | `1.0` | 检测上采样配置；当前检测器初始化流程未实际使用该值 |

修改配置后重新启动 Flask 服务。阈值过低可能增加误识别，阈值过高可能把已登记人员判定为未知，建议使用实际环境中的样本进行调整。

## 数据目录

```text
data/
├── face_db/                 # 人脸基准照片，每人一个子目录
│   └── <姓名>/base.jpg
├── embeddings_cache/        # 特征缓存
│   └── embeddings.pkl
├── logs/                    # 日志目录，当前预留
└── temp/                    # 网页上传时的临时文件

models/                      # 模型文件目录，启动时自动创建
```

`config.py` 导入时会自动创建这些目录。删除或损坏 `embeddings.pkl` 后，下一次启动会扫描 `data/face_db/` 并重新生成缓存。

## 项目结构

```text
FaceRecognition/
├── app.py                  # Flask 应用入口、页面路由、API 和视频流
├── config.py               # 路径、摄像头和算法参数
├── register_face.py        # OpenCV 窗口录入工具
├── requirements.txt        # Python 依赖清单
├── core/
│   ├── camera_handler.py   # 摄像头采集和 MJPEG 帧生成
│   ├── db_manager.py       # 人脸照片库和特征缓存管理
│   ├── face_engine.py      # MTCNN 检测和 FaceNet 特征提取
│   └── face_matcher.py     # 余弦相似度匹配和阈值判定
├── utils/
│   └── image_utils.py      # OpenCV 图像上的中文文字绘制
├── templates/              # Flask 页面模板
├── static/
│   └── css/style.css       # 页面样式
├── data/                   # 运行时数据，通常不提交特征缓存
├── models/                 # 模型文件
├── test_engine.py          # 模型、摄像头、人脸检测测试脚本
├── test_db_manager.py      # 人脸库和缓存加载测试脚本
└── test_write.py           # 图片写入测试脚本
```

## 检查与测试

初始化和检查人脸库：

```bash
python test_db_manager.py
```

检查模型、摄像头和人脸检测：

```bash
python test_engine.py
```

运行测试前请确保摄像头未被其他应用占用。测试脚本属于硬件和模型 smoke test，不是完整的自动化测试套件。

## 常见问题

### 无法打开摄像头

确认摄像头已连接且没有被其他程序使用；然后在 `config.py` 中尝试修改 `CAMERA_ID`。Windows 还需要在系统隐私设置中允许 Python 或终端访问摄像头。

### 浏览器无法访问摄像头

检查浏览器网站权限。录入页面使用浏览器的 `getUserMedia`，通常需要在 `localhost` 或 `127.0.0.1` 页面中运行并明确授予权限。

### 启动时报 `No module named facenet_pytorch` 或 `torch`

执行：

```bash
pip install torch torchvision facenet-pytorch
```

并确认 VS Code 使用的是已激活的 `face_rec` 解释器。

### 识别结果不稳定

改善光照和拍摄角度，录入清晰且正面的照片，并适当调整 `SIMILARITY_THRESHOLD`。当前每个人默认只使用一张基准照片。

### 修改人脸照片后结果没有变化

删除 `data/embeddings_cache/embeddings.pkl` 后重新启动，让系统根据人脸库照片重建缓存。

## 注意事项

- 本项目默认仅在本机运行，Flask 启动配置未包含登录认证和 HTTPS，不建议直接暴露到公网。
- 人脸图像和特征向量属于敏感生物识别信息，请在获得授权后采集，并妥善保护 `data/face_db/` 与缓存文件。
- 当前 `requirements.txt` 中保留了 `deepface`，但核心识别实现实际使用的是 `facenet_pytorch`；部署时请确保后者及其 PyTorch 依赖已安装。