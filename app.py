# app.py
import cv2
from flask import Flask, Response, render_template
from core.face_engine import FaceEngine
from core.db_manager import DatabaseManager
from core.face_matcher import FaceMatcher
from core.camera_handler import CameraHandler
from utils.image_utils import draw_chinese_text
from config import FRAME_SKIP

app = Flask(__name__)

# 初始化所有模块
print("🔧 初始化系统模块...")
db = DatabaseManager()
matcher = FaceMatcher(db)
camera = CameraHandler()

frame_count = 0

def generate_video_frames():
    """
    视频流生成器：逐帧检测人脸、比对、绘制姓名并输出
    """
    global frame_count
    engine = FaceEngine()  # 每个线程独立实例
    
    while True:
        success, frame = camera.get_frame()
        if not success:
            break
        
        frame_count += 1
        
        # 每隔 FRAME_SKIP 帧做一次推理（提高性能）
        if frame_count % FRAME_SKIP == 0:
            # 检测人脸
            faces, boxes, probs = engine.detect_faces(frame)
            
            # 对每张人脸进行比对
            for face, box, prob in zip(faces, boxes, probs):
                if prob < 0.7:  # 置信度太低则跳过
                    continue
                
                # 提取特征
                embedding = engine.get_embedding(face)
                if embedding is None:
                    continue
                
                # 匹配
                name, score = matcher.match(embedding)
                
                # 绘制人脸框和姓名（根据是否匹配成功选择颜色）
                x1, y1, x2, y2 = box
                if name != "Unknown":
                    color = (0, 255, 0)  # 绿色
                    label = f"{name} ({score:.2f})"
                else:
                    color = (0, 0, 255)  # 红色
                    label = f"Unknown ({score:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 使用中文绘制函数（如果绘制失败则回退到英文）
                try:
                    frame = draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=color)
                except:
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 编码为 JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """主页：显示视频流"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流接口"""
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 启动 Flask 服务...")
    print("📱 请在浏览器中访问: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)