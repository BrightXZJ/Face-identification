# app.py
import cv2
import os
import shutil
from flask import Flask, Response, render_template, request, jsonify
from werkzeug.utils import secure_filename

from core.face_engine import FaceEngine
from core.db_manager import DatabaseManager
from core.face_matcher import FaceMatcher
from core.camera_handler import CameraHandler
from utils.image_utils import draw_chinese_text
from config import FRAME_SKIP, FACE_DB_DIR

app = Flask(__name__)

# 初始化全局模块（仅一次）
print("🔧 初始化系统模块...")
db = DatabaseManager()
matcher = FaceMatcher(db)
camera = CameraHandler()
engine = FaceEngine()  # 用于视频流和添加人脸时的特征提取

# ---------- 页面路由 ----------
@app.route('/')
def index():
    """主页：功能导航"""
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    """人脸识别页面"""
    return render_template('recognition.html')

@app.route('/add_face')
def add_face():
    """添加人脸页面"""
    return render_template('add_face.html')

@app.route('/manage')
def manage():
    """人脸库管理页面"""
    return render_template('manage.html')

# ---------- 视频流 ----------
def generate_video_frames():
    """视频流生成器（带人脸识别标注）"""
    global engine
    frame_count = 0
    while True:
        success, frame = camera.get_frame()
        if not success:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            faces, boxes, probs = engine.detect_faces(frame)
            for face, box, prob in zip(faces, boxes, probs):
                if prob < 0.7:
                    continue
                embedding = engine.get_embedding(face)
                if embedding is None:
                    continue
                name, score = matcher.match(embedding)
                x1, y1, x2, y2 = box
                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"{name} ({score:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                try:
                    frame = draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=color)
                except:
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- API 接口 ----------
@app.route('/api/persons', methods=['GET'])
def get_persons():
    """获取所有人员姓名列表"""
    names = db.get_all_names()
    return jsonify({'persons': names})

@app.route('/api/add_person', methods=['POST'])
def api_add_person():
    """添加人员：接收 name 和图片文件"""
    name = request.form.get('name')
    if not name:
        return jsonify({'success': False, 'msg': '姓名不能为空'}), 400
    # 清理姓名（防止路径注入）
    name = secure_filename(name.strip())
    if not name:
        return jsonify({'success': False, 'msg': '姓名包含非法字符'}), 400

    # 检查是否已存在
    if name in db.get_all_names():
        return jsonify({'success': False, 'msg': f'人员 {name} 已存在'}), 400

    # 获取上传的图片
    file = request.files.get('image')
    if not file:
        return jsonify({'success': False, 'msg': '未上传图片'}), 400

    # 保存临时文件
    temp_dir = os.path.join(os.path.dirname(__file__), 'data', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f'{name}.jpg')
    file.save(temp_path)

    # 调用数据库管理器添加
    success = db.add_person(name, temp_path)
    os.remove(temp_path)  # 删除临时文件

    if success:
        return jsonify({'success': True, 'msg': f'添加 {name} 成功'})
    else:
        return jsonify({'success': False, 'msg': f'添加 {name} 失败（无法检测到人脸或图片无效）'}), 400

@app.route('/api/delete_person', methods=['POST'])
def api_delete_person():
    """删除人员"""
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'msg': '缺少姓名'}), 400
    success = db.delete_person(name)
    if success:
        return jsonify({'success': True, 'msg': f'已删除 {name}'})
    else:
        return jsonify({'success': False, 'msg': f'未找到 {name}'}), 404

if __name__ == '__main__':
    print("🚀 启动 Flask 服务...")
    print("📱 请在浏览器中访问: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)