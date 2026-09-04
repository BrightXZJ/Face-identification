# core/camera_handler.py
import cv2
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT

class CameraHandler:
    """
    摄像头处理器：采集视频流
    """
    def __init__(self, camera_id=None, width=None, height=None):
        """
        初始化摄像头
        参数:
            camera_id: 摄像头ID，默认从 config 读取
            width: 采集宽度
            height: 采集高度
        """
        self.camera_id = camera_id if camera_id is not None else CAMERA_ID
        self.width = width if width is not None else FRAME_WIDTH
        self.height = height if height is not None else FRAME_HEIGHT
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 ID: {self.camera_id}，请检查设备连接")
        
        # 设置分辨率（某些摄像头不支持自定义，会忽略）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        print(f"✅ 摄像头已打开: ID={self.camera_id}, 分辨率={self.width}x{self.height}")
    
    def get_frame(self):
        """
        读取一帧图像
        返回:
            success: bool, 是否成功
            frame: numpy.ndarray 或 None
        """
        success, frame = self.cap.read()
        if not success:
            return False, None
        return True, frame
    
    def release(self):
        """
        释放摄像头资源
        """
        if self.cap is not None:
            self.cap.release()
            print("✅ 摄像头已释放")
    
    def generate_frames(self):
        """
        生成器函数：不断产生 JPEG 编码的视频流帧
        用于 Flask 的 /video_feed 路由
        """
        while True:
            success, frame = self.get_frame()
            if not success:
                break
            
            # 将帧编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            # 按照 MJPEG 流格式返回
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + b'\r\n')
    
    def __del__(self):
        """析构函数：确保资源释放"""
        self.release()