# register_face.py
import cv2
import os
import shutil
import numpy as np
from core.face_engine import FaceEngine
from config import FACE_DB_DIR

class FaceRegistrar:
    def __init__(self):
        self.engine = FaceEngine()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
            exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.name = ""
        self.running = True

    def save_face(self, frame, name):
        faces, boxes, probs = self.engine.detect_faces(frame)
        if not faces:
            print("⚠️ No face detected")
            return False

        x1, y1, x2, y2 = boxes[0]
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            print("⚠️ Crop failed")
            return False

        # 目标路径
        person_dir = os.path.join(FACE_DB_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        save_path = os.path.join(person_dir, "base.jpg")

        # ---------- 先保存到桌面临时文件 ----------
        temp_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        temp_path = os.path.join(temp_dir, "temp_face.jpg")
        success = cv2.imwrite(temp_path, face_crop)
        if not success:
            print(f"❌ Failed to write temp file: {temp_path}")
            return False
        print(f"✅ Temp file saved: {temp_path}")

        # ---------- 复制到目标路径 ----------
        try:
            shutil.copy2(temp_path, save_path)
            os.remove(temp_path)   # 删除临时文件
        except Exception as e:
            print(f"❌ Failed to copy to target: {e}")
            print(f"   Please manually copy {temp_path} to {save_path}")
            return False

        # ---------- 验证 ----------
        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"✅ Saved: {save_path} ({size} bytes)")
            return True
        else:
            print(f"❌ File not found after copy: {save_path}")
            return False

    def run(self):
        print("\n" + "="*50)
        print("📸 Face Database Registration Tool")
        print("="*50)
        self.name = input("Enter person's name (e.g., ZhangSan): ").strip()
        if not self.name:
            print("❌ Name cannot be empty")
            return
        # 名字中去掉空格，避免路径问题
        self.name = self.name.replace(" ", "_")
        print(f"\nRegistering: {self.name}")
        print("Instructions: [SPACE] capture, [q] quit")
        print("="*50 + "\n")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            faces, boxes, probs = self.engine.detect_faces(frame)
            display = frame.copy()
            if faces:
                for (x1,y1,x2,y2), prob in zip(boxes, probs):
                    cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(display, f"{prob:.2f}", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(display, f"Register: {self.name} [SPACE]", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(display, "No face", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Registration - SPACE to save, q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if faces:
                    if self.save_face(frame, self.name):
                        print("✅ Capture successful. Press q to exit or continue.")
                        cv2.waitKey(500)
                else:
                    print("⚠️ No face")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    registrar = FaceRegistrar()
    registrar.run()