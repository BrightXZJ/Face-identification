import os
import cv2

def capture_faces():
    # 1. 加载 OpenCV 自带的人脸检测器 (Haar Cascade)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("错误：无法加载人脸检测级联分类器！")
        return

    # 2. 调用摄像头接口 (通常 0 为内置摄像头，1 或 2 为外置摄像头)
    # 如果外置摄像头打不开，请尝试将 0 修改为 1 或 2
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"错误：无法打开摄像头设备 (ID: {camera_id})！")
        return

    # 设置分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建保存抓拍人脸图像的文件夹
    save_dir = "captured_faces"
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    print("=== 摄像头调用成功 ===")
    print("操作指南：")
    print("  - 按 's' 键：抓拍并保存当前截取的第一个人脸")
    print("  - 按 'q' 键：退出程序")

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            print("错误：无法获取视频帧！")
            break

        # 转为灰度图（加速人脸检测）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. 进行人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # 用于存储裁剪出的最新人脸区域
        current_face_crop = None

        # 4. 在图像上绘制人脸框
        for (x, y, w, h) in faces:
            # 绘制绿色矩形框 (BGR: 0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Target", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 裁剪出人脸区域（留作保存）
            current_face_crop = frame[y:y + h, x:x + w]

        # 5. 实时渲染画框后的画面
        cv2.imshow("Face Capture Module", frame)

        # 6. 监听键盘输入
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 按下 's' 键抓拍保存
            if current_face_crop is not None:
                count += 1
                save_path = os.path.join(save_dir, f"face_{count}.jpg")
                cv2.imwrite(save_path, current_face_crop)
                print(f"[提示] 已成功抓拍并保存人脸图片至: {save_path}")
            else:
                print("[警告] 未在当前画面中检测到有效人脸，抓拍失败！")

        elif key == ord('q'):
            # 按下 'q' 键退出
            print("正在退出系统...")
            break

    # 7. 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()