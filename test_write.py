# test_write.py
import cv2
import os

# 测试用图片（纯黑）
test_img = np.zeros((100,100,3), dtype=np.uint8)

# 保存到桌面
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
test_path = os.path.join(desktop, "test_write.jpg")
success = cv2.imwrite(test_path, test_img)
print(f"写入桌面: {success}, 路径: {test_path}")

# 保存到 data/face_db/ 下（模拟）
os.makedirs("data/face_db/test", exist_ok=True)
test_path2 = "data/face_db/test/test.jpg"
success2 = cv2.imwrite(test_path2, test_img)
print(f"写入 data/face_db/test/: {success2}, 路径: {test_path2}")