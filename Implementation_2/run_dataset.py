import tensorflow as tf
import cv2
import numpy as np
import math
import os

# 加载已经保存的模型
model_path = "save/model.h5"  # 使用目录而非文件路径
model = tf.keras.models.load_model(model_path)

# 读取方向盘图像
steering_wheel_image_path = 'steering_wheel_image.jpg'
img = cv2.imread(steering_wheel_image_path, cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

smoothed_angle = 0
i = 0

# 定义一个函数用于处理图像
def preprocess_image(image_path):
    """读取并预处理图像"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} not found.")
        return None
    # 裁剪和调整图像大小
    image = cv2.resize(image[-150:], (200, 66)) / 255.0
    return np.expand_dims(image, axis=0)  # 添加批量维度

# 使用 cv2.VideoCapture() 读取图片（模拟视频流）
while True:
    # 读取输入图像并预处理
    image_path = f"driving_dataset/{i}.jpg"
    image = preprocess_image(image_path)
    if image is None:
        break  # 如果图像未找到，则退出循环

    # 进行预测
    degrees = model.predict(image)[0][0] * 180.0 / math.pi
    print("Predicted steering angle: {:.2f} degrees".format(degrees))

    # 显示驾驶图像
    full_image = cv2.imread(image_path)  # 读取全图用于显示
    cv2.imshow("frame", full_image)

    # 平滑角度过渡
    if abs(degrees - smoothed_angle) > 1e-6:  # 避免除以零的错误
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)

    # 旋转方向盘图像
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    # 显示方向盘旋转图像
    cv2.imshow("steering wheel", dst)

    # 检查退出条件
    if cv2.waitKey(10) == ord('q'):
        break

    i += 1

cv2.destroyAllWindows()
