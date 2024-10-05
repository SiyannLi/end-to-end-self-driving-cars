import tensorflow as tf
import cv2
import numpy as np
import model  # 确保这里是指向你的模型定义的正确文件

# 加载模型
model_path = "save/model.h5"  # 确保这是保存模型的正确路径
model = tf.keras.models.load_model(model_path)

img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

# 打开摄像头
cap = cv2.VideoCapture(0)
while cv2.waitKey(10) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 调整图像大小并归一化
    image = cv2.resize(frame, (200, 66)) / 255.0
    image = np.expand_dims(image, axis=0)  # 增加一个维度以匹配模型输入

    # 使用模型进行预测
    degrees = model.predict(image)[0][0] * 180 / np.pi  # 假设输出为[0][0]

    # 清屏
    print("\033c", end="")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow('frame', frame)

    # 平滑角度变化
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

cap.release()
cv2.destroyAllWindows()
