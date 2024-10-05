import numpy as np  # 添加这一行
import random
from PIL import Image  # 使用 PIL

xs = []
ys = []

# points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        # the paper by Nvidia uses the inverse of the turning radius,
        # but steering wheel angle is proportional to the inverse of turning radius
        # so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * np.pi / 180)  # 使用 np.pi

# get number of images
num_images = len(xs)

# shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        # 使用 PIL 打开图像并调整大小
        img = Image.open(train_xs[(train_batch_pointer + i) % num_train_images])
        img = img.crop((0, img.size[1] - 150, img.size[0], img.size[1]))  # 裁剪底部 150 像素
        img = img.resize((200, 66))  # 调整大小
        x_out.append(np.array(img) / 255.0)  # 归一化
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    # 转换为 numpy 数组并调整形状
    x_out = np.array(x_out)  # (batch_size, 66, 200, 3)
    y_out = np.array(y_out)  # (batch_size, 1)
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        # 使用 PIL 打开图像并调整大小
        img = Image.open(val_xs[(val_batch_pointer + i) % num_val_images])
        img = img.crop((0, img.size[1] - 150, img.size[0], img.size[1]))  # 裁剪底部 150 像素
        img = img.resize((200, 66))  # 调整大小
        x_out.append(np.array(img) / 255.0)  # 归一化
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])

    # 转换为 numpy 数组并调整形状
    x_out = np.array(x_out)  # (batch_size, 66, 200, 3)
    y_out = np.array(y_out)  # (batch_size, 1)
    val_batch_pointer += batch_size
    return x_out, y_out

