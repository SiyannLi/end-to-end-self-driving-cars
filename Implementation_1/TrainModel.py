import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, Lambda
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle


def keras_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(40, 40, 1)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))

    # 使用 TensorFlow 2.x 的学习率调节方法
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mean_squared_error")
    filepath = "Autopilot.h5"
    checkpoint1 = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint1]

    return model, callbacks_list


def loadFromPickle():
    with open("features_40", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)  # 水平翻转
    labels = np.append(labels, -labels, axis=0)  # 反转标签
    return features, labels


def main():
    features, labels = loadFromPickle()
    features, labels = augmentData(features, labels)
    features, labels = shuffle(features, labels)

    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0, test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 40, 40, 1)
    test_x = test_x.reshape(test_x.shape[0], 40, 40, 1)

    # 构建模型
    model, callbacks_list = keras_model()

    # 训练模型
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=64, callbacks=callbacks_list)

    # 打印模型摘要
    model.summary()  # 直接调用 model.summary() 查看模型结构
    model.save('Autopilot.h5')


if __name__ == "__main__":
    main()
