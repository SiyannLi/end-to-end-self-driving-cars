import os
import tensorflow as tf
import driving_data
import model

LOGDIR = './save'

# 定义超参数
L2NormConst = 0.001
epochs = 30
batch_size = 100
learning_rate = 1e-4

# 创建模型
model = model.create_model()  # 假设 model.py 中有一个 create_model 函数
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# 创建 TensorBoard 日志目录
logs_path = './logs'
summary_writer = tf.summary.create_file_writer(logs_path)

# 确保保存目录存在
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# 训练模型
for epoch in range(epochs):
    for i in range(int(driving_data.num_images / batch_size)):
        xs, ys = driving_data.LoadTrainBatch(batch_size)

        # 计算梯度并更新权重
        with tf.GradientTape() as tape:
            y_pred = model(xs, training=True)
            # 计算损失
            loss = loss_fn(ys, y_pred) + L2NormConst * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 每10步验证一次
        if i % 10 == 0:
            xs_val, ys_val = driving_data.LoadValBatch(batch_size)
            val_loss = loss_fn(ys_val, model(xs_val, training=False))
            print("Epoch: %d, Step: %d, Validation Loss: %g" % (epoch, epoch * (driving_data.num_images // batch_size) + i, val_loss.numpy()))

        # 写入 TensorBoard 日志
        with summary_writer.as_default():
            step = int(epoch * (driving_data.num_images // batch_size) + i)  # 确保step为整数
            tf.summary.scalar("loss", loss, step=step)

        # # 保存模型
        # if i % 10 == 0:  # 每10步保存一次模型
        #     checkpoint_path = os.path.join(LOGDIR, "model_epoch_{}_step_{}.h5".format(epoch, i))
        #     model.save(checkpoint_path)  # 保存整个模型到指定路径

model_path = "save/model.h5"  # 确保这是保存模型的正确路径
model.save(model_path)  # 保存整个模型到指定路径
print("Model training complete.")

