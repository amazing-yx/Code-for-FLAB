#   -*- coding = utf-8 -*-
#   @time : 2021/6/14 14:28
#   @ File : mnist_and_block_easy_test.py
#   @Software: PyCharm
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from Block import Block
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape, y.shape)

batchsz = 128

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

# db_iter = iter(db)
# sample = next(db_iter)
# print('batch:', sample[0].shape, sample[1].shape)


model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
    layers.Dense(10)  # [b, 32] => [b, 10], 330 = 32*10 + 10
])
model.build(input_shape=[None, 28*28])
model.summary()
# w = w - lr * grad
optimizer = optimizers.Adam(lr=1e-3)


def main():
    blockchains1 = [Block.create_genesis_block()]
    blockchains2 = [Block.create_genesis_block()]
    i = 1
    j = 1
    for epoch in range(30):

        for step, (x, y) in enumerate(db):


            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                # [b]
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                # print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))
                transaction1 = str(epoch) + str(step) + str(float(loss_ce)) + str(float(loss_mse))
                transaction1_biaoda = str('第' + str(epoch+1) + '遍' + '第' + str(step+100) + '步' + '交叉熵损失率：' + str(float(loss_ce)))
                blockchains1.append(Block(blockchains1[i - 1].hash, transaction1, datetime.datetime.now()))
                print("链1，区块{},交易的内容包括了模型学习的{}".format(i, transaction1_biaoda))
                print("该区块的哈希是", blockchains1[i].hash)
                i = i + 1

        # test
        total_correct = 0
        total_num = 0
        for x, y in db_test:

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            # [b, 10]
            logits = model(x)
            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        # print(epoch, 'test acc:', acc)
        transaction2 = str(epoch) + str(acc)
        blockchains2.append(Block(blockchains2[j - 1].hash, transaction2, datetime.datetime.now()))
        transaction2_biaoda = str(str(epoch + 1) + '遍后测试的准确率：' + str(acc))
        print("链2，区块{},交易内容包含，训练了第{}".format(j, transaction2_biaoda))
        print("区块的哈希是", blockchains2[j].hash)
        j = j + 1
    print(blockchains1[:3])
    print(blockchains2[:3])



if __name__ == '__main__':
    main()
