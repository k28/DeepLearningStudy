# coding:utf-8

#
# Kerasでゼロから作るDeep LearningのCNNを作成する
#

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import pandas as pd
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 10
epochs = 20

# input image demensions
img_rows, img_cols = 28, 28

# the data, split between tran and sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 画像のチャンネルが先か後かでデータをreshapeする
print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0],   1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test  = x_test.reshape(x_test.shape[0],   img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print(x_train.shape)
#(60000, 28, 28, 1)が出力される

# 学習データを0-1の範囲に正規化する
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convertclass vectors to binary matrices
# one_hot_label に変換する処理
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

#
# ここからモデルの構築
# Conv-ReLU-Pooling -- Affine-ReLU -- Affine-Softmax
#
model = Sequential()
model.add(Conv2D(30, kernel_size=(5, 5), strides=1,
    activation='relu',
    input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 過学習を防げるか? Dropoutを追加
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

# Softmaxの前にもDropoutを追加する
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

# 損失関数 交差エントロピー誤差平均
# 最適化関数 Adadelta
model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])

# 学習を行う
history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))

# plot learning history
plt.style.use("ggplot")
df = pd.DataFrame(history.history)
df.index += 1
df.index.name = "epoch"
df[["acc", "val_acc"]].plot(linewidth=2)
plt.savefig("plt_acc_history.png")

df[["loss", "val_loss"]].plot(linewidth=2)
plt.savefig("plt_loss_history.png")

score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


