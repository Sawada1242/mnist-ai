import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# データ読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

# モデル作成
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# コンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習
model.fit(x_train, y_train, epochs=5)
# 学習後
model.save("mnist_model.keras")
# 評価
model.evaluate(x_test, y_test)

# ===== ここから追加 =====

# 🔹 学習データ表示（5枚）
for i in range(5):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"正解ラベル: {y_train[i]}")
    plt.show()

# 🔹 予測
pred = model.predict(x_test)

# 🔹 テストデータ＋予測表示（5枚）
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"予測: {np.argmax(pred[i])} / 正解: {y_test[i]}")
    plt.show()