from tensorflow import keras
from PIL import Image
import numpy as np

# モデル読み込み
model = keras.models.load_model("mnist_model.keras")

# 画像読み込み
img = Image.open("test.png").convert("L")
img = img.resize((28, 28))

# 配列化
img = np.array(img)

# 白黒反転（MNISTに合わせる）
img = 255 - img

# 正規化
img = img / 255.0

# 形状変更
img = img.reshape(1, 28, 28)

# 予測
pred = model.predict(img)

print("予測結果:", np.argmax(pred))

import matplotlib.pyplot as plt

plt.imshow(img.reshape(28,28), cmap='gray')
plt.title("AIが見ている画像")
plt.show()