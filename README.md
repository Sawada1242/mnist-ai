# 手書き数字認識AI（MNIST）

## 概要
TensorFlowを用いて、手書き数字（0〜9）を分類するニューラルネットワークを実装しました。  
学習から予測、結果の可視化までを行っています。

## 使用技術
- Python
- TensorFlow / Keras
- Matplotlib

## モデル構成
- 入力層：28×28画像（Flatten）
- 中間層：128ノード（ReLU）
- 出力層：10クラス（Softmax）

## 結果
テストデータに対して約97%の精度を達成しました。

## 実行方法
```bash
python mnist.py
