# 必要なライブラリをインポート
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# 1. Irisデータセットをロード
iris = datasets.load_iris()

# 2. データをDataFrameに変換
# 'data'は花の特徴（がく片の長さ・幅、花びらの長さ・幅）、'target'は花の種類
features = pd.DataFrame(iris['data'], columns=iris['feature_names'])  # 花の特徴データ
target = iris['target']  # 花の種類

# 3. モデルを作成
# RandomForestClassifierは複数の決定木を使った分類アルゴリズム
model = RandomForestClassifier()

# 4. モデルにデータを与えて学習させる
model.fit(features, target)

# 5. 学習したモデルをファイルに保存
# 'wb'は書き込みモードを意味します（binary write mode）
with open('models/model_iris.pkl', 'wb') as f:
    pickle.dump(model, f)