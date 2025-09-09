# # 必要なライブラリをインポート
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle

# # 1. 保存されたモデルを読み込む
# # 'rb'は読み込みモードを意味します（binary read mode）
# with open('models/model_iris.pkl', 'rb') as f:
#     clf = pickle.load(f)


# # 2. サイドバーにスライダーを作成してユーザーから入力を受け取る
# st.sidebar.header('Input Features')

# # ユーザーがスライダーで花の特徴を入力
# sepal_length = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
# sepal_width = 0.0  # 幅は仮の値（0.0）を設定
# petal_length = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
# petal_width = 0.0  # 幅は仮の値（0.0）を設定


# # 3. メインパネルに入力された値を表示
# st.title('Iris Classifier')
# st.write('## Input Value')

# # 4. 入力値をDataFrameに変換
# # 入力されたデータを使ってモデルに渡すための形式に変換
# value_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
#                         columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

# # 5. 入力された値を表示
# st.write(value_df)


# # 6. モデルを使って予測を行う
# # predict_probaはそれぞれの花の種類の確率を返す
# pred_probs = clf.predict_proba(value_df)
# pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicolor', 'virginica'], index=['probability'])

# # 7. 予測結果を表示
# st.write('## Prediction')
# st.write(pred_df)

# # 8. 予測結果を使って最も可能性の高い花の種類を表示
# name = pred_df.idxmax(axis=1).tolist()  # 確率が最も高い花の種類を取得
# st.write('## Result')
# st.write('このアイリスはきっと', str(name[0]), 'です!')

import streamlit as st
import pandas as pd
import requests

st.title('Iris Classifier')

st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.sidebar.slider('sepal width (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.sidebar.slider('petal width (cm)', min_value=0.0, max_value=10.0, step=0.1)

iris = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}

targets = ['setosa', 'versicolor', 'virginica']

if st.sidebar.button("Predict"):
    # 入力された説明変数の表示
    st.write('## Input Value')
    iris_df = pd.DataFrame(iris, index=["data"])
    st.write(iris_df)

    # 予測の実行
    response = requests.post("http://localhost:8000/predict", json=iris)
    prediction = response.json()["prediction"]

    # 予測結果の表示
    st.write('## Prediction')
    st.write(prediction)

    # 予測結果の出力
    st.write('## Result')
    st.write('このアイリスはきっと',str(targets[int(prediction)]),'です!')


    