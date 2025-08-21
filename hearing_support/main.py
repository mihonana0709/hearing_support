# 必要なライブラリをインポート
import streamlit as st
import numpy as np
import pandas as pd

# タイトルとテキストを記入
st.title('Streamlit 基礎')
st.write('Hello World!')

# データフレームの準備
df = pd.DataFrame({
    '1列目' : [1, 2, 3, 4],
    '2列目' : [10, 20, 30, 40]
})

# 動的なテーブル
st.dataframe(df)

# 引数を使用した動的テーブル
st.dataframe(df.style.highlight_max(axis = 0) , width = 100 , height = 150)

# 静的なテーブル
st.table(df)

# 10 行 3 列のデータフレームを準備
df = pd.DataFrame(
    np.random.rand(10,3),
    columns = ['a', 'b', 'c']
)

#print(df)  # ← これでターミナルに表示されます

# 折れ線グラフ
st.line_chart(df)

# 面グラフ
st.area_chart(df)

# 棒グラフ
st.bar_chart(df)

# プロットする乱数をデータフレームで用意
df = pd.DataFrame(

    # 乱数 + 新宿の緯度と経度
    np.random.rand(100,2) / [50, 50] + [35.69, 139.70],
    columns = ['lat', 'lon']
)

# マップをプロット
st.map(df)

# Pillow
from PIL import Image

# 画像の読み込み
img = Image.open('iris.jpg')
st.image(img,caption = 'Iris' , use_column_width = True)

# チェックボックス
if st.checkbox('Show Image'):
    img = Image.open('iris.jpg')
    st.image(img,caption = 'Iris' , use_column_width = True)

# セレクトボックス
option = st.selectbox(
    '好きな数字を入力してください。',
    list(range(1, 11))
)

'あなたの好きな数字は' , option , 'です。'


# テキスト入力による値の動的変更
text = st.sidebar.text_input('あなたの好きなスポーツを教えて下さい。')

'あなたの好きなスポーツ：' , text


# スライダーによる値の動的変更
condition = st.sidebar.slider('あなたの今の調子は？', 0, 100, 50)

'コンディション：' , condition

# expander
expander1 = st.expander('質問1')
expander1.write('質問1の回答')
expander2 = st.expander('質問2')
expander2.write('質問2の回答')
expander3 = st.expander('質問3')
expander3.write('質問3の回答')


import time

latest_iteration = st.empty()
bar = st.progress(0)

# プログレスバーを0.1秒毎に進める
for i in range(100):
    latest_iteration.text(f'Iteration{i + 1}')
    bar.progress(i + 1)
    time.sleep(0.1)

'Done'



