from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 学習済みのモデルの読み込み
with open('models/model_iris.pkl', 'rb') as f:
    model = pickle.load(f)

# トップページ
@app.get('/')
async def index():
    return {"Iris": 'iris_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/predict')
async def predict(features: iris):
    return({'prediction':str(model.predict([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])[0])})