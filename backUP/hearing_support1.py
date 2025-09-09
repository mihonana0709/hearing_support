# 📦 必要なライブラリのインポート
import streamlit as st                    # Web UI構築ライブラリ
import sounddevice as sd                 # 音声入出力処理ライブラリ
import numpy as np                       # 数値処理ライブラリ
import queue                             # 音声バッファ管理用
import threading                         # 並列処理（録音と認識を分離）
import time                              # スリープ制御
from vosk import Model, KaldiRecognizer  # 軽量音声認識ライブラリ
import json                              # 認識結果のJSON処理

# 📁 音声認識モデルの読み込み
model_path = "model/vosk-model-small-ja-0.22"  # モデルフォルダのパス
model = Model(model_path)                      # モデルの読み込み
recognizer = KaldiRecognizer(model, 48000)     # サンプルレート指定（48kHz）

# 📦 音声バッファを保持するキュー（録音 → 認識へ渡す）
audio_queue = queue.Queue()

# 🧠 セッションステートの初期化（UIとスレッド間の状態共有）
if "latest_volume" not in st.session_state:
    st.session_state.latest_volume = 0.0        # 音量の最新値
# if "stop_flag" not in st.session_state:
#     st.session_state.stop_flag = threading.Event()  # 停止制御フラグ
# セッションステートの初期化（最初の run 時に一度だけ）
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = threading.Event()

if "thread_started" not in st.session_state:
    st.session_state.thread_started = False     # スレッド起動済みか
if "history" not in st.session_state:
    st.session_state.history = []               # 認識履歴

# 🔊 音量表示用プレースホルダ（毎回更新）
volume_placeholder = st.empty()

# 📝 認識結果表示用プレースホルダ
text_placeholder = st.empty()

# 🎧 音声入力コールバック（録音スレッド側）
def audio_callback(indata, frames, time_info, status):
    print("callbacktriggered")  # ← 追加：コールバックが呼ばれたことを確認するための出力
    volume = np.linalg.norm(indata)                     # 音量計算（ベクトルノルム）
    st.session_state.latest_volume = volume             # 音量をセッションに保存

    # mono_data = indata[:, 0].astype(np.int16)           # 左チャンネルのみ使用
    mono_data = indata.flatten().astype(np.int16)  # ← モノラルに変更
    audio_queue.put(mono_data)                          # 認識用にキューへ追加

    sd.play(indata, samplerate=48000)                   # イヤホン再生（任意）

# 🔁 音声認識処理（別スレッドで実行）
def recognize_worker():
    while not st.session_state.stop_flag.is_set():      # 停止フラグが立つまでループ
        try:
            data = audio_queue.get(timeout=1)           # 音声データを取得（1秒待機）
            if recognizer.AcceptWaveform(data.tobytes()):
                result = recognizer.Result()            # 認識結果（JSON文字列）
                text = json.loads(result)["text"]       # "text" フィールドを抽出
                st.session_state.history.append(text)   # 履歴に追加
        except queue.Empty:
            continue                                    # キューが空ならスキップ

# 🎤 音声入力ストリーム（録音スレッド）
def start_stream():
    try:
        with sd.InputStream(
            samplerate=48000,
            # channels=2,
            channels=1,  # ← モノラルに変更
            device=12,  # ← ここを追加
            callback=audio_callback
        ):
            while not st.session_state.stop_flag.is_set():
                time.sleep(0.1)                         # CPU負荷軽減のため待機
    except Exception as e:
        st.session_state.stop_flag.set()                # エラー時は停止
        st.error(f"⚠️ 音声入力エラー: {e}")

# 🖥️ Streamlit UI構築
st.title("🎙️ リアルタイム音声認識アプリ")
st.write("マイク音声をリアルタイムで文字起こしします。")

# ▶️ 音声認識開始ボタン
if st.button("🎙️ 音声認識を開始"):
    st.session_state.stop_flag.clear()                  # 停止フラグを解除
    if not st.session_state.thread_started:
        threading.Thread(target=recognize_worker, daemon=True).start()
        threading.Thread(target=start_stream, daemon=True).start()
        st.session_state.thread_started = True
    st.write("🎧 録音中... 話しかけてください")

# 🔊 音量バーの描画（UIスレッド側）
volume = st.session_state.latest_volume
volume_placeholder.progress(min(int(volume * 100), 100))  # 音量バー更新　# ← 10 → 100 に変更10、100
st.write(f"🔊 音量: {volume:.2f}")                        # 数値表示

# 📝 認識結果の表示
if st.session_state.history:
    latest_text = st.session_state.history[-1]
    text_placeholder.markdown(f"📝 **認識結果**：{latest_text}")

    with st.expander("🗂️ 認識履歴", expanded=False):
        for i, line in enumerate(st.session_state.history[::-1]):
            st.write(f"{len(st.session_state.history)-i}: {line}")

# ⏹️ 停止ボタン
if st.button("🛑 音声認識を停止"):
    st.session_state.stop_flag.set()                    # 停止フラグをセット
    st.session_state.thread_started = False             # スレッド状態をリセット
    st.write("🛑 音声認識を停止しました。")