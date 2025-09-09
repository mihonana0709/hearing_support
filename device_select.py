# 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import threading
import queue
import vosk
import json
import os
import logging

# ロガーの設定
logging.basicConfig(
    filename='hearing_support.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Voskモデルのパスを指定
MODEL_PATH = "vosk-model-small-ja-0.22"

# 音声増幅のためのゲイン（調整可能）
GAIN_FACTOR = 4.0  # ★ ここを2.0から4.0に増やしました ★

# セッション状態の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "current_transcription" not in st.session_state:
    st.session_state.current_transcription = ""
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "processing_thread" not in st.session_state:
    st.session_state.processing_thread = None

# 確実にキューを初期化する
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "transcription_queue" not in st.session_state:
    st.session_state.transcription_queue = queue.Queue()
if "volume_queue" not in st.session_state:
    st.session_state.volume_queue = queue.Queue()

# Voskモデルの読み込み関数 (キャッシュ)
@st.cache_resource
def load_vosk_model():
    """Voskモデルをキャッシュして読み込みます。"""
    if not os.path.exists(MODEL_PATH) or not os.path.isdir(MODEL_PATH):
        st.error(f"❌ Voskモデルが見つかりません。パス: {os.path.abspath(MODEL_PATH)}")
        st.info("モデルフォルダ（`vosk-model-small-ja-0.22`）をこのスクリプトと同じ場所に配置してください。")
        return None
    try:
        model = vosk.Model(MODEL_PATH)
        logger.info("LOG: Voskモデルの読み込みに成功しました。")
        return model
    except Exception as e:
        st.error(f"❌ モデル読み込み失敗: {e}")
        logger.error(f"LOG: Voskモデルの読み込みに失敗しました: {e}")
        return None

# Voskの音声認識処理を別スレッドで行うクラス
class AudioProcessingThread(threading.Thread):
    def __init__(self, audio_queue, transcription_queue, volume_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.transcription_queue = transcription_queue
        self.volume_queue = volume_queue
        self.stop_event = threading.Event()
        self.recognizer = vosk.KaldiRecognizer(load_vosk_model(), 16000)
        logger.info("LOG: AudioProcessingThread 初期化完了。")

    def run(self):
        logger.info("LOG: AudioProcessingThread 開始。")
        while not self.stop_event.is_set():
            try:
                frame = self.audio_queue.get(timeout=1)
                
                audio_data = frame.to_ndarray()
                
                # ここでゲインを適用して音声データを増幅
                amplified_data = audio_data * GAIN_FACTOR
                # オーディオデータをint16型に変換
                audio_data_int16 = amplified_data.astype(np.int16)
                
                # RMS計算の修正
                if audio_data_int16.size > 0:
                    audio_data_float = audio_data_int16.astype(np.float64)
                    rms = np.sqrt(np.mean(np.square(audio_data_float)))
                    self.volume_queue.put(rms)
                    logger.info(f"LOG: 音声データ取得、RMS={rms:.2f}")

                # Voskで音声認識
                if self.recognizer.AcceptWaveform(audio_data_int16.tobytes()):
                    result = self.recognizer.Result()
                    text = json.loads(result)["text"]
                    if text.strip():
                        self.transcription_queue.put(text)
                        logger.info(f"LOG: 認識結果 -> {text}")
                else:
                    partial_result = self.recognizer.PartialResult()
                    partial_text = json.loads(partial_result)["partial"]
                    if partial_text.strip():
                        # 部分的な結果もログに出力
                        logger.info(f"LOG: 認識中 -> {partial_text}")
                        self.transcription_queue.put(f"partial:{partial_text}")

            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
        logger.info("LOG: AudioProcessingThread 停止信号受信。")

# AudioProcessorBaseを継承したWebRTC用のクラス
class WebRtcAudioProcessor(AudioProcessorBase):
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        logger.info("LOG: WebRtcAudioProcessor 初期化完了。")

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_queue.put(frame)
        logger.info("LOG: WebRTCフレームをキューに格納。")
        return frame

# ---
### アプリケーションのメインUI ###
st.title("🎤 聴覚サポートアプリ")
st.write("マイク音声をリアルタイムで増幅・文字起こしし、イヤホンから再生します。")

# webrtc_ctxの初期化を try-except で囲む
try:
    processor = WebRtcAudioProcessor(st.session_state.audio_queue)
    logger.info("LOG: webrtc_streamerの呼び出し前")

    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: processor,
        media_stream_constraints={
            "audio": {
                "autoGainControl": True,
                "echoCancellation": True,
                "noiseSuppression": True,
            },
            "video": False
        },
    )

    logger.info("LOG: webrtc_streamerの呼び出し後")

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.stop()


# ストリームの状態表示
status_placeholder = st.empty()
if webrtc_ctx.state.playing:
    status_placeholder.info("🎧 音声認識中...話しかけてください")
    logger.info("LOG: webrtc_ctx.state.playingがTrueです")
else:
    status_placeholder.info("🛑 停止中")
    logger.info("LOG: webrtc_ctx.state.playingがFalseです")

# 音声認識スレッドを開始/停止
if webrtc_ctx.state.playing and st.session_state.processing_thread is None:
    st.session_state.processing_thread = AudioProcessingThread(
        st.session_state.audio_queue, 
        st.session_state.transcription_queue, 
        st.session_state.volume_queue
    )
    st.session_state.processing_thread.start()
    logger.info("LOG: 音声処理スレッドを起動しました")
elif not webrtc_ctx.state.playing and st.session_state.processing_thread is not None:
    # 停止ボタンが押されたとき、またはアプリが終了するとき
    st.session_state.processing_thread.stop()
    st.session_state.processing_thread.join()
    st.session_state.processing_thread = None
    logger.info("LOG: 音声処理スレッドを停止しました")

# ---
### 📊 リアルタイム音量レベル

with st.sidebar:
    st.markdown("### 📊 リアルタイム音量レベル")

    while not st.session_state.volume_queue.empty():
        st.session_state.volume_history.append(st.session_state.volume_queue.get())
        if len(st.session_state.volume_history) > 100:
            st.session_state.volume_history.pop(0)

    if webrtc_ctx.state.playing:
        st.line_chart(st.session_state.volume_history)
    else:
        st.info("マイク入力を待機中です...")


# ---
### 📝 認識結果

result_placeholder = st.empty()

while not st.session_state.transcription_queue.empty():
    text = st.session_state.transcription_queue.get()
    if text.startswith("partial:"):
        st.session_state.current_transcription = text.replace("partial:", "")
    else:
        st.session_state.history.append(text)
        st.session_state.current_transcription = ""

if st.session_state.current_transcription:
    result_placeholder.write(f"**（認識中）** {st.session_state.current_transcription}")
elif st.session_state.history:
    result_placeholder.write(st.session_state.history[-1])
else:
    result_placeholder.info("ここに文字起こし結果が表示されます。")

# ---
### 📋 履歴

if st.session_state.history:
    for line in reversed(st.session_state.history):
        st.write(f"- {line}")