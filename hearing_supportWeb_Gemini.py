# 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import vosk
import json
import os
import av
import threading
import numpy as np
import time
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

# セッション状態の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "current_transcription" not in st.session_state:
    st.session_state.current_transcription = ""
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "vosk_model_loaded" not in st.session_state:
    st.session_state.vosk_model_loaded = False
if "recognizer" not in st.session_state:
    st.session_state.recognizer = None

# Voskモデルの読み込み関数
@st.cache_resource
def load_vosk_model():
    """Voskモデルをキャッシュして読み込みます。"""
    if not os.path.exists(MODEL_PATH) or not os.path.isdir(MODEL_PATH):
        st.error(f"❌ Voskモデルが見つかりません。パス: {os.path.abspath(MODEL_PATH)}")
        st.info("モデルフォルダ（`vosk-model-small-ja-0.22`）をこのスクリプトと同じ場所に配置してください。")
        logger.error(f"LOG: Voskモデルのパスが見つかりません: {os.path.abspath(MODEL_PATH)}")
        st.session_state.vosk_model_loaded = False
        return None
    try:
        model = vosk.Model(MODEL_PATH)
        logger.info("LOG: Voskモデルの読み込みに成功しました。")
        st.session_state.vosk_model_loaded = True
        return model
    except Exception as e:
        st.error(f"❌ モデル読み込み失敗: {e}")
        logger.error(f"LOG: Voskモデルの読み込みに失敗しました: {e}")
        st.session_state.vosk_model_loaded = False
        return None

# AudioProcessorBaseを継承したカスタムクラス
class VoskAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.transcription_queue = []
        self.volume_history_queue = []
        
        # Vosk認識エンジンの初期化
        vosk_model = st.session_state.get("vosk_model", None)
        if vosk_model:
            self.recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
            logger.info("LOG: Vosk認識エンジンが初期化されました。")
        else:
            self.recognizer = None
            logger.warning("LOG: Voskモデルが未ロードのため、認識エンジンを初期化できません。")

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """WebRTCから受信したオーディオフレームを処理します。"""
        # 音声データをNumpy配列に変換
        audio_data = frame.to_ndarray(format="s16le")

        # 音量レベルを計算し、キューに追加
        rms = np.sqrt(np.mean(np.square(audio_data)))
        with self.lock:
            self.volume_history_queue.append(rms)

        # Voskで音声認識（処理を軽量化）
        if self.recognizer and len(self.volume_history_queue) % 5 == 0:  # 5フレームに1回だけ処理
            if self.recognizer.AcceptWaveform(audio_data.tobytes()):
                result = self.recognizer.Result()
                text = json.loads(result)["text"]
                if text.strip():
                    with self.lock:
                        self.transcription_queue.append(text)
            else:
                partial_result = self.recognizer.PartialResult()
                partial_text = json.loads(partial_result)["partial"]
                if partial_text.strip():
                    with self.lock:
                        self.transcription_queue.append(f"partial:{partial_text}")
        
        return frame

# webrtc_streamerコンポーネントを配置
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=VoskAudioProcessor,
    media_stream_constraints={
        "audio": {
            "autoGainControl": True,
            "echoCancellation": True,
            "noiseSuppression": True,
            "channelCount": 1,
            "sampleRate": 16000
        },
        "video": False
    },
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]},
)

# ストリームの状態表示
status_placeholder = st.empty()
if webrtc_ctx.state.playing:
    status_placeholder.info("🎧 音声認識中...話しかけてください")
else:
    status_placeholder.info("🛑 停止中")

# ---
### 📊 リアルタイム音量レベル

with st.sidebar:
    st.markdown("### 📊 リアルタイム音量レベル")

    if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        with webrtc_ctx.audio_processor.lock:
            st.session_state.volume_history.extend(webrtc_ctx.audio_processor.volume_history_queue)
            webrtc_ctx.audio_processor.volume_history_queue.clear()
        
        if st.session_state.volume_history:
            st.line_chart(st.session_state.volume_history[-50:])
    else:
        st.info("マイク入力を待機中です...")

# ---
### 📝 認識結果

result_placeholder = st.empty()

if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
    with webrtc_ctx.audio_processor.lock:
        while webrtc_ctx.audio_processor.transcription_queue:
            text = webrtc_ctx.audio_processor.transcription_queue.pop(0)
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
### 📋 履歴# 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import threading

# セッション状態の初期化
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []

# AudioProcessorBaseを継承したカスタムクラス
class SimpleAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.volume_history_queue = []
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """WebRTCから受信したオーディオフレームを処理します。"""
        # 音声データをNumpy配列に変換
        audio_data = frame.to_ndarray(format="s16le")

        # 音量レベルを計算し、キューに追加
        rms = np.sqrt(np.mean(np.square(audio_data)))
        with self.lock:
            self.volume_history_queue.append(rms)

        # 音量ブースト処理（ゲインを調整可能）
        gain = 2.0
        boosted_audio = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
        
        return frame.from_ndarray(boosted_audio)


# ---
### アプリケーションのメインUI ###
st.title("🎧 聴覚サポートアプリ（マイク・イヤホンテスト）")
st.write("マイク音声をリアルタイムで増幅し、イヤホンから再生します。")

# webrtc_streamerコンポーネントを配置
webrtc_ctx = webrtc_streamer(
    key="hearing_support",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=SimpleAudioProcessor,
    media_stream_constraints={
        "audio": {
            "autoGainControl": True,
            "echoCancellation": True,
            "noiseSuppression": True,
        },
        "video": False
    },
)

# ストリームの状態表示
status_placeholder = st.empty()
if webrtc_ctx.state.playing:
    status_placeholder.info("🎧 マイクとイヤホンの接続テスト中...")
else:
    status_placeholder.info("🛑 停止中")

# ---
### 📊 リアルタイム音量レベル

with st.sidebar:
    st.markdown("### 📊 リアルタイム音量レベル")

    if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        with webrtc_ctx.audio_processor.lock:
            st.session_state.volume_history.extend(webrtc_ctx.audio_processor.volume_history_queue)
            webrtc_ctx.audio_processor.volume_history_queue.clear()
        
        if st.session_state.volume_history:
            st.line_chart(st.session_state.volume_history[-50:])
    else:
        st.info("マイク入力を待機中です...")

if st.session_state.history:
    for line in reversed(st.session_state.history):
        st.write(f"- {line}")