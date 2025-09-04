# 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, get_devices
import vosk
import json
import os
import av # PyAVライブラリ
import threading
import numpy as np
import time
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Voskモデルのパスを指定（事前にダウンロードしておく）
MODEL_PATH = "vosk-model-small-ja-0.22"

# 認識履歴の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "recognizer" not in st.session_state:
    st.session_state.recognizer = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "current_transcription" not in st.session_state:
    st.session_state.current_transcription = ""

# Voskモデルの読み込み関数
@st.cache_resource
def load_vosk_model():
    try:
        model = vosk.Model(MODEL_PATH)
        logger.info("LOG: Voskモデルの読み込みに成功しました。")
        return model
    except Exception as e:
        st.error(f"❌ モデル読み込み失敗: {e}")
        logger.error(f"LOG: Voskモデルの読み込みに失敗しました: {e}")
        return None

# StreamlitのUI部分
st.title("🎤 聴覚サポートアプリ")
st.write("ブラウザのマイク音声をリアルタイムで文字起こしします。")

# デバイスリストを取得
devices = get_devices()
audio_input_devices = [d for d in devices if d.kind == "audioinput"]
audio_output_devices = [d for d in devices if d.kind == "audiooutput"]

audio_input_labels = [d.label for d in audio_input_devices]
audio_output_labels = [d.label for d in audio_output_devices]

# デバイス選択のためのセレクトボックス
audio_input_selected_label = st.selectbox(
    "マイクを選択", audio_input_labels
)
audio_input_device_id = next((d.id for d in audio_input_devices if d.label == audio_input_selected_label), None)

# webrtc_streamerコンポーネントを配置
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,  # 音声の送受信を停止
    audio_receiver_size=2048,
    media_stream_constraints={
        "audio": {
            "deviceId": {"exact": audio_input_device_id}
        },
        "video": False
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# サイドバーにUI要素を配置
with st.sidebar:
    # 音量レベルのグラフ表示
    st.markdown("---")
    st.markdown("### 📊 リアルタイム音量レベル")
    if webrtc_ctx.state.playing and st.session_state.volume_history:
        st.line_chart(st.session_state.volume_history)
    else:
        st.info("マイク入力を待機中です...")

# メイン画面にUI要素を配置
# マイクの状態表示
status_placeholder = st.empty()
if webrtc_ctx.state.playing:
    st.session_state.is_listening = True
    status_placeholder.info("🎧 音声認識中...話しかけてください")
else:
    st.session_state.is_listening = False
    status_placeholder.info("🛑 停止中")

# 認識結果表示エリア
st.markdown("### 📝 認識結果")
result_placeholder = st.empty()
if st.session_state.current_transcription:
    result_placeholder.write(st.session_state.current_transcription)
else:
    result_placeholder.info("ここに文字起こし結果が表示されます。")

st.markdown("---")
st.markdown("### 📋 履歴")
if st.session_state.history:
    for line in reversed(st.session_state.history):
        st.write(f"- {line}")

# WebRTCストリームの処理（バックグラウンド）
if webrtc_ctx.audio_receiver:
    logger.info("LOG: webrtc_ctx.audio_receiverが利用可能です。")
    if st.session_state.recognizer is None:
        try:
            vosk_model = load_vosk_model()
            if vosk_model:
                st.session_state.recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
                logger.info("LOG: Vosk認識エンジンが初期化されました。")
        except Exception as e:
            st.error(f"音声認識エンジン初期化失敗: {e}")
            logger.error(f"LOG: Vosk認識エンジン初期化失敗: {e}")
            st.session_state.is_listening = False
    
    try:
        while webrtc_ctx.state.playing:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            
            if frames:
                logger.info(f"LOG: {len(frames)} 個の音声フレームを受信しました。")
                for frame in frames:
                    audio_data = frame.to_ndarray()
                    
                    # 音量レベルを計算し、グラフ履歴を更新
                    rms = np.sqrt(np.mean(np.square(audio_data)))
                    st.session_state.volume_history.append(rms)
                    if len(st.session_state.volume_history) > 50:
                        st.session_state.volume_history.pop(0)
                    
                    # 認識処理
                    if st.session_state.recognizer:
                        logger.info("LOG: Vosk認識エンジンにデータを渡しています。")
                        if st.session_state.recognizer.AcceptWaveform(audio_data.tobytes()):
                            result = st.session_state.recognizer.Result()
                            text = json.loads(result)["text"]
                            if text.strip():
                                st.session_state.history.append(text)
                                st.session_state.current_transcription = ""
                                logger.info(f"LOG: 最終的な文字起こし結果: {text}")
                        else:
                            partial_result = st.session_state.recognizer.PartialResult()
                            partial_text = json.loads(partial_result)["partial"]
                            if partial_text.strip():
                                st.session_state.current_transcription = partial_text
                                logger.info(f"LOG: 部分的な文字起こし結果: {partial_text}")
            else:
                # フレームが受信されない場合でもループを継続
                logger.info("LOG: フレームが受信されませんでした。")
                time.sleep(0.1)
    except Exception as e:
        logger.error(f"LOG: 音声フレーム取得中にエラーが発生しました: {e}")
