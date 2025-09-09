# 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import threading
import queue
import logging
from scipy.signal import butter, lfilter

# ロガーの設定
logging.basicConfig(
    filename='hearing_support.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# セッション状態の初期化
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "processing_thread" not in st.session_state:
    st.session_state.processing_thread = None

# 確実にキューを初期化する
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "volume_queue" not in st.session_state:
    st.session_state.volume_queue = queue.Queue()

# バターワースフィルターの設計
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# フィルター処理
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 音声処理を別スレッドで行うクラス
class AudioProcessingThread(threading.Thread):
    def __init__(self, audio_queue, volume_queue, gain_factor, low_freq_boost, high_freq_boost):
        super().__init__()
        self.audio_queue = audio_queue
        self.volume_queue = volume_queue
        self.gain_factor = gain_factor
        self.low_freq_boost = low_freq_boost
        self.high_freq_boost = high_freq_boost
        self.stop_event = threading.Event()
        logger.info("LOG: AudioProcessingThread 初期化完了。")

    def run(self):
        logger.info("LOG: AudioProcessingThread 開始。")
        while not self.stop_event.is_set():
            try:
                frame = self.audio_queue.get(timeout=1)
                
                # 音声データをfloat64に変換して処理を開始
                audio_data_float = frame.to_ndarray().astype(np.float64)
                
                # イコライザー処理
                if self.low_freq_boost != 0:
                    low_freq_data = bandpass_filter(audio_data_float.copy(), 20, 500, 16000)
                    audio_data_float += low_freq_data * self.low_freq_boost
                if self.high_freq_boost != 0:
                    high_freq_data = bandpass_filter(audio_data_float.copy(), 2000, 8000, 16000)
                    audio_data_float += high_freq_data * self.high_freq_boost
                
                # 全体ゲインを適用
                amplified_data = audio_data_float * self.gain_factor
                
                if amplified_data.size > 0:
                    rms = np.sqrt(np.mean(np.square(amplified_data)))
                    self.volume_queue.put(rms)
                    logger.info(f"LOG: 音声データ取得、RMS={rms:.2f}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"LOG: 音声処理エラー -> {e}")
                # エラー発生時もスレッドを停止せず、次のフレームを待機

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
        return frame

# ---
### アプリケーションのメインUI ###
st.title("🎤 聴覚サポートアプリ")
st.write("マイク音声をリアルタイムで増幅・調整し、イヤホンから再生します。")

# メイン画面に音量・音質調整スライダーを配置
st.markdown("### 🔊 音声調整")
gain_slider = st.slider(
    "全体音量（ゲイン）",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="マイクで拾った音を増幅するレベルを調整します。"
)

low_freq_boost_slider = st.slider(
    "低音域調整",
    min_value=-2.0,
    max_value=2.0,
    value=0.0,
    step=0.1,
    help="低音域（〜500Hz）の音量を調整します。ノイズが気になる場合は下げてください。"
)

high_freq_boost_slider = st.slider(
    "高音域調整",
    min_value=-2.0,
    max_value=2.0,
    value=0.0,
    step=0.1,
    help="高音域（2kHz〜8kHz）の音量を調整します。聴力に合わせて上げると聞き取りやすくなります。"
)

st.markdown("---")
st.markdown("### 📊 リアルタイム音量レベル")

while not st.session_state.volume_queue.empty():
    st.session_state.volume_history.append(st.session_state.volume_queue.get())
    if len(st.session_state.volume_history) > 100:
        st.session_state.volume_history.pop(0)

if "webrtc_ctx" in st.session_state and st.session_state.webrtc_ctx.state.playing:
    st.line_chart(st.session_state.volume_history)
else:
    st.info("マイク入力を待機中です...")


# webrtc_ctxの初期化を try-except で囲む
try:
    processor = WebRtcAudioProcessor(st.session_state.audio_queue)
    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: processor,
        media_stream_constraints={
            "audio": True,
            "video": False
        },
    )
    st.session_state.webrtc_ctx = webrtc_ctx

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.stop()


# ストリームの状態表示
status_placeholder = st.empty()
if webrtc_ctx.state.playing:
    status_placeholder.info("🎧 音声増幅中...")
    if st.session_state.processing_thread is None or not st.session_state.processing_thread.is_alive():
        st.session_state.processing_thread = AudioProcessingThread(
            st.session_state.audio_queue, 
            st.session_state.volume_queue,
            gain_slider,
            low_freq_boost_slider,
            high_freq_boost_slider
        )
        st.session_state.processing_thread.start()
elif not webrtc_ctx.state.playing and st.session_state.processing_thread is not None:
    st.session_state.processing_thread.stop()
    st.session_state.processing_thread.join()
    st.session_state.processing_thread = None
    status_placeholder.info("🛑 停止中")
else:
    status_placeholder.info("🛑 停止中")
