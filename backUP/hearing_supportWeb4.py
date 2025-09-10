# ==========================
# 必要なライブラリをインポート
# ==========================
import streamlit as st                        # Webアプリを作るライブラリ
from streamlit_webrtc import (                # マイクやカメラを扱う拡張ライブラリ
    webrtc_streamer, WebRtcMode, AudioProcessorBase
)
import av                                     # 音声/動画フレームを扱うライブラリ
import numpy as np                            # 数値計算ライブラリ（行列・配列）
import threading                              # 並列処理（別スレッド実行）
import queue                                  # スレッド間でデータを渡すためのキュー
import logging                                # ログ出力用
from scipy.signal import butter, lfilter      # フィルター（音域調整）用

# ==========================
# ログの設定
# ==========================
logging.basicConfig(
    filename='hearing_support.log',           # ログを保存するファイル名
    level=logging.INFO,                       # ログレベル（INFO以上を記録）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)          # このプログラム専用のロガーを作成

# ==========================
# セッション変数（状態保持）
# ==========================
if "volume_history" not in st.session_state:  # 音量の履歴グラフを保存
    st.session_state.volume_history = []
if "processing_thread" not in st.session_state: # 音声処理用スレッドを保持
    st.session_state.processing_thread = None
if "audio_queue" not in st.session_state:     # 音声データを一時保存するキュー
    st.session_state.audio_queue = queue.Queue()
if "volume_queue" not in st.session_state:    # 音量レベルを保存するキュー
    st.session_state.volume_queue = queue.Queue()

# ==========================
# 音声フィルター関数
# ==========================
def butter_bandpass(lowcut, highcut, fs, order=5):
    # lowcut: 下限周波数
    # highcut: 上限周波数
    # fs: サンプリング周波数
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band') # バンドパスフィルター設計
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    # data: 音声データ
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)                   # データにフィルターを適用
    return y

# ==========================
# 音声パラメータを保持するクラス
# ==========================
class AudioParams:
    def __init__(self):
        self.gain_factor = 3.0                # 全体音量の倍率
        self.low_freq_boost = 0.0             # 低音域の強調レベル
        self.high_freq_boost = 0.0            # 高音域の強調レベル
        self.target_rms = 5000                # AGC(自動音量調整)の目標音量レベル

# セッションに音声パラメータを保持
if "audio_params" not in st.session_state:
    st.session_state.audio_params = AudioParams()

# ==========================
# 音声処理スレッドクラス
# ==========================
class AudioProcessingThread(threading.Thread):
    def __init__(self, audio_queue, volume_queue, params):
        super().__init__()
        self.audio_queue = audio_queue
        self.volume_queue = volume_queue
        self.params = params                  # 音声パラメータを参照
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():   # 停止フラグが立つまで処理を続ける
            try:
                frame = self.audio_queue.get(timeout=1) # 音声フレームを取得
                audio_data_float = frame.to_ndarray().astype(np.float64) # 数値化

                # --- パラメータを読み込む ---
                gain_factor = self.params.gain_factor
                low_freq_boost = self.params.low_freq_boost
                high_freq_boost = self.params.high_freq_boost
                target_rms = self.params.target_rms

                # --- EQ処理（低音/高音補正） ---
                if low_freq_boost != 0:
                    low_freq_data = bandpass_filter(audio_data_float.copy(), 20, 500, 16000)
                    audio_data_float += low_freq_data * low_freq_boost
                if high_freq_boost != 0:
                    high_freq_data = bandpass_filter(audio_data_float.copy(), 2000, 8000, 16000)
                    audio_data_float += high_freq_data * high_freq_boost

                # --- AGC（自動音量調整） ---
                rms = np.sqrt(np.mean(np.square(audio_data_float))) + 1e-6
                agc_gain = target_rms / rms
                audio_data_float *= agc_gain

                # --- ゲイン（音量調整） ---
                amplified_data = audio_data_float * gain_factor

                # --- クリッピング防止 ---
                amplified_data = np.clip(amplified_data, -32767, 32767)

                # --- RMSをモニタ用に保存 ---
                rms_after = np.sqrt(np.mean(np.square(amplified_data)))
                self.volume_queue.put(rms_after)

            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()

# ==========================
# WebRTCの音声処理クラス
# ==========================
class WebRtcAudioProcessor(AudioProcessorBase):
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_queue.put(frame)           # 音声フレームをキューに入れる
        return frame                          # そのまま出力に回す

# ==========================
# Streamlit UI部分
# ==========================
st.title("🎤 聴覚サポートアプリ")
st.write("マイク音声をリアルタイムで増幅・調整し、イヤホンから再生します。")

# --- 音量・音質調整スライダー ---
st.markdown("### 🔊 音声調整")
st.session_state.audio_params.gain_factor = st.slider(
    "全体音量（ゲイン）", 0.1, 200.0, 3.0, 0.1
)
st.session_state.audio_params.low_freq_boost = st.slider(
    "低音域調整", -2.0, 2.0, 0.0, 0.1
)
st.session_state.audio_params.high_freq_boost = st.slider(
    "高音域調整", -2.0, 2.0, 0.0, 0.1
)
st.session_state.audio_params.target_rms = st.slider(
    "AGC 目標音量レベル", 1000, 50000, 5000, 500
)

# --- WebRTC（マイク入力 + イヤホン出力） ---
try:
    processor = WebRtcAudioProcessor(st.session_state.audio_queue)
    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDRECV,            # 双方向（マイク入力→出力再生）
        audio_processor_factory=lambda: processor,
        media_stream_constraints={
            "audio": True,                   # 音声のみ
            "video": False
        },
    )
    st.session_state.webrtc_ctx = webrtc_ctx
except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.stop()

# --- 音声処理スレッドの開始/停止 ---
if webrtc_ctx.state.playing:
    if st.session_state.processing_thread is None or not st.session_state.processing_thread.is_alive():
        st.session_state.processing_thread = AudioProcessingThread(
            st.session_state.audio_queue, 
            st.session_state.volume_queue,
            st.session_state.audio_params
        )
        st.session_state.processing_thread.start()
else:
    if st.session_state.processing_thread is not None:
        st.session_state.processing_thread.stop()
        st.session_state.processing_thread.join()
        st.session_state.processing_thread = None

# --- 音量グラフ ---
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
