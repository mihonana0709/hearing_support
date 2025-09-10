# ==========================
# 必要な道具（ライブラリ）を呼び出す
# ==========================
import streamlit as st                        # streamlit → ウェブアプリを作るための道具
from streamlit_webrtc import (                # streamlit_webrtc → マイクや音を扱うための道具
    webrtc_streamer,                          # webrtc_streamer → マイクの音を拾ってイヤホンに出す道具
    WebRtcMode,                               # WebRtcMode → マイクの使い方のモード
    AudioProcessorBase                        # AudioProcessorBase → 音を加工する基本の型
)
import av                                     # av → 音や動画のデータを扱う道具
import numpy as np                            # numpy → 数を計算する道具（足す・掛ける・平均など）
import threading                              # threading → 「同時に動かす」仕組みを作る道具
import queue                                  # queue → 「順番にデータを並べる箱」
import logging                                # logging → 記録を残す道具
from scipy.signal import butter, lfilter      # scipy.signal → 音を加工（フィルター）する道具

# ==========================
# 記録の設定（ログ）
# ==========================
logging.basicConfig(
    filename='hearing_support.log',           # filename → 記録を保存するファイルの名前
    level=logging.INFO,                       # level → どのくらいの情報を残すか（INFO以上）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 記録の書き方
)
logger = logging.getLogger(__name__)          # logger → このプログラム専用の記録係

# ==========================
# アプリの中で覚えておく「状態（セッション変数）」
# ==========================
if "volume_history" not in st.session_state:  # volume_history → 音量の動きを保存
    st.session_state.volume_history = []
if "processing_thread" not in st.session_state: # processing_thread → 音を加工する別の動き
    st.session_state.processing_thread = None
if "audio_queue" not in st.session_state:     # audio_queue → 音のデータを一時的に置く箱
    st.session_state.audio_queue = queue.Queue()
if "volume_queue" not in st.session_state:    # volume_queue → 音量の数値を一時的に置く箱
    st.session_state.volume_queue = queue.Queue()

# ==========================
# 音をフィルターで整える関数
# ==========================
def butter_bandpass(lowcut, highcut, fs, order=5):
    # lowcut → 下の音の切る位置
    # highcut → 上の音の切る位置
    # fs → サンプリング周波数（1秒に何回音を測るか）
    nyquist = 0.5 * fs                        # nyquist → 周波数の半分（理論上の限界）
    low = lowcut / nyquist                    # low → 下の音の割合
    high = highcut / nyquist                  # high → 上の音の割合
    b, a = butter(order, [low, high], btype='band') # butter → フィルターを作る
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    # data → 音のデータ
    b, a = butter_bandpass(lowcut, highcut, fs, order=order) # b,a → フィルターの材料
    y = lfilter(b, a, data)                   # lfilter → フィルターをかける
    return y

# ==========================
# 音の調整パラメータ（数値の設定）
# ==========================
class AudioParams:
    def __init__(self):
        self.gain_factor = 3.0                # gain_factor → 音量を何倍にするか
        self.low_freq_boost = 0.0             # low_freq_boost → 低い音をどれくらい強くするか
        self.high_freq_boost = 0.0            # high_freq_boost → 高い音をどれくらい強くするか
        self.target_rms = 5000                # target_rms → 自動音量調整の目標値

# セッションに音の調整パラメータを保存
if "audio_params" not in st.session_state:
    st.session_state.audio_params = AudioParams()

# ==========================
# 音を加工する「別の動き（スレッド）」
# ==========================
class AudioProcessingThread(threading.Thread):
    def __init__(self, audio_queue, volume_queue, params):
        super().__init__()
        self.audio_queue = audio_queue        # 音のデータを入れる箱
        self.volume_queue = volume_queue      # 音量の数値を入れる箱
        self.params = params                  # 音の設定（パラメータ）
        self.stop_event = threading.Event()   # 止めるためのスイッチ

    def run(self):
        while not self.stop_event.is_set():   # 止めるスイッチが押されていなければ動く
            try:
                frame = self.audio_queue.get(timeout=1) # 音のデータを1つ取り出す
                audio_data_float = frame.to_ndarray().astype(np.float64) # 数字に変換

                # 設定（パラメータ）を読み込み
                gain_factor = self.params.gain_factor
                low_freq_boost = self.params.low_freq_boost
                high_freq_boost = self.params.high_freq_boost
                target_rms = self.params.target_rms

                # 低音の強調
                if low_freq_boost != 0:
                    low_freq_data = bandpass_filter(audio_data_float.copy(), 20, 500, 16000)
                    audio_data_float += low_freq_data * low_freq_boost

                # 高音の強調
                if high_freq_boost != 0:
                    # high_freq_data = bandpass_filter(audio_data_float.copy(), 2000, 8000, 16000)
                    # フィルターの設計において、クリティカルな周波数を 0 から 1 の間の値に変更します。
                    # 0.1 は、下限の周波数を意味します。
                    # 0.5 は、上限の周波数を意味します。
                    high_freq_data = bandpass_filter(audio_data_float.copy(), 0.1, 0.5, 16000)
                    audio_data_float += high_freq_data * high_freq_boost

                # 自動音量調整（AGC）
                rms = np.sqrt(np.mean(np.square(audio_data_float))) + 1e-6
                agc_gain = target_rms / rms
                audio_data_float *= agc_gain

                # 音量を全体的に大きくする
                amplified_data = audio_data_float * gain_factor

                # 音が大きすぎないように切る
                amplified_data = np.clip(amplified_data, -32767, 32767)

                # 今の音量を箱に入れる（グラフ用）
                rms_after = np.sqrt(np.mean(np.square(amplified_data)))
                self.volume_queue.put(rms_after)

            except queue.Empty:               # 箱が空っぽだったら
                continue                      # 何もせず次へ進む

    def stop(self):                           # 止める関数
        self.stop_event.set()

# ==========================
# WebRTC（マイク入力とイヤホン出力）
# ==========================
class WebRtcAudioProcessor(AudioProcessorBase):
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_queue.put(frame)           # 音のデータを箱に入れる
        return frame                          # そのままイヤホンに出す

# ==========================
# Streamlitで画面を作る
# ==========================
st.title("🎤 聴覚サポートアプリ")  # アプリの名前
st.write("マイクの音を大きくして、イヤホンから聞こえるようにします。")

# 音量・音質を調整するつまみ（スライダー）
st.markdown("### 🔊 音の調整")
st.session_state.audio_params.gain_factor = st.slider("全体音量（ゲイン）", 0.1, 200.0, 3.0, 0.1)
st.session_state.audio_params.low_freq_boost = st.slider("低音を強くする", -2.0, 2.0, 0.0, 0.1)
st.session_state.audio_params.high_freq_boost = st.slider("高音を強くする", -2.0, 2.0, 0.0, 0.1)
st.session_state.audio_params.target_rms = st.slider("自動音量の目標", 1000, 50000, 5000, 500)

# マイクから音を取って、イヤホンに出す
try:
    processor = WebRtcAudioProcessor(st.session_state.audio_queue)
    webrtc_ctx = webrtc_streamer(
        key="speech",                         # この処理の名前
        mode=WebRtcMode.SENDRECV,             # マイクから取って出すモード
        audio_processor_factory=lambda: processor, # 音を加工する工場
        media_stream_constraints={
            "audio": True,                    # 音を使う
            "video": False                    # 映像は使わない
        },
    )
    st.session_state.webrtc_ctx = webrtc_ctx
except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.stop()

# 音を加工するスレッドの開始と停止
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

# 音量のグラフを出す
st.markdown("---")
st.markdown("### 📊 音量のグラフ")

while not st.session_state.volume_queue.empty():
    st.session_state.volume_history.append(st.session_state.volume_queue.get())
    if len(st.session_state.volume_history) > 100:
        st.session_state.volume_history.pop(0)

if "webrtc_ctx" in st.session_state and st.session_state.webrtc_ctx.state.playing:
    st.line_chart(st.session_state.volume_history) # グラフに描く
else:
    st.info("マイク入力を待っています…")
