# 必要パッケージ:
# pip install streamlit streamlit-webrtc av vosk numpy

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import vosk
import json
import time
import logging
import os

# -------------------------
# 設定・初期化
# -------------------------
st.set_page_config(page_title="聴覚サポートアプリ", layout="wide")

logging.basicConfig(
    filename=os.path.join(os.getcwd(), "hearing_support.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "vosk-model-small-ja-0.22"  # 事前にフォルダを配置

if "history" not in st.session_state:
    st.session_state.history = []
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "partial_text" not in st.session_state:
    st.session_state.partial_text = ""
if "is_recognizing" not in st.session_state:
    st.session_state.is_recognizing = False  # 開始/停止ボタン用

# -------------------------
# Voskモデル（キャッシュ）
# -------------------------
@st.cache_resource(show_spinner=True)
def load_vosk_model():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Voskモデルフォルダが見つかりません: {MODEL_PATH}\n"
            "https://alphacephei.com/vosk/models から ja のモデルを解凍して配置してください。"
        )
    model = vosk.Model(MODEL_PATH)
    return model

# -------------------------
# オーディオプロセッサ
# -------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.model = load_vosk_model()
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.recognizer.SetWords(True)

        # 出力側の音量（GUIから書き換える）
        self.gain = 1.0

        # レベル表示用
        self.level_rms = 0.0

        # 認識テキスト
        self.partial_text = ""
        self.final_texts = []

        # 変換（リサンプル）用
        self.resampler_16k_mono = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=16000
        )

    def _to_int16(self, x: np.ndarray) -> np.ndarray:
        # 期待するdtypeに揃える
        if x.dtype != np.int16:
            x = np.clip(x, -1.0, 1.0)
            x = (x * 32767.0).astype(np.int16)
        return x

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # ndarray: shape=(channels, samples)
        samples = frame.to_ndarray()
        # レベル計算（float化してRMS）
        f32 = samples.astype(np.float32)
        # 正規化（int16想定）
        if samples.dtype == np.int16:
            f32 = f32 / 32768.0
        rms = np.sqrt(np.mean(f32**2))
        self.level_rms = float(rms)

        # ---- 音量調整（出力用） ----
        # 出力は入力と同じサンプリングで返す（ブラウザ出力用）
        out = f32 * float(self.gain)
        out = np.clip(out, -1.0, 1.0)
        out_int16 = (out * 32767.0).astype(np.int16)

        # ---- 音声認識（16kHz, mono）----
        try:
            # frame -> 16k mono
            mono16k_frame = self.resampler_16k_mono.resample(frame)
            mono16k = mono16k_frame.to_ndarray()  # shape=(1, n)
            mono16k = self._to_int16(mono16k)
            pcm_bytes = mono16k.tobytes()

            if st.session_state.get("is_recognizing", False):
                if self.recognizer.AcceptWaveform(pcm_bytes):
                    result = json.loads(self.recognizer.Result())
                    text = (result.get("text") or "").strip()
                    if text:
                        self.final_texts.append(text)
                        self.partial_text = ""
                else:
                    presult = json.loads(self.recognizer.PartialResult())
                    ptext = (presult.get("partial") or "").strip()
                    self.partial_text = ptext
            else:
                # 停止中は部分結果をクリア
                self.partial_text = ""
        except Exception as e:
            logger.error(f"recognition error: {e}")

        # 変更したサンプルでAudioFrameを作って返す（イヤホンへ出力）
        out_frame = av.AudioFrame.from_ndarray(out_int16, layout=frame.layout)
        out_frame.sample_rate = frame.sample_rate
        return out_frame

# -------------------------
# UI
# -------------------------
st.title("🎤 聴覚サポートアプリ")
st.caption("マイク入力を**リアルタイムで文字起こし**し、**音量を調整**してイヤホンへ返します。")

col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("⚙️ コントロール")
    volume = st.slider("🔊 音量（出力ゲイン）", 0.0, 3.0, 1.0, 0.05, key="gain_slider")

    c1, c2 = st.columns(2)
    if c1.button("▶️ 認識開始", use_container_width=True):
        st.session_state.is_recognizing = True
    if c2.button("⏹ 認識停止", use_container_width=True):
        st.session_state.is_recognizing = False

    if st.button("🧹 履歴クリア", use_container_width=True):
        st.session_state.history.clear()

    st.markdown("---")
    st.markdown("### 📊 リアルタイム音量レベル")
    # 後で随時更新
    chart_placeholder = st.empty()

with col_left:
    st.subheader("📝 認識結果")
    partial_placeholder = st.empty()
    history_placeholder = st.empty()

st.markdown("---")
st.caption(
    "※ PCの再生音を文字起こししたい場合は「ステレオミキサー」や仮想オーディオデバイス（VB-CABLE等）をマイク入力に設定してください。"
)

# -------------------------
# WebRTC
# -------------------------
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,   # 受信した音声を処理して送り返す
    audio_receiver_size=1024,
    media_stream_constraints={
        "audio": {
            "autoGainControl": True,
            "echoCancellation": True,
            "noiseSuppression": True,
        },
        "video": False,
    },
    async_processing=True,  # 音声処理を別スレッドで
)

# -------------------------
# 状態の反映＆表示更新
# -------------------------
if webrtc_ctx and webrtc_ctx.state.playing:
    # AudioProcessorのインスタンスを取得
    ap: AudioProcessor = webrtc_ctx.audio_processor
    if ap is not None:
        # スライダー値をプロセッサに伝える（出力ボリューム）
        ap.gain = float(volume)

        # レベル履歴を更新（最大200点）
        st.session_state.volume_history.append(ap.level_rms)
        if len(st.session_state.volume_history) > 200:
            st.session_state.volume_history = st.session_state.volume_history[-200:]

        # 部分結果
        st.session_state.partial_text = ap.partial_text

        # 確定結果を履歴に取り込み
        if ap.final_texts:
            st.session_state.history.extend(ap.final_texts)
            ap.final_texts = []

    # 画面表示を更新
    if st.session_state.volume_history:
        chart_placeholder.line_chart(st.session_state.volume_history)
    else:
        chart_placeholder.info("入力待機中…")

    if st.session_state.partial_text:
        partial_placeholder.markdown(f"**いま聞こえた:** {st.session_state.partial_text}")
    else:
        partial_placeholder.markdown("_（部分結果なし）_")

    if st.session_state.history:
        # 新しい順で数件だけ
        latest = st.session_state.history[-40:]
        history_md = "\n".join(f"- {line}" for line in latest[::-1])
        history_placeholder.markdown(history_md)
    else:
        history_placeholder.info("ここに確定した文字起こしが表示されます。")
else:
    chart_placeholder = chart_placeholder if "chart_placeholder" in locals() else st.empty()
    chart_placeholder.info("接続を開始してください。")
