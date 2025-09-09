# Streamlitライブラリをインポート（Web UI構築用）
import streamlit as st

# WebRTC機能を提供するstreamlit-webrtcから必要なクラスをインポート
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# 音声認識エンジンVoskをインポート
import vosk

# 音声認識結果をJSON形式で扱うためのライブラリ
import json

# 音声データを数値配列として処理するための数値計算ライブラリ
import numpy as np

# WebRTC音声フレームを扱うためのPyAVライブラリ
import av

# ログ出力用ライブラリ（デバッグや記録に使用）
import logging

# ログの基本設定（INFOレベル以上を表示）
logging.basicConfig(level=logging.INFO)

# このスクリプト専用のロガーを作成
logger = logging.getLogger(__name__)

# Voskモデルの保存パス（事前にダウンロードしておく必要あり）
MODEL_PATH = "vosk-model-small-ja-0.22"

# Streamlitのセッション状態に履歴がなければ初期化
if "history" not in st.session_state:
    st.session_state.history = []

# 現在の文字起こし結果を保持する変数を初期化
if "current_transcription" not in st.session_state:
    st.session_state.current_transcription = ""

# 音量履歴（RMS値）を保持するリストを初期化
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []

# Voskモデルを読み込む関数（Streamlitのキャッシュ機能で高速化）
@st.cache_resource
def load_vosk_model():
    return vosk.Model(MODEL_PATH)

# WebRTCの音声フレームを処理するクラス（AudioProcessorBaseを継承）
class MyAudioProcessor(AudioProcessorBase):
    def __init__(self):
        # Voskモデルを読み込み、認識器を初期化（サンプリングレート16kHz）
        self.model = load_vosk_model()
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)

    # 音声フレームが届くたびに呼ばれるメソッド
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # 音声フレームをNumPy配列に変換（1次元の波形データ）
        audio_data = frame.to_ndarray()

        # 音量レベル（RMS）を計算し、履歴に追加
        rms = np.sqrt(np.mean(np.square(audio_data)))
        st.session_state.volume_history.append(rms)

        # 履歴が50件を超えたら古いデータを削除（グラフ表示のため）
        if len(st.session_state.volume_history) > 50:
            st.session_state.volume_history.pop(0)

        # 音声認識処理：確定結果が得られた場合
        if self.recognizer.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(self.recognizer.Result())["text"]
            if result.strip():  # 空文字でなければ履歴に追加
                st.session_state.history.append(result)
                st.session_state.current_transcription = ""
        else:
            # 部分認識結果（リアルタイム表示用）
            partial = json.loads(self.recognizer.PartialResult())["partial"]
            if partial.strip():
                st.session_state.current_transcription = partial

        # 処理後の音声フレームをそのまま返す（音声再生には使わないが必須）
        return frame

# StreamlitのUI構築部分
st.title("🎤 聴覚サポートアプリ")  # アプリのタイトル
st.write("ブラウザのマイク音声をリアルタイムで文字起こしします。")  # 説明文

# WebRTCストリームの初期化（音声のみ送受信）
webrtc_ctx = webrtc_streamer(
    key="speech",  # コンポーネントの識別キー
    mode=WebRtcMode.SENDRECV,  # 音声の送受信モード
    audio_processor_factory=MyAudioProcessor,  # 音声処理クラスを指定
    media_stream_constraints={"audio": True, "video": False},  # 音声のみ取得
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}  # STUNサーバー設定
)

# サイドバーに音量グラフを表示
with st.sidebar:
    st.markdown("### 📊 リアルタイム音量レベル")
    if st.session_state.volume_history:
        st.line_chart(st.session_state.volume_history)  # 音量履歴をグラフ表示
    else:
        st.info("マイク入力を待機中です...")  # 音声がまだ届いていない場合の表示

# メイン画面に現在の認識結果を表示
st.markdown("### 📝 認識結果")
if st.session_state.current_transcription:
    st.write(st.session_state.current_transcription)  # 部分認識結果を表示
else:
    st.info("ここに文字起こし結果が表示されます。")  # 初期状態のメッセージ

# 認識履歴を表示（最新10件を逆順で）
st.markdown("---")
st.markdown("### 📋 履歴")
for line in reversed(st.session_state.history[-10:]):
    st.write(f"- {line}")