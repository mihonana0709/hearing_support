# 📦 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import vosk
import json
import os

# 📁 Voskモデルのパスを指定（事前にダウンロードしておく）
MODEL_PATH = "vosk-model-small-ja-0.22"  # フォルダ名を正確に指定

# 📝 認識履歴の初期化（Streamlitセッションステート）
if "history" not in st.session_state:
    st.session_state.history = []

# 🎤 音声処理クラス（WebRTCから受信した音声をVoskで認識）
class VoskAudioProcessor(AudioProcessorBase):
    def __init__(self):
        # 🔁 モデルの読み込みは初回のみ（遅延ロードでエラー回避）
        self.model = None
        self.rec = None

    def recv(self, frame):
        print("🔊 recv() called")  # ← Streamlitのターミナルに出力される
        # 🔁 モデルが未ロードならここで読み込む（初回のみ）
        if self.model is None:
            if not os.path.exists(MODEL_PATH):
                st.error(f"❌ モデルフォルダが見つかりません: {MODEL_PATH}")
                return frame  # 認識せずにそのまま返す
            
            try:
                self.model = vosk.Model(MODEL_PATH)
                self.rec = vosk.KaldiRecognizer(self.model, 16000)
                st.success("✅ モデル読み込み成功")  # ← tryブロック内に追加してもOK

            except Exception as e:
                st.error(f"❌ モデル読み込み失敗: {e}")
                return frame


        # 🎧 音声フレームをNumPy配列に変換
        audio = frame.to_ndarray()

        # 🔍 音声データの構造を確認（デバッグ用）
        st.write(f"🔊 音声データの一部: {audio[:5]}")
        st.write(f"📐 audio.shape: {audio.shape}")
        st.write(f"📦 audio.dtype: {audio.dtype}")

        # recv() 内でログを session_state に保存
        if "debug_log" not in st.session_state:
            st.session_state.debug_log = []

        st.session_state.debug_log.append({
            "audio_head": audio[:5].tolist(),
            "shape": str(audio.shape),
            "dtype": str(audio.dtype)
        })


        # 🧠 Voskで音声認識を実行
        if self.rec.AcceptWaveform(audio):
            result = self.rec.Result()
            text = json.loads(result)["text"]
            st.write(f"🧪 認識結果: {text}")  # ← UIに表示して確認

            # ✏️ 空文字でなければ履歴に追加
            if text.strip():
                st.session_state.history.append(text)

        # 🔁 処理後のフレームをそのまま返す（音声再生は不要）
        return frame

# 🖼️ StreamlitのUI部分
st.title("🎤 音声認識アプリ（Web対応）")
st.subheader("🔍 デバッグログ")
# for log in st.session_state.get("debug_log", []):
#     st.markdown(f"- 音声先頭: `{log['audio_head']}` / 形状: `{log['shape']}` / 型: `{log['dtype']}`")
# st.markdown("ブラウザからマイク入力 → Voskでリアルタイム認識")
# デバッグログが存在する場合のみ表示
if "debug_log" in st.session_state and st.session_state.debug_log:
    for i, log in enumerate(st.session_state.debug_log):
        st.markdown(f"**#{i+1}** 音声先頭: `{log['audio_head']}` / 形状: `{log['shape']}` / 型: `{log['dtype']}`")
else:
    st.info("まだ音声データが届いていません。マイク入力を確認してください。")

# 🌐 WebRTC接続設定（STUNサーバーを明示的に指定）
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# 🔌 WebRTCによる音声受信と処理の開始
webrtc_streamer(
    key="speech",  # セッション識別キー
    mode=WebRtcMode.SENDRECV,  # 音声送受信モード
    audio_processor_factory=VoskAudioProcessor,  # 音声処理クラス
    media_stream_constraints={"audio": True, "video": False},  # 音声のみ
    rtc_configuration=RTC_CONFIGURATION,  # STUN設定で接続安定化
    async_processing=True,  # 非同期処理を有効化
)

# 📋 認識履歴の表示
st.subheader("📝 認識履歴")
for line in st.session_state.history:
    st.markdown(f"- {line}")