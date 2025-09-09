import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("🎤 マイク接続テスト")
st.write("このアプリは、マイクがブラウザで正常に動作することを確認します。")

webrtc_streamer(
    key="mic-test",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"audio": True, "video": False}
)