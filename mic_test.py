import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("マイク入力シンプルテスト")

webrtc_streamer(
    key="simple-test",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"audio": True, "video": False},
)
