# 必要なライブラリをインポート
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import json
import av

# ロガーの設定
logging.basicConfig(
    filename='hearing_support.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---
### アプリケーションのメインUI ###
st.title("🎤 聴覚サポートアプリ - デバイス選択")
st.write("マイクとイヤホンのデバイスを個別に選択し、動作を確認します。")

# デバイス一覧を取得
try:
    media_devices = json.loads(av.get_media_devices("audioinput"))
    device_options = {dev['label']: dev['deviceId'] for dev in media_devices}
    selected_device_name = st.selectbox(
        "使用するマイクを選択してください:",
        options=list(device_options.keys())
    )
    selected_device_id = device_options[selected_device_name]
    logger.info(f"LOG: 選択されたマイクデバイス: {selected_device_name} ({selected_device_id})")
except Exception as e:
    st.warning("マイクデバイスの取得に失敗しました。")
    st.error(f"エラー詳細: {e}")
    selected_device_id = None
    selected_device_name = "マイクデバイスが見つかりません"

if selected_device_id:
    # 選択したマイクデバイスでWebRTCストリームを生成
    webrtc_ctx = webrtc_streamer(
        key="device_test",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "audio": {
                "deviceId": {"exact": selected_device_id},
            },
            "video": False
        },
    )

    if webrtc_ctx.state.playing:
        st.info("🎧 音声が処理されています。イヤホンから自分の声が聞こえるか確認してください。")
    else:
        st.warning("🛑 停止中")