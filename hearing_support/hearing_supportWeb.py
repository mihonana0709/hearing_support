# 必要なライブラリをインポート
# インポートエラーが発生する場合は、以下のコマンドをターミナルで実行してください:
# pip install streamlit-webrtc
# pip install av
# 仮想環境を使用している場合は、VS Codeで正しいインタープリターが選択されているか確認してください。
import streamlit as st # Streamlitをインポートし、Webアプリケーションを構築します
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase # WebRTC機能を提供します
import vosk # 音声認識エンジンVoskをインポートします
import json # JSONデータを扱います
import os # ファイルパスを扱います
import av # PyAVライブラリをインポートし、音声データを処理します
import threading # スレッドを扱います
import numpy as np # 数値計算ライブラリNumpyをインポートし、音声データを数値として扱います
import time # 時間関連の関数を扱います
import logging # ログ出力を扱います

# ロガーの設定
logging.basicConfig(level=logging.INFO) # ログレベルをINFOに設定します
logger = logging.getLogger(__name__) # このスクリプトのロガーを作成します

# Voskモデルのパスを指定（事前にダウンロードしておく）
MODEL_PATH = "vosk-model-small-ja-0.22" # Voskの日本語モデルのパスを定義します

# 認識履歴の初期化
if "history" not in st.session_state: # セッション状態に'history'が存在しない場合
    st.session_state.history = [] # 履歴リストを初期化します
if "is_listening" not in st.session_state: # 'is_listening'が存在しない場合
    st.session_state.is_listening = False # 聴取状態をFalseに初期化します
if "recognizer" not in st.session_state: # 'recognizer'が存在しない場合
    st.session_state.recognizer = None # Vosk認識エンジンをNoneに初期化します
if "model_loaded" not in st.session_state: # 'model_loaded'が存在しない場合
    st.session_state.model_loaded = False # モデル読み込み状態をFalseに初期化します
if "volume_history" not in st.session_state: # 'volume_history'が存在しない場合
    st.session_state.volume_history = [] # 音量履歴リストを初期化します
if "start_time" not in st.session_state: # 'start_time'が存在しない場合
    st.session_state.start_time = time.time() # 開始時間を記録します
if "current_transcription" not in st.session_state: # 'current_transcription'が存在しない場合
    st.session_state.current_transcription = "" # 現在の文字起こし結果を空文字列に初期化します

# Voskモデルの読み込み関数
@st.cache_resource
def load_vosk_model(): # Voskモデルをキャッシュして読み込みます
    try:
        model = vosk.Model(MODEL_PATH) # 指定されたパスからモデルを読み込みます
        logger.info("LOG: Voskモデルの読み込みに成功しました。") # ログに成功メッセージを出力します
        return model # モデルを返します
    except Exception as e: # 例外が発生した場合
        st.error(f"❌ モデル読み込み失敗: {e}") # Streamlitにエラーメッセージを表示します
        logger.error(f"LOG: Voskモデルの読み込みに失敗しました: {e}") # ログにエラーメッセージを出力します
        return None # Noneを返します

# StreamlitのUI部分
st.title("🎤 聴覚サポートアプリ") # アプリケーションのタイトルを設定します
st.write("ブラウザのマイク音声をリアルタイムで文字起こしします。") # アプリケーションの説明文を表示します

# webrtc_streamerコンポーネントを配置
webrtc_ctx = webrtc_streamer( # WebRTCストリーミングコンポーネントを作成します
    key="speech", # コンポーネントの一意なキー
    mode=WebRtcMode.SENDRECV,  # 音声の送受信を有効化します
    audio_receiver_size=2048, # 音声受信バッファのサイズを設定します
    media_stream_constraints={ # メディアストリームの制約を設定します
        "audio": {
            "autoGainControl": True, # 自動ゲインコントロールを有効化します
            "echoCancellation": True, # エコーキャンセリングを有効化します
            "noiseSuppression": True, # ノイズ抑制を有効化します
        },
        "video": False # ビデオを無効にします
    },
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # ICEサーバーを設定します
)

# JavaScriptを使用して音声ストリームのボリュームを制御
st.markdown("""
<script>
    const audio = document.querySelector('audio'); // HTMLのaudio要素を取得します
    if (audio) { // audio要素が存在する場合
        audio.volume = 0.5; // デフォルトの音量を0.5に設定します
        const volumeSlider = document.querySelector('#volume-slider input'); // ボリュームスライダーのinput要素を取得します
        if (volumeSlider) { // スライダー要素が存在する場合
            volumeSlider.addEventListener('input', (event) => { // スライダーの変更を監視します
                audio.volume = event.target.value; // スライダーの値に基づいて音量を設定します
            });
        }
    }
</script>
""", unsafe_allow_html=True) # HTMLの埋め込みを許可します

# サイドバーにUI要素を配置
with st.sidebar: # サイドバーにコンテンツを配置します
    # 音量レベルのグラフ表示
    st.markdown("---") # 区切り線を表示します
    st.markdown("### 📊 リアルタイム音量レベル") # サブタイトルを表示します
    if webrtc_ctx.state.playing and st.session_state.volume_history: # 再生中で音量履歴が存在する場合
        st.line_chart(st.session_state.volume_history) # 音量履歴を折れ線グラフで表示します
    else:
        st.info("マイク入力を待機中です...") # マイク入力待機中のメッセージを表示します

# ボリュームコントロール
volume = st.slider("🔊 音量", 0.0, 1.0, 0.5, 0.05, key="volume-slider") # 音量調整用のスライダーを作成します

# メイン画面にUI要素を配置
# マイクの状態表示
status_placeholder = st.empty() # 状態表示用のプレースホルダーを作成します
if webrtc_ctx.state.playing: # WebRTCが再生中の場合
    st.session_state.is_listening = True # 聴取状態をTrueに設定します
    status_placeholder.info("🎧 音声認識中...話しかけてください") # 音声認識中のメッセージを表示します
else:
    st.session_state.is_listening = False # 聴取状態をFalseに設定します
    status_placeholder.info("🛑 停止中") # 停止中のメッセージを表示します

# 認識結果表示エリア
st.markdown("### 📝 認識結果") # サブタイトルを表示します
result_placeholder = st.empty() # 認識結果表示用のプレースホルダーを作成します
if st.session_state.current_transcription: # 現在の文字起こし結果が存在する場合
    result_placeholder.write(st.session_state.current_transcription) # 結果を表示します
else:
    result_placeholder.info("ここに文字起こし結果が表示されます。") # 結果待機中のメッセージを表示します

st.markdown("---") # 区切り線を表示します
st.markdown("### 📋 履歴") # サブタイトルを表示します
if st.session_state.history: # 履歴が存在する場合
    for line in reversed(st.session_state.history): # 履歴を新しい順に表示します
        st.write(f"- {line}") # 各行をリスト形式で表示します

# WebRTCストリームの処理（バックグラウンド）
if webrtc_ctx.audio_receiver: # 音声受信機が利用可能な場合
    logger.info("LOG: webrtc_ctx.audio_receiverが利用可能です。") # ログメッセージを出力します
    if st.session_state.recognizer is None: # 認識エンジンが初期化されていない場合
        try:
            vosk_model = load_vosk_model() # Voskモデルを読み込みます
            if vosk_model: # モデルが正常に読み込まれた場合
                # サンプリングレートを16000に固定
                st.session_state.recognizer = vosk.KaldiRecognizer(vosk_model, 16000) # 認識エンジンを初期化します
                logger.info("LOG: Vosk認識エンジンが初期化されました。サンプリングレート: 16000") # ログメッセージを出力します
        except Exception as e: # 例外が発生した場合
            st.error(f"音声認識エンジン初期化失敗: {e}") # Streamlitにエラーメッセージを表示します
            logger.error(f"LOG: 音声認識エンジン初期化失敗: {e}") # ログにエラーメッセージを出力します
            st.session_state.is_listening = False # 聴取状態をFalseに設定します
    
    try:
        while webrtc_ctx.state.playing: # WebRTCが再生中の間ループを続けます
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1) # 1秒以内に音声フレームを取得します
            
            if frames: # フレームが取得できた場合
                logger.info(f"LOG: {len(frames)} 個の音声フレームを受信しました。") # 受信したフレーム数を出力します
                for frame in frames: # 各フレームをループします
                    audio_data = frame.to_ndarray() # 音声データをNumpy配列に変換します
                    
                    # 音量レベルを計算し、グラフ履歴を更新
                    rms = np.sqrt(np.mean(np.square(audio_data))) # RMS（二乗平均平方根）を計算します
                    st.session_state.volume_history.append(rms) # 音量履歴に追加します
                    if len(st.session_state.volume_history) > 50: # 履歴が50を超えた場合
                        st.session_state.volume_history.pop(0) # 最も古い履歴を削除します
                    
                    # 認識処理
                    if st.session_state.recognizer: # 認識エンジンが利用可能な場合
                        logger.info("LOG: Vosk認識エンジンにデータを渡しています。") # ログメッセージを出力します
                        if st.session_state.recognizer.AcceptWaveform(audio_data.tobytes()): # 認識結果が確定した場合
                            result = st.session_state.recognizer.Result() # 確定した結果を取得します
                            text = json.loads(result)["text"] # JSONからテキストを抽出します
                            if text.strip(): # テキストが空でない場合
                                st.session_state.history.append(text) # 履歴に追加します
                                st.session_state.current_transcription = "" # 現在の文字起こしをリセットします
                                logger.info(f"LOG: 最終的な文字起こし結果: {text}") # ログに最終結果を出力します
                        else:
                            partial_result = st.session_state.recognizer.PartialResult() # 部分的な結果を取得します
                            partial_text = json.loads(partial_result)["partial"] # JSONから部分的なテキストを抽出します
                            if partial_text.strip(): # 部分的なテキストが空でない場合
                                st.session_state.current_transcription = partial_text # 現在の文字起こしを更新します
                                logger.info(f"LOG: 部分的な文字起こし結果: {partial_text}") # ログに部分的な結果を出力します
            else:
                # フレームが受信されない場合でもループを継続
                logger.info("LOG: フレームが受信されませんでした。") # ログメッセージを出力します
                time.sleep(0.1) # 100ミリ秒待機します
    except Exception as e: # 例外が発生した場合
        logger.error(f"LOG: 音声フレーム取得中にエラーが発生しました: {e}") # ログにエラーメッセージを出力します
