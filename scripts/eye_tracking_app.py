import streamlit as st

st.title("🧠 파킨슨병 동공 움직임 분석")
st.write("사용자가 업로드한 영상을 분석하여 중증도를 예측합니다.")

uploaded_file = st.file_uploader("👁️ 영상 파일 업로드 (mp4)", type=["mp4"])

if uploaded_file:
    st.success("업로드 완료! 분석 코드는 곧 연결 예정입니다 👀")
