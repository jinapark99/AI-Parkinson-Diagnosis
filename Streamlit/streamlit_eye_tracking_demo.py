import streamlit as st
import subprocess
import os

# 페이지 설정
st.set_page_config(
    page_title="시선추적분석 시스템",
    page_icon="👁️",
    layout="centered"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .start-button {
        background-color: #28a745;
        color: white;
        padding: 1.5rem 3rem;
        border: none;
        border-radius: 50px;
        font-size: 1.8rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        display: block;
        margin: 0 auto;
        width: 300px;
    }
    .start-button:hover {
        background-color: #218838;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">시선추적분석을 시작하겠습니다</h1>', unsafe_allow_html=True)
    
    # 제목과 버튼 사이에 여백 추가
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # 중앙 정렬을 위한 컨테이너
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 시작하기 버튼을 중앙에 배치
        if st.button("시작하기", key="start_button", help="눈 깜빡임 측정을 시작합니다", use_container_width=True):
            st.info("🎬 시선추적분석을 시작합니다...")
            
            # 기존 eye_tracking_test_V2.py 실행
            script_path = "Scripts_V2/eye_tracking_test_V2.py"
            
            if os.path.exists(script_path):
                try:
                    # Python 스크립트 실행
                    result = subprocess.run(["python", script_path], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=300)  # 5분 타임아웃
                    
                    if result.returncode == 0:
                        st.success("✅ 시선추적분석이 완료되었습니다!")
                        st.info("결과는 Data_V2/blinking_test_data 폴더에 저장되었습니다.")
                    else:
                        st.error(f"❌ 오류가 발생했습니다: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    st.warning("⏰ 분석이 시간 초과되었습니다. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"❌ 예상치 못한 오류: {str(e)}")
            else:
                st.error("❌ eye_tracking_test_V2.py 파일을 찾을 수 없습니다.")
    
    # 버튼 아래에 여백 추가
    st.markdown("<br><br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
