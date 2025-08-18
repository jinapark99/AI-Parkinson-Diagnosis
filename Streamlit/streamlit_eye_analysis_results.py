import streamlit as st
import pandas as pd
import json
import os
import glob
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # Added missing import for np.corrcoef

# 페이지 설정
st.set_page_config(
    page_title="시선 분석 결과",
    page_icon="📊",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
        <style>
            .main-header {
                font-size: 2.8rem;
                font-weight: 700;
                text-align: center;
                color: #2c3e50;
                margin-bottom: 3rem;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
                letter-spacing: -0.5px;
            }
            .metric-container {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 1.5rem;
                border-radius: 20px;
                margin: 1.5rem 0;
                border: 1px solid #dee2e6;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .encouragement-box {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 2rem;
                border-radius: 20px;
                text-align: center;
                margin: 3rem 0;
                box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
                border: none;
            }
            .trend-section {
                background: #ffffff;
                padding: 2rem;
                border-radius: 20px;
                margin: 2rem 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                border: 1px solid #f1f3f4;
            }
            .detail-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 1rem 2rem;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.1rem;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                transition: all 0.3s ease;
            }
            .detail-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }
        </style>
""", unsafe_allow_html=True)

def load_latest_results():
    """최근 결과 파일들을 로드하고 분석"""
    try:
        # Data_V2/eye_tracking_data 폴더에서 최근 파일들 찾기 (스크립트 기준 경로)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.normpath(os.path.join(base_dir, "..", "Data_V2", "eye_tracking_data"))
        if not os.path.exists(data_folder):
            return None, "데이터 폴더를 찾을 수 없습니다."
        
        # 년-월 폴더들 찾기
        month_folders = glob.glob(os.path.join(data_folder, "*/"))
        if not month_folders:
            return None, "데이터가 없습니다."
        
        # 가장 최근 월 폴더 찾기
        latest_month = max(month_folders, key=os.path.getctime)
        
        # 해당 월 폴더에서 최근 파일들 찾기
        json_files = glob.glob(os.path.join(latest_month, "*.json"))
        csv_files = glob.glob(os.path.join(latest_month, "*.csv"))
        
        if not json_files and not csv_files:
            return None, "분석할 데이터 파일이 없습니다."
        
        # 가장 최근 파일들 분석
        results = []
        
        # JSON 파일들 분석 (가장 최근 것부터)
        json_files.sort(key=os.path.getctime, reverse=True)
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 데이터 구조 검증 (아이트래킹 테스트용)
                    has_reaction_times = 'reaction_times' in data and len(data['reaction_times']) > 0
                    has_eye_sync_scores = 'eye_sync_scores' in data and len(data['eye_sync_scores']) > 0
                    has_movement_data = 'total_movements' in data
                    
                    # 분석 가능한 데이터가 있는 경우만 추가
                    if has_reaction_times or has_eye_sync_scores or has_movement_data:
                        results.append(data)
                        
            except Exception as e:
                continue
        
        # CSV 파일들 분석 (가장 최근 것부터)
        csv_files.sort(key=os.path.getctime, reverse=True)
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # CSV 데이터를 결과 형식으로 변환 (아이트래킹 테스트용)
                    result = {
                        'timestamp': os.path.basename(csv_file).split('_')[2].split('.')[0],
                        'total_frames': len(df),
                        'eye_sync_scores': df['eye_sync_score'].tolist() if 'eye_sync_score' in df.columns else [],
                        'reaction_times': df['reaction_time'].dropna().tolist() if 'reaction_time' in df.columns else []
                    }
                    results.append(result)
            except Exception as e:
                continue
        
        if not results:
            return None, "분석할 수 있는 데이터가 없습니다."
        
        return results, None
        
    except Exception as e:
        return None, f"데이터 로드 오류: {str(e)}"

def calculate_metrics(results):
    """결과 데이터에서 지표 계산"""
    if not results:
        return None
    
    # 가장 최근 결과 사용
    latest_result = results[0]  # 이미 정렬되어 있음 (최신이 0번째)
    
    metrics = {}
    
    # 1. 레이턴시 (반응 속도) - 아이트래킹 반응 시간으로 계산
    if 'reaction_times' in latest_result and len(latest_result['reaction_times']) > 0:
        reaction_times = latest_result['reaction_times']
        avg_reaction_time = sum(reaction_times) / len(reaction_times)
        
        # 0.3초 이하: 정상(1.0), 1초 이상: 비정상(0.0)
        if avg_reaction_time <= 0.3:
            latency_score = 1.0
        elif avg_reaction_time >= 1.0:
            latency_score = 0.0
        else:
            latency_score = 1.0 - ((avg_reaction_time - 0.3) / 0.7)
        metrics['latency'] = max(0.0, min(1.0, latency_score))
    else:
        metrics['latency'] = 0.5  # 기본값
    
    # 2. 동체움직임 (두 눈동자 협응) - 눈 동기화 점수로 계산
    if 'eye_sync_scores' in latest_result and len(latest_result['eye_sync_scores']) > 10:
        eye_sync_scores = latest_result['eye_sync_scores']
        avg_sync_score = sum(eye_sync_scores) / len(eye_sync_scores)
        
        # 0.9 이상: 정상(1.0), 0.6 이하: 비정상(0.0)
        if avg_sync_score >= 0.9:
            coordination_score = 1.0
        elif avg_sync_score <= 0.6:
            coordination_score = 0.0
        else:
            coordination_score = (avg_sync_score - 0.6) / 0.3
        metrics['coordination'] = max(0.0, min(1.0, coordination_score))
    else:
        metrics['coordination'] = 0.5  # 기본값
    
    # 3. 깜빡임 수 - 총 움직임 수로 계산 (아이트래킹 테스트에서는 움직임 추적)
    if 'total_movements' in latest_result:
        movements = latest_result['total_movements']
        
        # 8회 이상: 정상(1.0), 3회 이하: 비정상(0.0)
        if movements >= 8:
            movement_score = 1.0
        elif movements <= 3:
            movement_score = 0.0
        else:
            # 3-8 사이를 0.0-1.0으로 매핑
            movement_score = (movements - 3) / 5
        metrics['blink_rate'] = movement_score
    else:
        metrics['blink_rate'] = 0.5  # 기본값
    
    return metrics

def create_progress_bars(metrics):
    """진행률 바 차트 생성"""
    if not metrics:
        return None
    
    # Plotly를 사용한 가로 진행률 바
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('레이턴시 (반응 속도)', '동체움직임 (두 눈동자 협응)', '움직임 수'),
        vertical_spacing=0.35,
        specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # 모던한 색상 팔레트
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. 레이턴시
    fig.add_trace(
        go.Bar(
            x=[metrics['latency'] * 100],
            y=[''],
            orientation='h',
            marker_color=colors[0],
            name='레이턴시',
            text=f"{metrics['latency']:.1%}",
            textposition='auto',
            textfont=dict(size=16, color='white'),
            marker=dict(
                line=dict(width=0),
                pattern=dict(fillmode="overlay", size=[1], solidity=[0.3])
            )
        ),
        row=1, col=1
    )
    
    # 2. 동체움직임
    fig.add_trace(
        go.Bar(
            x=[metrics['coordination'] * 100],
            y=[''],
            orientation='h',
            marker_color=colors[1],
            name='동체움직임',
            text=f"{metrics['coordination']:.1%}",
            textposition='auto',
            textfont=dict(size=16, color='white'),
            marker=dict(
                line=dict(width=0),
                pattern=dict(fillmode="overlay", size=[1], solidity=[0.3])
            )
        ),
        row=2, col=1
    )
    
    # 3. 움직임 수
    fig.add_trace(
        go.Bar(
            x=[metrics['blink_rate'] * 100],
            y=[''],
            orientation='h',
            marker_color=colors[2],
            name='움직임 수',
            text=f"{metrics['blink_rate']:.1%}",
            textposition='auto',
            textfont=dict(size=16, color='white'),
            marker=dict(
                line=dict(width=0),
                pattern=dict(fillmode="overlay", size=[1], solidity=[0.3])
            )
        ),
        row=3, col=1
    )
    
    # 레이아웃 설정 - 깔끔하고 모던하게
    fig.update_layout(
        title=dict(
            text="시선 분석 결과",
            font=dict(size=20, color='#2c3e50'),
            x=0.5
        ),
        xaxis_title="정상도 (%)",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=80, b=40),
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False),
        xaxis2=dict(range=[0, 100], showgrid=False, zeroline=False),
        xaxis3=dict(range=[0, 100], showgrid=False, zeroline=False)
    )
    
    # 각 서브플롯에 기준선 추가 - 더 연하게
    for i in range(1, 4):
        fig.add_vline(x=80, line_dash="dot", line_color="#bdc3c7", line_width=0.5, row=i, col=1)
        fig.add_vline(x=60, line_dash="dot", line_color="#e8c39e", line_width=0.5, row=i, col=1)
    
    return fig

def get_encouragement_message(metrics):
    """결과에 따른 격려 메시지 생성"""
    if not metrics:
        return "데이터를 분석할 수 없습니다."
    
    # 전체 점수 계산
    total_score = (metrics['latency'] + metrics['coordination'] + metrics['blink_rate']) / 3
    
    if total_score >= 0.8:
        return "🎉 훌륭합니다! 시선 반응이 매우 정상적입니다. 건강한 상태를 유지하고 계세요!"
    elif total_score >= 0.6:
        return "👍 좋습니다! 전반적으로 정상적인 시선 반응을 보이고 있습니다. 꾸준한 관리가 필요합니다."
    elif total_score >= 0.4:
        return "⚠️ 주의가 필요합니다. 일부 지표에서 개선의 여지가 있습니다. 정기적인 검사를 권장합니다."
    else:
        return "💪 힘내세요! 현재 상태를 정확히 파악하는 것이 중요합니다. 전문의와 상담하시는 것을 권장합니다."

def calculate_daily_scores(results):
    """일별 종합 점수 계산 - 같은 날짜는 가장 최근 것만 선택"""
    daily_scores = []
    date_scores = {}  # 날짜별로 가장 최근 점수 저장
    
    for result in results:
        # 타임스탬프에서 날짜와 시간 추출
        timestamp = result.get('timestamp', '')
        if timestamp:
            try:
                # 2025-08-18_22-45-01 형식에서 날짜와 시간 추출
                date_str = timestamp.split('_')[0]  # 2025-08-18
                time_str = timestamp.split('_')[1]  # 22-45-01
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                time_obj = datetime.strptime(time_str, '%H-%M-%S')
                
                # 3개 지표 점수 계산
                metrics = calculate_metrics([result])
                if metrics:
                    total_score = (metrics['latency'] + metrics['coordination'] + metrics['blink_rate']) / 3
                    
                    # 같은 날짜가 있으면 더 최근 시간인지 확인
                    if date_str in date_scores:
                        if time_obj > date_scores[date_str]['time']:
                            # 더 최근 시간이면 업데이트
                            date_scores[date_str] = {
                                'date': date_obj,
                                'score': total_score * 100,
                                'timestamp': timestamp,
                                'time': time_obj
                            }
                    else:
                        # 새로운 날짜면 추가
                        date_scores[date_str] = {
                            'date': date_obj,
                            'score': total_score * 100,
                            'timestamp': timestamp,
                            'time': time_obj
                        }
            except Exception as e:
                continue
    
    # 날짜별 최신 점수만 추출하여 리스트로 변환
    daily_scores = list(date_scores.values())
    
    # 날짜순으로 정렬
    daily_scores.sort(key=lambda x: x['date'])
    return daily_scores

def create_trend_graph(daily_scores):
    """시간별 트렌드 그래프 생성"""
    if not daily_scores:
        return None
    
    # 첫 측정 점수를 기준으로 Y축 범위 설정
    first_score = daily_scores[0]['score']
    
    if first_score >= 90:
        # 90점 이상인 경우: 80~100점 범위
        y_min, y_max = 80, 100
        y_center = 90
    else:
        # 일반적인 경우: 첫 점수 ±10점 범위
        y_min = max(0, first_score - 10)
        y_max = min(100, first_score + 10)
        y_center = first_score
    
    # Y축 20개 칸으로 분할
    y_range = y_max - y_min
    y_step = y_range / 20
    
    # 데이터 포인트 준비
    dates = [score['date'] for score in daily_scores]
    scores = [score['score'] for score in daily_scores]
    
    # Plotly 그래프 생성
    fig = go.Figure()
    
    # 선 그래프 추가
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='종합 점수',
        line=dict(color='#1f77b4', width=4),
        marker=dict(
            size=16, 
            color='#1f77b4',
            line=dict(color='white', width=2),
            symbol='circle'
        ),
        hovertemplate='날짜: %{x}<br>점수: %{y:.1f}점<extra></extra>'
    ))
    
    # Y축 설정 (20개 칸)
    y_ticks = []
    y_tick_texts = []
    for i in range(21):  # 0~20번째 칸
        tick_value = y_min + (i * y_step)
        y_ticks.append(tick_value)
        y_tick_texts.append(f'{tick_value:.0f}')
    
    fig.update_layout(
        title=dict(
            text="📈 일별 시선 분석 트렌드",
            font=dict(size=18, color='#2c3e50'),
            x=0.5
        ),
        xaxis_title="날짜",
        yaxis_title="종합 점수",
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(
            range=[y_min, y_max],
            tickmode='array',
            tickvals=y_ticks,
            ticktext=y_tick_texts,
            gridcolor='rgba(236, 240, 241, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        ),
        xaxis=dict(
            gridcolor='rgba(236, 240, 241, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='#bdc3c7',
            linewidth=1
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#2c3e50',
            font_size=12
        )
    )
    
    # 기준선 추가 (첫 측정 점수) - 더 세련되게
    fig.add_hline(
        y=y_center, 
        line_dash="dot", 
        line_color="#e67e22", 
        line_width=2,
        annotation=dict(
            text=f"기준점: {y_center:.0f}점",
            font=dict(size=12, color='#e67e22'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e67e22',
            borderwidth=1
        )
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">시선 분석 결과</h1>', unsafe_allow_html=True)
    
    # 데이터 로드
    results, error = load_latest_results()
    
    if error:
        st.error(f"❌ {error}")
        return
    
    # 지표 계산
    metrics = calculate_metrics(results)
    
    if not metrics:
        st.warning("⚠️ 지표를 계산할 수 없습니다.")
        return
    
    # 진행률 바 차트만 표시 (메인 화면)
    st.markdown("### 📊 분석 결과")
    fig = create_progress_bars(metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # 격려 메시지
    encouragement = get_encouragement_message(metrics)
    st.markdown(f"""
    <div class="encouragement-box">
        <h3>💝 격려 메시지</h3>
        <p style="font-size: 1.2rem;">{encouragement}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 트렌드 그래프 추가 - 깔끔한 섹션으로
    st.markdown("<div class='trend-section'>", unsafe_allow_html=True)
    st.markdown("### 📈 일별 트렌드")
    
    # 일별 점수 계산
    daily_scores = calculate_daily_scores(results)
    
    if daily_scores:
        # 1달(30일) 이내 데이터만 필터링
        from datetime import timedelta
        current_date = datetime.now()
        month_ago = current_date - timedelta(days=30)
        
        recent_scores = [score for score in daily_scores if score['date'] >= month_ago]
        
        if recent_scores:
            trend_fig = create_trend_graph(recent_scores)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # 그래프 아래 정보 표시 - 더 깔끔하게
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"**📊 분석 기간**: {recent_scores[0]['date'].strftime('%Y-%m-%d')} ~ {recent_scores[-1]['date'].strftime('%Y-%m-%d')}")
                with col_info2:
                    st.markdown(f"**📈 총 측정 횟수**: {len(recent_scores)}회")
        else:
            st.info("📅 최근 1달간의 데이터가 없습니다.")
    else:
        st.info("📊 트렌드 분석을 위한 데이터가 부족합니다.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 우측 하단에 상세 정보 버튼 배치
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3열로 나누어 중앙에 버튼 배치
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 상세 정보 보기", key="detail_button", use_container_width=True, help="상세한 분석 결과를 확인합니다"):
            st.session_state.show_details = True
    
    # 상세 정보가 표시되어야 하는 경우
    if st.session_state.get('show_details', False):
        st.markdown("---")
        st.markdown("### 📋 상세 분석 정보")
        
        # 수치 결과 표시
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric("레이턴시", f"{metrics['latency']:.1%}", "반응 속도")
        
        with detail_col2:
            st.metric("동체움직임", f"{metrics['coordination']:.1%}", "두 눈 협응")
        
        with detail_col3:
            st.metric("움직임 수", f"{metrics['blink_rate']:.1%}", "움직임 추적")
        
        # 데이터 정보
        if results:
            latest = results[0]  # 최신 데이터
            st.markdown("#### 📊 원본 데이터")
            st.write(f"**분석 시간**: {latest.get('timestamp', '알 수 없음')}")
            st.write(f"**반응 시간**: {len(latest.get('reaction_times', []))}개")
            st.write(f"**움직임 횟수**: {latest.get('total_movements', 0)}회")
            
            # 반응 시간 상세
            if 'reaction_times' in latest:
                st.write(f"**반응 시간들**: {[f'{x:.3f}초' for x in latest['reaction_times']]}")
                st.write(f"**평균 반응 시간**: {sum(latest['reaction_times']) / len(latest['reaction_times']):.3f}초")
            
            # 눈 동기화 점수 상세
            if 'eye_sync_scores' in latest:
                st.write(f"**평균 동기화 점수**: {sum(latest['eye_sync_scores']) / len(latest['eye_sync_scores']):.3f}")
        
        # 상세 정보 숨기기 버튼
        if st.button("📊 상세 정보 숨기기", key="hide_detail_button"):
            st.session_state.show_details = False
            st.rerun()

if __name__ == "__main__":
    main()
