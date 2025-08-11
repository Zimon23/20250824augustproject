import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Mini Analytics Dashboard", layout="wide")

st.title("📊 Mini Analytics Dashboard")
st.caption("Streamlit + Docker + GitHub Actions + Docker Hub")

# --- 데이터 준비(예제 시계열) ---
@st.cache_data
def make_data(days=30, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.today(), periods=days)
    values = rng.normal(loc=100, scale=10, size=days).cumsum()
    df = pd.DataFrame({"date": dates, "value": values})
    return df

with st.sidebar:
    st.header("⚙️ Controls")
    days = st.slider("날짜 길이(일)", 10, 120, 30, 5)
    seed = st.number_input("랜덤 시드", value=42, step=1)
    st.markdown("---")
    st.write("간단한 TODO")
    if "todos" not in st.session_state:
        st.session_state.todos = []
    new_todo = st.text_input("할 일 추가")
    if st.button("추가") and new_todo.strip():
        st.session_state.todos.append({"text": new_todo.strip(), "done": False})
    for i, t in enumerate(st.session_state.todos):
        cols = st.columns([0.1, 0.7, 0.2])
        with cols[0]:
            t["done"] = st.checkbox("", value=t["done"], key=f"todo_{i}")
        with cols[1]:
            st.write("~~"+t["text"]+"~~" if t["done"] else t["text"])
        with cols[2]:
            if st.button("삭제", key=f"del_{i}"):
                st.session_state.todos.pop(i)
                st.experimental_rerun()

df = make_data(days=days, seed=seed)

left, right = st.columns([2,1])
with left:
    st.subheader("시계열 미니 차트")
    st.line_chart(df.set_index("date"))

with right:
    st.subheader("요약 통계")
    st.metric("최근 값", f"{df['value'].iloc[-1]:.2f}")
    st.metric("최소", f"{df['value'].min():.2f}")
    st.metric("최대", f"{df['value'].max():.2f}")
    st.metric("변동폭", f"{(df['value'].max()-df['value'].min()):.2f}")

st.markdown("---")
st.subheader("CSV 업로드(선택)")
up = st.file_uploader("date,value 형식의 CSV 업로드 시 차트 교체", type=["csv"])
if up:
    try:
        user_df = pd.read_csv(up)
        # 최소한의 유효성 체크
        assert set(["date","value"]).issubset(set(map(str.lower, user_df.columns)))
        # 대소문자 컬럼 보정
        cols = {c: c.lower() for c in user_df.columns}
        user_df.rename(columns=cols, inplace=True)
        user_df["date"] = pd.to_datetime(user_df["date"])
        user_df = user_df.sort_values("date")
        st.success("업로드 성공! 아래 차트로 교체합니다.")
        st.line_chart(user_df.set_index("date"))
    except Exception as e:
        st.error(f"CSV 형식 오류: {e}")
