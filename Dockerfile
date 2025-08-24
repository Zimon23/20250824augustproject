FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

WORKDIR /app

# 빌드 도구(일부 패키지 컴파일 대비)
# 권장: 한 블록으로 병합
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# 모델/로그 저장 위치 권한
RUN useradd -m appuser && mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]