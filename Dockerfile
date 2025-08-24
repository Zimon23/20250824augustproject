FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# 기존 PATH 유지 + appuser 로컬 bin만 추가
ENV PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential adduser \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# adduser는 /usr/sbin에 있으므로 절대경로로 호출
RUN /usr/sbin/adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]