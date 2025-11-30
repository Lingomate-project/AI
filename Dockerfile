# 베이스 이미지: 가벼운 python 3.11
FROM python:3.11-slim

# 시스템 기본 패키지 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 코드 복사
COPY . .

# 로깅/버퍼링 설정
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# FastAPI 포트
EXPOSE 8000

# 컨테이너가 뜰 때 실행할 명령
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
