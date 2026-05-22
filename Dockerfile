FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./

RUN pip install --upgrade pip \
    && pip install \
        "joblib>=1.5.3" \
        "litestar[standard]>=2.21.1" \
        "pydantic>=2.13.4" \
        "scikit-learn==1.6.1" \
        "pandas>=2.2.0"

COPY app ./app
COPY main.py ./
COPY data ./data

EXPOSE 8000

CMD ["litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]
