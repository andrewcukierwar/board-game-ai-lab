# docker/api.Dockerfile  ── multi-stage build

# ---------- builder stage ----------
FROM python:3.11-slim AS builder

WORKDIR /opt/venv
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app

# copy the ready-made virtual-env and your source code
COPY --from=builder /opt/venv /opt/venv
COPY api/ ./api
COPY games/ ./games

# gunicorn serves Flask on port 8000
EXPOSE 8000
CMD ["gunicorn", "api.app:app", "--bind=0.0.0.0:8000", "--workers=1", "--threads=4"]
