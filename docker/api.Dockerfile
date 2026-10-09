# docker/api.Dockerfile  ── multi-stage build

# ---------- builder stage ----------
FROM python:3.11-slim AS builder

WORKDIR /opt/venv
COPY requirements-api.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt

# Build the optional bounded Victor search on the target architecture. The
# compiler stays in the builder; no compilation or oracle import occurs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
COPY games/ /opt/build/games/
RUN cd /opt/build && /opt/venv/bin/python -c 'from games.connect4.victor.native import build; build()'

# ---------- runtime stage ----------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app

# copy the ready-made virtual-env and your source code
COPY --from=builder /opt/venv /opt/venv
COPY api/ ./api
COPY games/ ./games
COPY --from=builder /opt/build/games/connect4/victor/.native-*.so ./games/connect4/victor/

# One worker is required by the process-local session store. Render supplies PORT.
EXPOSE 8000
CMD ["gunicorn", "api.app:app", "--config=python:api.gunicorn_config", "--workers=1", "--threads=4"]
