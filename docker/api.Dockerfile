# docker/api.Dockerfile  ── multi-stage build

# ---------- builder stage ----------
FROM python:3.11-slim AS builder

WORKDIR /opt/venv
COPY requirements-api.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt

# Optional AlphaZero inference runtime. Default false keeps the image torch-free;
# ALPHAZERO_ENABLED must also be set at run time to expose the agent.
ARG ALPHAZERO_RUNTIME=false
COPY requirements-alphazero.txt .
RUN if [ "$ALPHAZERO_RUNTIME" = "true" ]; then \
      /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        -r requirements-alphazero.txt; \
    elif [ "$ALPHAZERO_RUNTIME" != "false" ]; then \
      echo "ALPHAZERO_RUNTIME must be true or false" >&2; exit 1; \
    fi

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
# Approved inference-only AlphaZero artifact (1.3 MB) and its provenance manifest; read only.
COPY models/alphazero/ ./models/alphazero/
COPY --from=builder /opt/build/games/connect4/victor/.native-*.so ./games/connect4/victor/

# One worker is required by the process-local session store. Render supplies PORT.
EXPOSE 8000
CMD ["gunicorn", "api.app:app", "--config=python:api.gunicorn_config", "--workers=1", "--threads=4"]
