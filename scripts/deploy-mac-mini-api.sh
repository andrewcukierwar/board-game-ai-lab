#!/usr/bin/env bash
# Reviewed ARM64 production update. Never pulls/merges or touches other worktrees.
# Keeps the prior container stopped as bgai-api-rollback for manual recovery.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_FILE="${BGAI_API_ENV_FILE:-$HOME/.config/board-game-ai-lab/api.env}"
CANDIDATE="bgai-api-candidate"
ROLLBACK="bgai-api-rollback"
MODE="${1:-lite}"
if [[ $# -gt 1 || "$MODE" != lite && "$MODE" != alphazero ]]; then
  echo "Usage: $0 [lite|alphazero] (default: lite, torch-free and disabled)" >&2
  exit 1
fi
ALPHAZERO_RUNTIME=false
ALPHAZERO_ENABLED=false
if [[ "$MODE" == alphazero ]]; then
  ALPHAZERO_RUNTIME=true
  ALPHAZERO_ENABLED=true
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing private environment file: $ENV_FILE" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to deploy a dirty checkout; commit or discard changes first." >&2
  exit 1
fi
if docker container inspect "$ROLLBACK" >/dev/null 2>&1; then
  echo "Existing $ROLLBACK container found. Inspect/remove it deliberately before redeploying." >&2
  exit 1
fi
if ! docker container inspect bgai-api >/dev/null 2>&1; then
  echo "The current bgai-api container is missing; expected an existing deployment." >&2
  exit 1
fi
if docker container inspect "$CANDIDATE" >/dev/null 2>&1; then
  echo "Existing candidate found; refusing to remove a container this run does not own." >&2
  exit 1
fi
docker info >/dev/null
SHA="$(git rev-parse HEAD)"
TAG="bgai-api:${SHA}-${MODE}"
echo "Building reviewed revision $SHA as $TAG (linux/arm64)"
docker buildx build --platform linux/arm64 --load --build-arg ALPHAZERO_RUNTIME="$ALPHAZERO_RUNTIME" -t "$TAG" -f docker/api.Dockerfile .
ARCH="$(docker image inspect "$TAG" --format '{{.Architecture}}')"
if [[ "$ARCH" != arm64 ]]; then
  echo "Unexpected image architecture: $ARCH" >&2
  exit 1
fi

PROMOTION_STARTED=0
CANDIDATE_ID=""
on_exit() {
  local code=$?
  trap - EXIT
  if [[ -n "$CANDIDATE_ID" ]]; then docker rm -f "$CANDIDATE_ID" >/dev/null 2>&1 || true; fi
  if [[ "$code" -ne 0 && "$PROMOTION_STARTED" -eq 1 ]]; then
    echo "Deployment failed. Restoring previously running container..." >&2
    docker rm -f bgai-api >/dev/null 2>&1 || true
    if docker container inspect "$ROLLBACK" >/dev/null 2>&1; then
      docker rename "$ROLLBACK" bgai-api
      docker start bgai-api >/dev/null
      echo "Rollback container restarted. Recheck health and public Funnel." >&2
    fi
  fi
  exit "$code"
}
trap on_exit EXIT

# Probe exactly the new image locally; the candidate is NOT public.
CANDIDATE_ID="$(docker create --rm \
  --name "$CANDIDATE" \
  -p 127.0.0.1:8001:8000 \
  --env-file "$ENV_FILE" \
  -e PORT=8000 \
  -e ALPHAZERO_ENABLED="$ALPHAZERO_ENABLED" \
  -e EVALUATION_SOURCE_COMMIT="$SHA" \
  "$TAG")"
docker start "$CANDIDATE_ID" >/dev/null

wait_for_api() {
  local base="$1" attempt
  for attempt in $(seq 1 30); do
    if curl -fsS --connect-timeout 2 --max-time 3 "$base/v1/connect4/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}
verify_provenance() {
  local base="$1" actual
  actual="$(curl -fsS --max-time 5 "$base/v1/connect4/provenance" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("source_commit") or "")')"
  [[ "$actual" == "$SHA" ]]
}
wait_for_api http://127.0.0.1:8001
verify_provenance http://127.0.0.1:8001
python3 scripts/smoke-api-release.py --base http://127.0.0.1:8001 --mode "$MODE" --commit "$SHA"
docker rm -f "$CANDIDATE_ID" >/dev/null
CANDIDATE_ID=""

echo "Candidate checks passed. Swapping port 8000 container (active games will reset)."
docker rename bgai-api "$ROLLBACK"
PROMOTION_STARTED=1
docker stop "$ROLLBACK" >/dev/null
docker run -d \
  --name bgai-api \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  --env-file "$ENV_FILE" \
  -e PORT=8000 \
  -e ALPHAZERO_ENABLED="$ALPHAZERO_ENABLED" \
  -e EVALUATION_SOURCE_COMMIT="$SHA" \
  "$TAG" >/dev/null
wait_for_api http://127.0.0.1:8000
verify_provenance http://127.0.0.1:8000
python3 scripts/smoke-api-release.py --base http://127.0.0.1:8000 --mode "$MODE" --commit "$SHA"
PROMOTION_STARTED=0
echo "DEPLOY PASS: $SHA"
echo "Previous container remains stopped as $ROLLBACK."
echo "Verify Tailscale Funnel and public UI; then deliberately remove $ROLLBACK when stable."
