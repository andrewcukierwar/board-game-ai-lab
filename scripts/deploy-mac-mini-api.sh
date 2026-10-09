#!/usr/bin/env bash
# Reviewed ARM64 production update. Never pulls/merges or touches other worktrees.
# Keeps the prior container stopped as bgai-api-rollback for manual recovery.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_FILE="${BGAI_API_ENV_FILE:-$HOME/.config/board-game-ai-lab/api.env}"
CANDIDATE="bgai-api-candidate"
ROLLBACK="bgai-api-rollback"

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
docker info >/dev/null
SHA="$(git rev-parse HEAD)"
TAG="bgai-api:${SHA:0:12}"
echo "Building reviewed revision $SHA as $TAG (linux/arm64)"
docker buildx build --platform linux/arm64 --load -t "$TAG" -f docker/api.Dockerfile .
ARCH="$(docker image inspect "$TAG" --format '{{.Architecture}}')"
if [[ "$ARCH" != arm64 ]]; then
  echo "Unexpected image architecture: $ARCH" >&2
  exit 1
fi

PROMOTION_STARTED=0
on_exit() {
  local code=$?
  trap - EXIT
  docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
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
docker run -d --rm \
  --name "$CANDIDATE" \
  -p 127.0.0.1:8001:8000 \
  --env-file "$ENV_FILE" \
  -e PORT=8000 \
  -e EVALUATION_SOURCE_COMMIT="$SHA" \
  "$TAG" >/dev/null

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
curl -fsS --max-time 10 -X POST \
  http://127.0.0.1:8001/v1/connect4/start_game \
  -H 'Content-Type: application/json' \
  -d '{"player1":{"type":"human"},"player2":{"type":"negamax","depth":2}}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["revision"]==0 and r["game_id"]'
docker rm -f "$CANDIDATE" >/dev/null

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
  -e EVALUATION_SOURCE_COMMIT="$SHA" \
  "$TAG" >/dev/null
wait_for_api http://127.0.0.1:8000
verify_provenance http://127.0.0.1:8000
PROMOTION_STARTED=0
echo "DEPLOY PASS: $SHA"
echo "Previous container remains stopped as $ROLLBACK."
echo "Verify Tailscale Funnel and public UI; then deliberately remove $ROLLBACK when stable."
