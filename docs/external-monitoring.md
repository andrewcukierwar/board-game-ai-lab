# External monitoring — Board Game AI Lab (Mac Mini + Tailscale Funnel)

**Recommended provider:** [Better Stack Uptime](https://betterstack.com/uptime)
(independent external checks and email alerts). As of October 2026, the
personal-project free tier lists **10 monitors/heartbeats** and an uptime
monitoring interval of **3 minutes**. Verify terms while signing up; there
is no API token or provider account in this repository.

This is **not** a GitHub Actions cron job: an independent public Internet
probe still runs if the Mac Mini, Docker, Tailscale, GitHub Actions, or the
home network becomes unavailable.

## One-time setup in Better Stack's dashboard

1. From the **Mac Mini**, run `tailscale funnel status` and identify its
   public `https://<mac>.<tailnet>.ts.net` URL. Do not use the internal
   MagicDNS-only address or the old Render backend URL.
2. Sign in at <https://betterstack.com/uptime> and create an HTTP monitor:
   - **Name:** `Board Game AI Lab — Mac Mini API`
   - **URL:** `https://<mac>.<tailnet>.ts.net/v1/connect4/health`
   - **Request method:** GET
   - **Failure condition:** non-2xx status or endpoint unavailable (if keyword
     monitoring is available on the account, also check the body contains
     exactly `OK`).
   - **Interval:** 3 minutes on the available free tier.
   - **Timeout:** 15–30 seconds; don't require a sub-second response.
   - **Alerts:** enable and verify email to the account owner's address for
     both outage and recovery. Confirm the incident/escalation policy.
   No authentication headers, API keys or Tailscale tailnet identity are
   needed: Funnel exposes this HTTP endpoint to the public Internet.
3. Add a **separate** HTTP monitor:
   - **Name:** `Board Game AI Lab — Render UI`
   - **URL:** `https://board-game-ai-lab-ui.onrender.com/connect4`
   - **Failure condition:** non-2xx or unavailable; same interval/alerts.
   The UI check intentionally monitors static availability, not actual gameplay.
4. Verify both monitors show **Up**, and use the service's **test alert**
   feature or an isolated temporary *test monitor* for a guaranteed 404 to
   confirm email notification. Avoid intentionally turning off production
   Funnel just to test alarms.
5. Record the monitor names and links privately. Don't commit any Better
   Stack API tokens or notification addresses to the repository. If using a
   status page, decide deliberately whether to make it public.

Provider documentation:
- <https://betterstack.com/docs/uptime/monitoring-start/>
- <https://betterstack.com/docs/uptime/keyword-monitor/>
- <https://betterstack.com/pricing>

## What alerts mean

| Observation | Likely first check |
| --- | --- |
| Backend down; UI up | Tailscale Funnel/DNS, Docker Desktop, `bgai-api`, host connectivity/power |
| Backend up; UI down | Render static site or frontend routing |
| Both up, gameplay broken | Browser Network errors, backend JSON errors, CORS, missing session state, compute limits |
| Backend up, explanations fail | `EXPLANATIONS_ENABLED`, OpenAI settings/provider availability/quotas |
| Backend slow only under load | `docker stats bgai-api`, expensive agent concurrency, Mac Mini training pressure |

Quick checks from the Mac Mini:

```bash
docker ps --filter name=bgai-api
docker logs --tail 50 bgai-api
docker stats --no-stream bgai-api
curl -fsS --connect-timeout 3 --max-time 5 http://127.0.0.1:8000/v1/connect4/health
tailscale funnel status
```

Check the **Funnel URL from another device without a VPN-only route**, since
a successful localhost health check does not prove public access.

## Limitations

- `/health` returns `OK` if Flask responds. It does **not** run a move,
  validate GPT explanation provider access, or benchmark agent latency.
  Keep the public UI gameplay smoke checks after deployment.
- The Tailscale Funnel endpoint is public; avoid embedding API keys or
  authorization tokens in monitoring URLs.
- Uptime alerts do not automatically restart Docker/Tailscale; automatic
  startup and sleep prevention are configured separately on the Mac Mini.
- External monitoring is independent of CI. A green GitHub Actions build
  does not establish that this Mac Mini deployment is available.
- The old Render backend remains available only as a temporary rollback
  while it is retained; CI no longer updates its image.
