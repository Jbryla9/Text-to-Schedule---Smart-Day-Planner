# Text-to-Schedule — Smart Day Planner

A small FastAPI application that converts natural-language plans into a validated,
conflict-aware daily schedule. It uses Groq-hosted `openai/gpt-oss-20b` with strict
JSON Schema output, Pydantic validation, and a deterministic greedy planner.

## Features

- `POST /schedule` natural-language scheduling endpoint
- `GET /health` health check
- Interactive Swagger UI at `/docs`
- Explicit overlapping-event conflicts and unscheduled-task reasons
- Controlled errors for missing keys, provider failures, and invalid schedules
- Local rotating application logs and privacy-conscious JSONL metrics
- Optional CLI using the same parser and planner

## Local setup (PowerShell)

```powershell
Set-Location "C:\Users\kubab\OneDrive\Pulpit\Projekty\docker_git_smart_scheduler"
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If the existing OneDrive-synchronized `.venv` reports `Access denied`, recreate it:

```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `.env` and set a newly generated Groq key:

```env
GROQ_API_KEY=<your-new-key>
SMART_SCHEDULER_MODEL=openai/gpt-oss-20b
```

Do not reuse the key that was previously stored in the project. Rotate it in the
Groq console because plaintext copies existed locally.

Start the API:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Open:

- API overview: <http://127.0.0.1:8000/>
- Health check: <http://127.0.0.1:8000/health>
- Interactive API: <http://127.0.0.1:8000/docs>

## Create a schedule

PowerShell example:

```powershell
$body = @{
    text = "Wake at 07:00, standup at 10:00 for 30 minutes, deep work for 3 hours high priority, gym at 18:00, sleep at 23:00"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/schedule" `
    -ContentType "application/json" `
    -Body $body
```

The response contains the provider model, latency, token usage, validated input
structure, planned slots, conflicts, warnings, and unscheduled tasks.

## CLI

```powershell
python -m app.cli "Wake at 07:00, meeting at 10:00 for 30 minutes, project work for 2 hours, sleep at 23:00"
```

## Monitoring

Runtime files are written under `logs/`:

- `application.log` — rotating operational log, up to 1 MB with three backups
- `events.jsonl` — one compact event per scheduling run

Event records include model, latency, attempts, token counts, success, validation
error count, number of tasks, conflicts, and unscheduled tasks. User schedule text,
raw model output, and API keys are deliberately excluded.

Optional environment settings:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Required Groq credential |
| `SMART_SCHEDULER_MODEL` | `openai/gpt-oss-20b` | Groq model ID |
| `SMART_SCHEDULER_LOG_LEVEL` | `INFO` | Python log level |
| `SMART_SCHEDULER_LOG_DIR` | `logs` | Local log directory |

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use fake provider responses and never call Groq.

## Docker (optional)

```powershell
docker build -t smart-scheduler .
docker run --rm -p 8000:8000 --env-file .env smart-scheduler
```

The container runs as an unprivileged user and exposes port `8000`.

## API behavior

| Status | Meaning |
|---|---|
| `200` | Schedule created successfully |
| `422` | Empty request, schema error, or schedule could not be validated |
| `502` | Groq provider request failed |
| `503` | `GROQ_API_KEY` is missing |
