LLM Day Planner
Turn messy natural‑language day descriptions into a minute‑by‑minute plan.
An LLM (Groq / Llama 3.1 8B) extracts wake‑up time, sleep, fixed events, and flexible tasks. A deterministic greedy planner then builds a complete schedule, respecting priorities, time windows, and constraints.

🗣️ Input: “wake 7, gym 1h, meeting 3pm, sleep 23”

📅 Output: a full DayPlan with placed tasks, fixed events, free gaps, and unscheduled items with reasons

🔄 Provider‑agnostic: swap Groq for any OpenAI‑compatible API (Azure, Ollama, vLLM) by changing one config block

✅ Pydantic‑validated pipeline

🧪 Deterministic and unit‑testable planner

Perfect for prototyping scheduling assistants, personal day organisers, or as a base for smarter LLM‑driven planning.

*Keywords: scheduling, LLM, groq, llama3, pydantic, greedy, planner, natural-language-processing*

## Files & what each does

| File | Role |
|------|------|
| `models.py` | **Contract / JSON schema** - Pydantic models for `Schedule`, `FixedEvent`, `Task`, etc. The single source of truth that the parser and planner both depend on. |
| `parser.py` | **LLM parser** - Calls Groq (or any OpenAI‑compatible API) with a structured prompt to convert messy user text into a validated `Schedule` object. Handles retries on validation failure. |
| `planner.py` | **Greedy day planner** - Takes a `Schedule` and fits all tasks into a minute‑by‑minute timeline, respecting priorities, preferred time windows, and fixed events. Returns a `DayPlan` with placed tasks, free gaps, and unscheduled items (with reasons). |
| `pipeline.py` | **End‑to‑end pipeline** - Wires the parser and planner together. `run("wake 7, gym 1h, meeting 3pm, sleep 23")` returns a `PipelineResult` containing the parse result and the final `DayPlan`. |
| `main.py` | **Demo script** - Quick end‑to‑end example. Runs the pipeline and prints a human‑readable plan. Can be run with a command‑line query or uses built‑in demos. |
| `test_parser.py` | **Parser unit tests** - Runs 10+ natural‑language inputs through the LLM parser and checks that the output contains the expected fields (fixed events, tasks, ambiguities, etc.). |
| `test_planner.py` | **Planner unit tests** - Fully deterministic tests for the greedy planner. Feeds hand‑crafted `Schedule` objects and verifies correct placement, priority order, fallback behaviour, and no overlaps. |

## How to use

### 1. Install dependencies
```bash
pip install pydantic openai
```

### 2. Set your API key
Get a free Groq API key from console.groq.com.
Then export it as an environment variable:

```bash
export GROQ_API_KEY="gsk_your_key_here"        # Linux/macOS
set GROQ_API_KEY="gsk_your_key_here"           # Windows cmd
$env:GROQ_API_KEY="gsk_your_key_here"          # PowerShell
```
To use a different provider (Azure, Ollama, vLLM), just edit the LLMConfig dataclass in parser.py - no parser code changes needed.

### 3. Run the demo
```bash
python main.py "wake 7, gym 1h, meeting 3pm, dinner 19, sleep 23"
```
Or pass multiple queries, or none to use the built‑in DEMO_QUERIES.

### 4. Use in your own code
```python
from pipeline import run

pipeline_result = run("wake 8am, yoga 45min, dentist 12pm, sleep 10pm")
if pipeline_result.success:
    print(pipeline_result.plan.summary())
else:
    print("Parsing failed:", pipeline_result.parse.errors)
```
Or use the pieces separately:

```python
from parser import ScheduleParser
from planner import GreedyPlanner

parser = ScheduleParser()
result = parser.parse("wake 8am, yoga 45min, dentist 12pm, sleep 10pm")

if result.success:
    planner = GreedyPlanner()
    plan = planner.plan(result.schedule)
    print(plan.summary())
else:
    print("Parsing failed:", result.errors)
```
### 5. Run tests
```bash
python test_parser.py           # basic run
python test_parser.py --verbose # detailed output
python test_parser.py --case 3  # run only case 3

python test_planner.py          # basic run
python test_planner.py -v       # verbose
```
