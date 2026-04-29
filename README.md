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

