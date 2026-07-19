"""
agent_v2.py  —  AI Essay Grading Agent (Multi-Agent Decision Layer)
IT9204 | Student 202507410 Faiza Malik

Multi-agent architecture:
  RouterAgent    → classifies query, selects sub-agent
  ScoringAgent   → predict_score only
  AdviceAgent    → retrieve_rubric only
  StatsAgent     → get_essay_statistics only
  GradingAgent   → all three tools, full structured feedback

Each agent returns a typed ResponseEnvelope so the UI
renders only the sections relevant to that query type.

Before running:
  1. Start FastAPI: python main.py
  2. $env:GROQ_API_KEY="gsk_..."
  3. python agent_v2.py
"""

import os
import json
import requests
import dataclasses
from typing import Optional, List
from dataclasses import dataclass, field
import pandas as pd
from groq import Groq

from rag_tool import retrieve_rubric_criteria


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FASTAPI_URL  = "http://127.0.0.1:8000/predict"

if not GROQ_API_KEY:
    raise SystemExit(
        "ERROR: GROQ_API_KEY not set.\n"
        "  PowerShell: $env:GROQ_API_KEY='gsk_...'\n"
        "  Then run: python agent_v2.py"
    )

client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Response Envelope — typed output that drives conditional UI rendering
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ResponseEnvelope:
    """
    Each field maps to exactly one UI section.
    Only populated fields are rendered — the UI must check for None.
    """
    agent_type:   str = ""          # "scoring" | "advice" | "stats" | "grading"

    # ── ScoringAgent ──────────────────────────────────────────────────────────
    score:        Optional[float] = None   # 0-10
    raw_score:    Optional[float] = None   # 0-6 approx
    band:         Optional[str]   = None   # Emerging/Beginning/…/Exemplary
    model_used:   Optional[str]   = None
    word_count:   Optional[int]   = None

    # ── GradingAgent (extends scoring) ────────────────────────────────────────
    summary:      Optional[str]       = None
    strengths:    Optional[str]       = None
    improvements: Optional[str]       = None
    next_steps:   Optional[List[str]] = None

    # ── AdviceAgent ───────────────────────────────────────────────────────────
    rubric_criteria: Optional[List[str]] = None
    advice_text:     Optional[str]       = None

    # ── StatsAgent ────────────────────────────────────────────────────────────
    stats: Optional[dict] = None

    # ── Shared ────────────────────────────────────────────────────────────────
    error: Optional[str] = None


def envelope_to_dict(env: ResponseEnvelope) -> dict:
    return dataclasses.asdict(env)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations (shared across all agents)
# ─────────────────────────────────────────────────────────────────────────────
def predict_score(essay_text: str) -> dict:
    essay = essay_text.strip()
    if not essay:
        return {"error": "essay_text is empty"}
    if len(essay.split()) < 5:
        return {"error": "Essay too short"}
    try:
        response = requests.post(
            FASTAPI_URL,
            json    = {"essay_text": essay},
            timeout = 15,
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "FastAPI not running. Start with: python main.py"}
    except Exception as e:
        return {"error": str(e)}


def retrieve_rubric(query: str) -> dict:
    try:
        criteria = retrieve_rubric_criteria(query, k=3)
        return {"criteria": criteria, "query_used": query}
    except Exception as e:
        return {"error": str(e)}


_asap_cache = None

def _load_asap():
    global _asap_cache
    if _asap_cache is not None:
        return _asap_cache
    for fname in ["data/asap_train.csv",
                  "data/ASAP_2_Final_github_train.csv",
                  "data/train.csv"]:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                df = df.rename(columns={"full_text": "essay_text",
                                        "score": "raw_score"})
                df["raw_score"]  = pd.to_numeric(df["raw_score"], errors="coerce")
                df["essay_text"] = df["essay_text"].astype(str)
                df = df.dropna(subset=["essay_text", "raw_score"]).reset_index(drop=True)
                df["score"]      = (df["raw_score"] - 1) / 5 * 10
                df["word_count"] = df["essay_text"].apply(lambda t: len(t.split()))
                _asap_cache = df
                return df
            except Exception:
                continue
    return None


def get_essay_statistics(query_type: str, score_value: float = None) -> dict:
    df = _load_asap()
    if df is None:
        return {"error": "ASAP dataset not found in data/ folder."}

    if query_type == "distribution":
        bands = {
            "1-2 (Emerging)"   : int((df["score"] <= 2).sum()),
            "3-4 (Beginning)"  : int(((df["score"] > 2) & (df["score"] <= 4)).sum()),
            "5-6 (Developing)" : int(((df["score"] > 4) & (df["score"] <= 6)).sum()),
            "7-8 (Proficient)" : int(((df["score"] > 6) & (df["score"] <= 8)).sum()),
            "9-10 (Exemplary)" : int((df["score"] > 8).sum()),
        }
        return {
            "total_essays"  : len(df),
            "mean_score"    : round(float(df["score"].mean()), 2),
            "median_score"  : round(float(df["score"].median()), 2),
            "score_bands"   : bands,
            "interpretation": (
                f"Average score is {round(float(df['score'].mean()),1)}/10. "
                f"Middle 50% score between "
                f"{round(float(df['score'].quantile(0.25)),1)} and "
                f"{round(float(df['score'].quantile(0.75)),1)}."
            ),
        }

    elif query_type == "examples" and score_value is not None:
        tol      = 0.75
        examples = df[
            (df["score"] >= score_value - tol) &
            (df["score"] <= score_value + tol)
        ].nlargest(2, "word_count")
        if examples.empty:
            return {"message": f"No essays found near score {score_value}"}
        result = []
        for _, row in examples.iterrows():
            preview = row["essay_text"][:300] + (
                "..." if len(row["essay_text"]) > 300 else ""
            )
            result.append({
                "score"     : round(float(row["score"]), 2),
                "word_count": int(row["word_count"]),
                "preview"   : preview,
            })
        return {"score_requested": score_value, "examples": result}

    elif query_type == "word_stats":
        df["band"] = pd.cut(
            df["score"],
            bins=[0, 2, 4, 6, 8, 10],
            labels=["1-2", "3-4", "5-6", "7-8", "9-10"],
        )
        stats = (
            df.groupby("band", observed=True)["word_count"]
            .agg(["mean", "median"]).round(0)
        )
        return {
            "avg_words_by_band": {
                str(band): {
                    "mean_words"  : int(row["mean"]),
                    "median_words": int(row["median"]),
                }
                for band, row in stats.iterrows()
            },
            "insight": "Higher-scoring essays tend to be longer but quality matters more.",
        }

    return {"error": f"Unknown query_type '{query_type}'."}


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (reusable across agents)
# ─────────────────────────────────────────────────────────────────────────────
TOOL_PREDICT = {
    "type": "function",
    "function": {
        "name": "predict_score",
        "description": (
            "Predict a numeric essay score using the trained ML model. "
            "The score is deterministic — never estimate or modify it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "essay_text": {
                    "type"       : "string",
                    "description": "The complete text of the student essay.",
                }
            },
            "required": ["essay_text"],
        },
    },
}

TOOL_RUBRIC = {
    "type": "function",
    "function": {
        "name": "retrieve_rubric",
        "description": (
            "Retrieve rubric criteria and source texts from the knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type"       : "string",
                    "description": "Natural language query to find rubric criteria.",
                }
            },
            "required": ["query"],
        },
    },
}

TOOL_STATS = {
    "type": "function",
    "function": {
        "name": "get_essay_statistics",
        "description": "Query ASAP 2.0 dataset for score context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["distribution", "examples", "word_stats"],
                },
                "score_value": {
                    "type"       : "number",
                    "description": "Score 0-10 for finding example essays.",
                },
            },
            "required": ["query_type"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: score → band label
# ─────────────────────────────────────────────────────────────────────────────
def score_to_band(score: float) -> str:
    if score <= 2:  return "Emerging"
    if score <= 4:  return "Beginning"
    if score <= 6:  return "Developing"
    if score <= 8:  return "Proficient"
    return "Exemplary"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one agentic Groq loop
# ─────────────────────────────────────────────────────────────────────────────
def _run_groq_loop(
    system_prompt: str,
    user_message:  str,
    tools:         list,
    verbose:       bool = True,
    max_iter:      int  = 8,
) -> str:
    """
    Runs a Groq agentic loop with the given tools.
    Returns the final text content from the model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    TOOL_MAP = {
        "predict_score"       : predict_score,
        "retrieve_rubric"     : retrieve_rubric,
        "get_essay_statistics": get_essay_statistics,
    }

    for iteration in range(max_iter):
        if verbose:
            print(f"  [loop iter {iteration + 1}]")

        response = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = messages,
            tools       = tools,
            tool_choice = "auto",
            max_tokens  = 2048,
        )

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            return msg.content or ""

        if finish_reason == "tool_calls" and msg.tool_calls:
            messages.append({
                "role"      : "assistant",
                "content"   : msg.content or "",
                "tool_calls": [
                    {
                        "id"      : tc.id,
                        "type"    : "function",
                        "function": {
                            "name"     : tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    tool_args = {}

                if verbose:
                    print(f"    → {tool_name}({str(tool_args)[:80]})")

                fn     = TOOL_MAP.get(tool_name, lambda **_: {"error": "unknown tool"})
                result = fn(**tool_args) if tool_args else fn()

                if verbose:
                    print(f"    ← {str(result)[:100]}")

                messages.append({
                    "role"        : "tool",
                    "tool_call_id": tc.id,
                    "content"     : json.dumps(result),
                })
        else:
            break

    return "Agent could not complete the request."


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Agent 1: ScoringAgent
# Tools: predict_score only
# Output: score, band, model_used, word_count
# ─────────────────────────────────────────────────────────────────────────────
class ScoringAgent:
    SYSTEM = """You are a scoring-only agent. Your ONLY job:
1. Call predict_score with the essay text.
2. Return ONLY the following JSON (no markdown, no extra text):

{
  "score": <float 0-10>,
  "raw_score": <float 0-6>,
  "band": "<Emerging|Beginning|Developing|Proficient|Exemplary>",
  "model_used": "<model name from tool result>",
  "word_count": <integer>
}

Do NOT add summaries, feedback, or any other fields.
The score must come EXACTLY from the predict_score tool — never invent one."""

    def run(self, essay_text: str, verbose: bool = True) -> ResponseEnvelope:
        env = ResponseEnvelope(agent_type="scoring")
        try:
            raw = _run_groq_loop(
                system_prompt = self.SYSTEM,
                user_message  = essay_text,
                tools         = [TOOL_PREDICT],
                verbose       = verbose,
            )
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            env.score      = data.get("score")
            env.raw_score  = data.get("raw_score")
            env.band       = data.get("band") or score_to_band(env.score or 0)
            env.model_used = data.get("model_used", "ML Model")
            env.word_count = data.get("word_count")
        except Exception as e:
            env.error = f"ScoringAgent error: {e}"
        return env


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Agent 2: AdviceAgent
# Tools: retrieve_rubric only
# Output: rubric_criteria, advice_text
# ─────────────────────────────────────────────────────────────────────────────
class AdviceAgent:
    SYSTEM = """You are a writing-advice agent. Your ONLY job:
1. Call retrieve_rubric with a suitable query.
2. Return ONLY the following JSON (no markdown, no extra text):

{
  "rubric_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
  "advice_text": "<3-5 sentences of concrete writing advice based on rubric>"
}

Do NOT mention or produce any numeric scores."""

    def run(self, user_query: str, verbose: bool = True) -> ResponseEnvelope:
        env = ResponseEnvelope(agent_type="advice")
        try:
            raw = _run_groq_loop(
                system_prompt = self.SYSTEM,
                user_message  = user_query,
                tools         = [TOOL_RUBRIC],
                verbose       = verbose,
            )
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            env.rubric_criteria = data.get("rubric_criteria", [])
            env.advice_text     = data.get("advice_text", "")
        except Exception as e:
            env.error = f"AdviceAgent error: {e}"
        return env


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Agent 3: StatsAgent
# Tools: get_essay_statistics only
# Output: stats dict
# ─────────────────────────────────────────────────────────────────────────────
class StatsAgent:
    SYSTEM = """You are a dataset statistics agent. Your ONLY job:
1. Call get_essay_statistics with query_type="distribution" (default) or
   "word_stats" or "examples" (with score_value) as appropriate.
2. Return ONLY the following JSON (no markdown, no extra text):

{
  "stats": { <exact data returned by the tool> }
}"""

    def run(self, user_query: str, verbose: bool = True) -> ResponseEnvelope:
        env = ResponseEnvelope(agent_type="stats")
        try:
            raw = _run_groq_loop(
                system_prompt = self.SYSTEM,
                user_message  = user_query,
                tools         = [TOOL_STATS],
                verbose       = verbose,
            )
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            env.stats = data.get("stats", {})
        except Exception as e:
            env.error = f"StatsAgent error: {e}"
        return env


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Agent 4: GradingAgent
# Tools: all three, in sequence
# Output: full envelope (score + grading fields)
# ─────────────────────────────────────────────────────────────────────────────
class GradingAgent:
    SYSTEM = """You are a full essay grading agent. Your job:
1. Call predict_score → get the numeric score.
2. Call retrieve_rubric with a query tailored to the score level.
3. Call get_essay_statistics with query_type="distribution" for context.
4. Return ONLY the following JSON (no markdown, no extra text):

{
  "score": <float>,
  "raw_score": <float>,
  "band": "<band>",
  "model_used": "<model>",
  "word_count": <int>,
  "summary": "<2-3 sentence overall assessment>",
  "strengths": "<what the essay does well, rubric-aligned>",
  "improvements": "<specific, actionable suggestions>",
  "next_steps": ["<step 1>", "<step 2>", "<step 3>"],
  "stats": { <distribution data from tool> }
}

CRITICAL: score must come EXACTLY from predict_score — never invent it."""

    def run(self, essay_text: str, verbose: bool = True) -> ResponseEnvelope:
        env = ResponseEnvelope(agent_type="grading")
        try:
            raw = _run_groq_loop(
                system_prompt = self.SYSTEM,
                user_message  = essay_text,
                tools         = [TOOL_PREDICT, TOOL_RUBRIC, TOOL_STATS],
                verbose       = verbose,
            )
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            env.score        = data.get("score")
            env.raw_score    = data.get("raw_score")
            env.band         = data.get("band") or score_to_band(env.score or 0)
            env.model_used   = data.get("model_used", "ML Model")
            env.word_count   = data.get("word_count")
            env.summary      = data.get("summary")
            env.strengths    = data.get("strengths")
            env.improvements = data.get("improvements")
            env.next_steps   = data.get("next_steps", [])
            env.stats        = data.get("stats")
        except Exception as e:
            env.error = f"GradingAgent error: {e}"
        return env


# ─────────────────────────────────────────────────────────────────────────────
# RouterAgent — classifies query → delegates to correct sub-agent
# ─────────────────────────────────────────────────────────────────────────────
class RouterAgent:
    """
    Lightweight LLM-based router. Uses a small model for classification only.
    Falls back to keyword heuristics if LLM call fails.
    """

    # Keyword heuristics (fast path, no API call needed)
    _SCORE_ONLY_KEYS  = {"what score", "score would", "score this", "quick score",
                         "give me a score", "just the score"}
    _ADVICE_KEYS      = {"rubric", "criteria", "writing advice", "how to write",
                         "what makes", "tips", "improve my writing", "writing tips"}
    _STATS_KEYS       = {"statistics", "distribution", "average score", "dataset",
                         "word stats", "how many essays", "score breakdown"}
    _GRADING_KEYS     = {"grade this", "grade my", "full feedback", "full grading",
                         "feedback", "strengths", "improvements", "comparison",
                         "assess this", "evaluate this"}

    def _keyword_classify(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(k in t for k in self._STATS_KEYS):
            return "stats"
        if any(k in t for k in self._ADVICE_KEYS) and len(text.split()) < 60:
            return "advice"
        if any(k in t for k in self._SCORE_ONLY_KEYS):
            return "scoring"
        if any(k in t for k in self._GRADING_KEYS):
            return "grading"
        return None

    def _llm_classify(self, text: str) -> str:
        """Ask the LLM to classify — used when keywords are ambiguous."""
        prompt = f"""Classify this user message into exactly ONE category.
Reply with ONLY the category word, nothing else.

Categories:
  scoring  → user wants only a numeric score for an essay
  grading  → user wants full feedback (score + strengths + improvements)
  advice   → user wants writing tips or rubric info, no essay submitted
  stats    → user wants dataset statistics or score distributions

User message:
\"\"\"{text[:500]}\"\"\"

Category:"""
        try:
            response = client.chat.completions.create(
                model      = "llama-3.1-8b-instant",
                messages   = [{"role": "user", "content": prompt}],
                max_tokens = 10,
                tools      = [],          # no tools for router
            )
            return response.choices[0].message.content.strip().lower()
        except Exception:
            return "grading"  # safe default

    def route(self, user_message: str, verbose: bool = True) -> ResponseEnvelope:
        # Fast keyword path
        category = self._keyword_classify(user_message)

        # Slow LLM path for ambiguous messages
        if category is None:
            category = self._llm_classify(user_message)

        if verbose:
            print(f"\n{'═'*60}")
            print(f"  RouterAgent → '{category}' agent selected")
            print(f"{'═'*60}")

        if category == "scoring":
            return ScoringAgent().run(user_message, verbose)
        elif category == "advice":
            return AdviceAgent().run(user_message, verbose)
        elif category == "stats":
            return StatsAgent().run(user_message, verbose)
        else:
            return GradingAgent().run(user_message, verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — single entry point for the UI / FastAPI endpoint
# ─────────────────────────────────────────────────────────────────────────────
router = RouterAgent()

def grade_essay(user_message: str, verbose: bool = True) -> dict:
    """
    Main entry point.
    Returns a dict (serialisable ResponseEnvelope).
    The UI should inspect 'agent_type' and render only the relevant sections.

    Rendering contract:
      agent_type == "scoring"  → show: score, band, model_used, word_count
      agent_type == "advice"   → show: rubric_criteria, advice_text
      agent_type == "stats"    → show: stats
      agent_type == "grading"  → show: all scoring fields + summary/strengths/
                                        improvements/next_steps + stats
    """
    envelope = router.route(user_message, verbose=verbose)
    return envelope_to_dict(envelope)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper for CLI output
# ─────────────────────────────────────────────────────────────────────────────
def _pretty_print(result: dict):
    atype = result.get("agent_type", "unknown")
    print(f"\n{'─'*60}")
    print(f"  Agent type : {atype.upper()}")
    print(f"{'─'*60}")

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    if atype == "scoring":
        print(f"  Score      : {result['score']} / 10.0")
        print(f"  Raw approx : {result['raw_score']} / 6")
        print(f"  Band       : {result['band']}")
        print(f"  Model      : {result['model_used']}")
        print(f"  Words      : {result['word_count']}")

    elif atype == "advice":
        print("  Rubric Criteria:")
        for c in (result.get("rubric_criteria") or []):
            print(f"    • {c}")
        print(f"\n  Advice:\n  {result.get('advice_text', '')}")

    elif atype == "stats":
        import pprint
        pprint.pprint(result.get("stats", {}))

    elif atype == "grading":
        print(f"  Score      : {result['score']} / 10.0  (~{result['raw_score']} / 6)")
        print(f"  Band       : {result['band']}")
        print(f"  Words      : {result['word_count']}")
        print(f"\n  Summary:\n  {result.get('summary', '')}")
        print(f"\n  Strengths:\n  {result.get('strengths', '')}")
        print(f"\n  Improvements:\n  {result.get('improvements', '')}")
        steps = result.get("next_steps") or []
        if steps:
            print("\n  Next Steps:")
            for i, s in enumerate(steps, 1):
                print(f"    {i}. {s}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo — four query paths
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  IT9204 — AI Essay Grading Agent v2")
    print("  Multi-Agent Decision Layer")
    print("  Student: 202507410 Faiza Malik")
    print("=" * 60)

    # ── Path 1: Score only ────────────────────────────────────────────────────
    print("\n\nPATH 1 — Score only (ScoringAgent)")
    r1 = grade_essay(
        "What score would this essay receive?\n\n"
        "The water cycle is very important. Water evaporates from the "
        "ocean when the sun heats it. Then it becomes clouds and rains. "
        "The water goes back to the ocean and the cycle starts again.",
    )
    _pretty_print(r1)

    # ── Path 2: Writing advice ────────────────────────────────────────────────
    print("\n\nPATH 2 — Writing advice (AdviceAgent)")
    r2 = grade_essay(
        "What are the rubric criteria for a strong essay? "
        "What makes thesis and evidence effective?"
    )
    _pretty_print(r2)

    # ── Path 3: Dataset statistics ────────────────────────────────────────────
    print("\n\nPATH 3 — Statistics (StatsAgent)")
    r3 = grade_essay("Show me the score distribution statistics for the dataset.")
    _pretty_print(r3)

    # ── Path 4: Full grading ──────────────────────────────────────────────────
    print("\n\nPATH 4 — Full grading (GradingAgent)")
    r4 = grade_essay(
        "Please grade this essay fully with feedback:\n\n"
        "The widespread adoption of artificial intelligence in schools presents "
        "both significant opportunities and serious challenges that educators "
        "must carefully evaluate. AI-powered adaptive learning tools can "
        "personalise instruction to individual student needs, identifying "
        "gaps in understanding and adjusting difficulty in real time. "
        "Research suggests students receiving personalised feedback improve "
        "more rapidly than those in traditional settings. However, integrating "
        "AI raises important equity questions. Schools in under-resourced "
        "communities may lack the infrastructure needed, potentially widening "
        "achievement gaps. A thoughtful approach would harness AI benefits "
        "while preserving the irreplaceable human elements of teaching."
    )
    _pretty_print(r4)

    print("\n\n" + "=" * 60)
    print("  All four paths complete.")
    print("  Each result dict has 'agent_type' — render only relevant UI sections.")
    print("=" * 60)