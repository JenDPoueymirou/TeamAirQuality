"""
prompt.py
---------
Builds the grounded system prompt that gets sent to the LLM on every request.

The system prompt does two things:
  1. Tells the model who it is and what rules it must follow
  2. Injects the retrieved data rows directly into the context

The grounding rules are the most important part. Without them, Llama 3.1
will happily answer from its training knowledge about NYC air quality —
which is real knowledge but NOT from your dataset and cannot be cited
or verified against your specific data.

Note on OpenRouter / OpenAI SDK:
  Unlike the Anthropic SDK which has a separate system= parameter,
  OpenRouter uses the OpenAI convention where the system prompt is
  the FIRST entry in the messages list with role="system".
  build_messages() handles this correctly.
"""

SYSTEM_TEMPLATE = """You are a data analyst for the NYC Air Pollution and Disease dataset.
This dataset covers all 5 New York City boroughs from 2005 to 2024, with
19,261 rows of neighborhood-level air quality and health outcome data.

The dataset tracks:
- Air pollutants: PM2.5 (mcg/m3), NO2 (ppb), Ozone (ppb), AQI
- Health outcomes: Asthma ER rates, asthma ED rates, cardiovascular hospitalization
  rates, cardiovascular ED rates, respiratory hospitalization rates,
  respiratory ED rates, PM2.5 attributable deaths (all per 100,000)
- Traffic: Annual truck vehicle miles traveled
- Geography: Borough, UHF neighborhood, ZIP code, time period

GROUNDING RULES — follow these on every response:
1. You may ONLY make factual claims that are directly supported by the
   retrieved rows shown below. Do not use outside knowledge for statistics,
   rates, or comparisons.
2. Every number or rate you state must be followed by its source in
   parentheses, like this: (Row 3)
3. If the user asks about something not covered by the retrieved rows,
   say exactly: "I don't have that data in my current context. Try asking
   with a specific borough, neighborhood, or year."
4. You MAY use general knowledge to explain what a metric means — for example,
   explaining what PM2.5 is or why ozone forms — but label it clearly as
   background context, not dataset findings.
5. Keep answers to 4-6 sentences for now. Be direct and lead with the
   key finding.

RETRIEVED DATA ROWS:
{rows}

Answer the user's question using only the rows above."""


def build_system_prompt(chunks: list[str]) -> str:
    """
    Inject retrieved row chunks into the system prompt template.

    If no rows were retrieved (empty dataset query or retrieval failure),
    the model is told explicitly so it doesn't guess.
    """
    if not chunks:
        rows_text = "No matching rows were retrieved for this query."
    else:
        rows_text = "\n".join(chunks)

    return SYSTEM_TEMPLATE.format(rows=rows_text)


def build_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """
    Assemble the full messages list for the OpenRouter / OpenAI API call.

    Structure:
      [system message]         ← grounded system prompt with injected rows
      [history message 0]      ← previous turns (if any)
      [history message 1]
      ...
      [current user message]   ← what the user just asked

    The system prompt is injected fresh on EVERY request with the rows
    retrieved for that specific question. This means the model always
    sees the most relevant data for what was just asked, not leftover
    context from a previous turn.

    History carries the conversation flow but the grounding always
    reflects the current question.
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Append previous conversation turns
    # Validate each entry has role and content before including
    for turn in history:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            if turn["role"] in ("user", "assistant"):
                messages.append({
                    "role": turn["role"],
                    "content": str(turn["content"])
                })

    # Append the current user message
    messages.append({
        "role": "user",
        "content": user_message
    })

    return messages