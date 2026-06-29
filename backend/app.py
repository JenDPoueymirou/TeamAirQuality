"""
app.py
------
Streamlit frontend for the NYC Air Quality Chatbot.

Run locally:  streamlit run backend/app.py
Deploy:       set API_URL env var to your Render service URL,
              then deploy this file as a separate Render Web Service with
              start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NYC Air Quality Chatbot",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "").rstrip("/")


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_borough_stats(base: str) -> dict:
    r = requests.get(f"{base}/stats/borough", timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_correlations(base: str) -> dict:
    r = requests.get(f"{base}/stats/correlations", timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_health(base: str) -> dict:
    r = requests.get(f"{base}/health", timeout=8)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_usage(base: str) -> dict:
    r = requests.get(f"{base}/usage/summary", timeout=8)
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("NYC Air Quality")
    st.caption("Bloomberg Hackathon · Gemini 2.0 Flash")
    st.divider()

    url_input = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        placeholder="https://your-service.onrender.com",
        help="Your deployed Render service URL",
    )
    if url_input.rstrip("/") != st.session_state.api_url:
        st.session_state.api_url = url_input.rstrip("/")
        st.cache_data.clear()
        st.rerun()

    base = st.session_state.api_url

    if base:
        try:
            health = fetch_health(base)
            st.success(f"Live — {health.get('csv_rows', 0):,} rows loaded")
            st.caption(f"LLM: {health.get('llm_model', 'n/a')}")
        except requests.exceptions.Timeout:
            st.warning("API is waking up — try again in 30 s")
        except Exception:
            st.error("Cannot reach API — check the URL above")

        st.divider()

        try:
            usage = fetch_usage(base)
            today = usage.get("requests_today", 0)
            limit = usage.get("daily_limit", 1500)
            remaining = usage.get("requests_remaining", limit)
            st.metric("Requests today", today, delta=f"{remaining} remaining")
            st.progress(min(today / limit, 1.0))
        except Exception:
            pass

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_stats, tab_about = st.tabs(["Chat", "Statistics", "About"])


# ── Chat ──────────────────────────────────────────────────────────────────────

with tab_chat:
    st.header("NYC Air Quality Chatbot")
    st.caption(
        "Ask about air pollution and health outcomes across NYC's five boroughs (2005–2024). "
        "Every answer cites the dataset rows it draws from."
    )

    # Render existing conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Sources & filters"):
                    m = msg["meta"]
                    c1, c2 = st.columns(2)
                    c1.metric("Rows retrieved", m.get("rows_retrieved", 0))
                    c2.metric("Citations", "Valid" if m.get("citation_valid") else "Missing")
                    if m.get("filters_applied"):
                        st.json(m["filters_applied"])

    # New message input
    if user_msg := st.chat_input("Ask about NYC air quality…"):
        if not st.session_state.api_url:
            st.error("Enter your API URL in the sidebar first.")
            st.stop()

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Build history (all turns except the one just added)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # Call API and stream response
        with st.chat_message("assistant"):
            with st.spinner("Searching dataset and generating answer…"):
                try:
                    r = requests.post(
                        f"{st.session_state.api_url}/chat",
                        json={"message": user_msg, "history": history},
                        timeout=120,
                    )

                    if r.status_code == 200:
                        d = r.json()
                        st.markdown(d["answer"])

                        meta = {
                            "rows_retrieved": d.get("rows_retrieved"),
                            "citation_valid": d.get("citation_valid"),
                            "filters_applied": d.get("filters_applied"),
                        }
                        with st.expander("Sources & filters"):
                            c1, c2 = st.columns(2)
                            c1.metric("Rows retrieved", d.get("rows_retrieved", 0))
                            c2.metric("Citations", "Valid" if d.get("citation_valid") else "Missing")
                            if d.get("filters_applied"):
                                st.json(d["filters_applied"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": d["answer"],
                            "meta": meta,
                        })

                    elif r.status_code == 429:
                        st.error(
                            "Rate limit reached. "
                            "The free tier allows 15 requests per minute — wait a moment and try again."
                        )
                    else:
                        st.error(f"API error {r.status_code}: {r.text[:300]}")

                except requests.exceptions.Timeout:
                    st.warning(
                        "Request timed out. If this is the first chat after a cold start, "
                        "the server is loading the embedding model — wait 30 seconds and try again."
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API. Check the URL in the sidebar.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ── Statistics ────────────────────────────────────────────────────────────────

with tab_stats:
    st.header("NYC Air Quality & Health Statistics")

    if not st.session_state.api_url:
        st.info("Enter your API URL in the sidebar to load statistics.")
        st.stop()

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Per-Borough Averages")
        try:
            bdata = fetch_borough_stats(st.session_state.api_url)
            df = pd.DataFrame(bdata["boroughs"]).set_index("borough")
            numeric_cols = df.select_dtypes("number").columns.tolist()

            metric = st.selectbox(
                "Metric",
                options=numeric_cols,
                format_func=lambda x: x.replace("_", " ").title(),
            )
            chart_df = df[[metric]].dropna().sort_values(metric, ascending=False)
            st.bar_chart(chart_df, height=380)

            with st.expander("Raw data"):
                st.dataframe(df[numeric_cols].round(2), use_container_width=True)

        except Exception as e:
            st.error(f"Could not load borough stats: {e}")

    with col_right:
        st.subheader("Pollutant — Health Correlations (citywide)")
        try:
            cdata = fetch_correlations(st.session_state.api_url)
            if "citywide" in cdata:
                corr_df = pd.DataFrame(cdata["citywide"]).round(2)

                keep = [c for c in [
                    "pm25", "no2", "ozone",
                    "asthma_er_rate", "cardiovascular_hosp_rate",
                    "respiratory_hosp_rate", "pm25_deaths",
                ] if c in corr_df.columns]

                if keep:
                    corr_df = corr_df.loc[keep, keep]

                st.dataframe(
                    corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                    use_container_width=True,
                    height=380,
                )
                st.caption(
                    "Color scale: green = strong positive correlation, "
                    "red = strong negative correlation."
                )
        except Exception as e:
            st.error(f"Could not load correlations: {e}")


# ── About ─────────────────────────────────────────────────────────────────────

with tab_about:
    st.header("About This Project")

    st.markdown("""
**Research question:** Which NYC communities bear a disproportionate pollution burden,
and how does that burden correlate with measurable health outcomes?

The dataset covers **2005–2024** at UHF42 neighborhood granularity — the same geography
used by NYC DOHMH for public health surveillance. Each row links air quality measurements
(PM2.5, NO2, ozone, AQI) to health outcomes (asthma ER rates, cardiovascular hospitalization
rates, PM2.5-attributable deaths) and traffic density (truck VMT).
    """)

    st.subheader("How the chatbot works")
    st.markdown("""
Every `/chat` request runs this pipeline before calling the LLM:

1. **Intent extraction** — detects borough names, UHF neighborhoods, years, ZIP codes without an LLM
2. **Structured filter** — filters the in-memory dataset using detected values
3. **Semantic search** — embeds the question with Gemini text-embedding-004 and queries ChromaDB
4. **Grounded prompt** — injects the 8 most relevant rows directly into the system prompt
5. **LLM call** — Gemini 2.0 Flash answers citing only the provided rows as `(Row N)`
    """)

    st.subheader("Key findings")
    findings = [
        ("South Bronx burden", "Hunts Point and Mott Haven carry outsized pollution loads relative to the rest of the city"),
        ("PM2.5 improvement", "Citywide PM2.5 declined from 11.1 µg/m³ in 2009 to 6.1 µg/m³ in 2022"),
        ("Traffic correlation", "PM2.5 and NO2 correlate at r = 0.96 — same traffic sources drive both"),
        ("Health co-occurrence", "Respiratory and cardiovascular hospitalization rates correlate at r = 0.87"),
        ("Asthma hotspot", "Central Harlem has the highest neighborhood-level asthma ER rate (~260 per 100,000)"),
    ]
    for title, body in findings:
        st.markdown(f"- **{title}** — {body}")

    st.subheader("Data sources")
    st.markdown("""
| Source | Role | Cost |
|--------|------|------|
| NYC Open Data (Socrata) | Historical air quality & health (2005–2024) | Free |
| EPA AirNow | Real-time AQI by ZIP code | Free |
| PurpleAir | Community PM2.5 sensors (NYC) | Free |
| Google Gemini 2.0 Flash | LLM inference | Free tier |
| Gemini text-embedding-004 | Query & document embeddings | Free tier |
    """)
