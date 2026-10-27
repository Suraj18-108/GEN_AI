"""
streamlit_app.py: Streamlit Web Application for GPT-2 Interactive Story Generator.

Features:
- Compatible with Streamlit Community Cloud (share.streamlit.io) and local execution.
- High-performance caching with @st.cache_resource for singleton GPT-2 engine.
- Responsive dark-navy AI studio theme.
- Interactive story generation with 10 genres, temperature, and token sliders.
- Real-time reading font style customizer (Classic Serif, Modern Sans, Typewriter, Literary, Cursive).
- Side-by-side temperature comparison panel (customizable Temperature A & B).
- Evidence-based comparative summary table with lexical diversity (TTR) metrics.
"""

import streamlit as st
from model import (
    SUPPORTED_STYLES,
    calculate_lexical_diversity,
    get_story_engine,
    validate_story_inputs,
)

# Page Configuration
st.set_page_config(
    page_title="GPT-2 Interactive Story Generator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Font Styles Map
FONT_STYLE_MAP = {
    "Classic Serif (Georgia)": "Georgia, Cambria, 'Times New Roman', serif",
    "Modern Sans (Inter/Segoe UI)": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "Typewriter (Courier New)": "'Courier New', Courier, Consolas, monospace",
    "Literary Novel (Palatino)": "'Palatino Linotype', 'Book Antiqua', Palatino, serif",
    "Cursive Script (Handwriting)": "'Brush Script MT', 'Lucida Handwriting', cursive, serif",
}


@st.cache_resource(show_spinner="Loading GPT-2 model into memory (cached across sessions)...")
def get_cached_engine():
    """Load model once and cache across Streamlit runs."""
    engine = get_story_engine()
    engine.load_model()
    return engine


engine = get_cached_engine()

# Initialize session state variables
if "story_text" not in st.session_state:
    st.session_state.story_text = ""
if "story_metrics" not in st.session_state:
    st.session_state.story_metrics = None
if "comp_output_a" not in st.session_state:
    st.session_state.comp_output_a = None
if "comp_output_b" not in st.session_state:
    st.session_state.comp_output_b = None
if "comp_summary" not in st.session_state:
    st.session_state.comp_summary = ""

# Sidebar: Controls & Options
with st.sidebar:
    st.markdown("### ⚙️ Typography & Settings")
    selected_font = st.selectbox(
        "Reading Font Style",
        options=list(FONT_STYLE_MAP.keys()),
        index=0,
        help="Customizes the reading font family across all generated story outputs.",
    )

    st.markdown("---")
    st.markdown("### 🖥️ Engine Information")
    st.markdown(f"**Model:** `openai-community/gpt2` (124M)")
    st.markdown(f"**Compute Device:** `{engine.device.upper()}`")
    st.markdown(f"**Max Context Window:** `1024 tokens`")
    st.markdown(f"**Sampling:** `Top-k: 50, Top-p: 0.92`")

# Inject Custom Dynamic Styling
active_font_css = FONT_STYLE_MAP[selected_font]
st.markdown(
    f"""
    <style>
        .app-header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 24px;
            color: #f8fafc;
            box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.4);
        }}
        .app-header h1 {{
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}
        .app-header .subtitle {{
            font-size: 1.1rem;
            color: #a5b4fc;
            margin-bottom: 8px;
        }}
        .story-container {{
            font-family: {active_font_css} !important;
            font-size: 1.1rem;
            line-height: 1.75;
            padding: 18px;
            background-color: #0b1120;
            border: 1px solid #1e293b;
            border-radius: 10px;
            color: #f1f5f9;
            min-height: 180px;
            white-space: pre-wrap;
        }}
        .comp-card-a {{
            border-left: 4px solid #38bdf8 !important;
        }}
        .comp-card-b {{
            border-left: 4px solid #a855f7 !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="app-header">
        <h1>GPT-2 Interactive Story Generator</h1>
        <div class="subtitle">Turn a single sentence into an imaginative story.</div>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
            Powered by OpenAI's causal GPT-2 language model and Hugging Face Transformers. Craft customized narratives,
            tune generation randomness via temperature scaling, customize generation length, and inspect empirical token counts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Workspace: 2-Column Layout
col_input, col_output = st.columns([1, 1.15], gap="large")

with col_input:
    st.markdown("### ✍️ Story Input Panel")

    opening_sentence = st.text_area(
        "Opening Sentence",
        placeholder="The old house at the end of the road held a secret...",
        height=110,
        help="Enter the opening line or premise for your story (required).",
    )

    col_style, col_temp = st.columns(2)
    with col_style:
        style = st.selectbox("Writing Style", options=SUPPORTED_STYLES, index=0)

    with col_temp:
        temperature = st.slider(
            "Temperature",
            min_value=0.2,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="Lower: predictable/focused. Higher: varied/adventurous.",
        )

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=20,
        max_value=300,
        value=150,
        step=10,
        help="Maximum number of tokens to generate (excluding prompt).",
    )

    col_btn_gen, col_btn_clear = st.columns([2, 1])
    with col_btn_gen:
        btn_generate = st.button("✨ Generate Story", type="primary", use_container_width=True)
    with col_btn_clear:
        btn_clear = st.button("🔄 Clear", use_container_width=True)

    if btn_clear:
        st.session_state.story_text = ""
        st.session_state.story_metrics = None
        st.rerun()

    if btn_generate:
        is_valid, err_msg = validate_story_inputs(opening_sentence, temperature, max_new_tokens)
        if not is_valid:
            st.error(f"⚠️ {err_msg}")
        else:
            with st.spinner("Generating narrative continuation with GPT-2..."):
                res = engine.generate_story(
                    opening_sentence=opening_sentence,
                    style=style,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                if res["success"]:
                    st.session_state.story_text = res["story"]
                    st.session_state.story_metrics = res
                    st.success("Story generated successfully!")
                else:
                    st.error(f"❌ {res['error']}")

with col_output:
    st.markdown("### 📖 Generated Story Panel")

    if st.session_state.story_text:
        st.markdown(
            f'<div class="story-container">{st.session_state.story_text}</div>',
            unsafe_allow_html=True,
        )

        # Token Metrics
        m = st.session_state.story_metrics
        if m:
            st.markdown("<br>", unsafe_allow_html=True)
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("New Tokens", m["new_tokens"])
            col_m2.metric("Prompt Tokens", m["prompt_tokens"])
            col_m3.metric("Total Tokens", m["total_tokens"])
            col_m4.metric("Style", m["style"])
            col_m5.metric("Temp", f"{m['temperature']:.1f}")
    else:
        st.info("Your generated story and token metrics will appear here once you click **'Generate Story'**.")

st.markdown("---")

# Temperature Comparison Panel
st.markdown("### 🔬 Temperature Comparison")
st.markdown(
    """
    Compare how the same GPT-2 model generates two distinct narratives from your exact opening sentence
    under two selectable temperature extremes:
    """
)

col_t_a, col_t_b, col_t_tok = st.columns(3)
with col_t_a:
    comp_temp_a = st.slider("Temperature A", 0.2, 1.5, 0.5, 0.1, help="First temperature (default: 0.5)")
with col_t_b:
    comp_temp_b = st.slider("Temperature B", 0.2, 1.5, 1.2, 0.1, help="Second temperature (default: 1.2)")
with col_t_tok:
    comp_tokens = st.slider("Comparison Max Tokens", 20, 250, 150, 10, help="Max tokens for both outputs")

if st.button("⚡ Run Temperature Comparison", type="secondary"):
    is_valid_a, err_a = validate_story_inputs(opening_sentence, comp_temp_a, comp_tokens)
    if not is_valid_a:
        st.error(f"⚠️ {err_a}")
    else:
        with st.spinner("Generating dual comparative stories at T={comp_temp_a} and T={comp_temp_b}..."):
            cmp_res = engine.compare_temperatures(
                opening_sentence=opening_sentence,
                style=style,
                temp_a=comp_temp_a,
                temp_b=comp_temp_b,
                max_new_tokens=comp_tokens,
                seed=42,
            )
            if cmp_res["success"]:
                st.session_state.comp_output_a = cmp_res["output_a"]
                st.session_state.comp_output_b = cmp_res["output_b"]
                st.session_state.comp_summary = cmp_res["summary"]
            else:
                st.error(f"❌ {cmp_res.get('error')}")

# Display Comparative Outputs
if st.session_state.comp_output_a and st.session_state.comp_output_b:
    out_a = st.session_state.comp_output_a
    out_b = st.session_state.comp_output_b

    col_out_a, col_out_b = st.columns(2)
    with col_out_a:
        st.markdown(f"#### Output A (Temperature: {out_a['temperature']:.1f})")
        st.caption(f"**Tokens:** {out_a['new_tokens']} generated ({out_a['total_tokens']} total)")
        st.markdown(
            f'<div class="story-container comp-card-a">{out_a["story"]}</div>',
            unsafe_allow_html=True,
        )

    with col_out_b:
        st.markdown(f"#### Output B (Temperature: {out_b['temperature']:.1f})")
        st.caption(f"**Tokens:** {out_b['new_tokens']} generated ({out_b['total_tokens']} total)")
        st.markdown(
            f'<div class="story-container comp-card-b">{out_b["story"]}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.comp_summary:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(st.session_state.comp_summary)
