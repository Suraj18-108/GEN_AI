"""
app.py: Gradio Frontend Application for GPT-2 Interactive Story Generator.

Features:
- Responsive, dark-navy workspace with modern AI typography and card styling.
- Interactive story generation with selectable writing style, temperature, and max new tokens.
- Real-time token count auditing (distinguishing prompt tokens from newly generated tokens).
- Dynamic font style customizer (switch between Classic Serif, Modern Sans, Typewriter, Literary, and Cursive).
- Customizable temperature comparison panel (selectable Temperature A, Temperature B, and token limits).
- One-click text copying and workspace reset.
- Non-blocking error handling and informative system alerts.
- Fully compatible with modern Gradio (6.x+) and older versions.
"""

import logging
import os
import sys
import gradio as gr

from model import (
    SUPPORTED_STYLES,
    GPT2StoryEngine,
    get_story_engine,
    validate_story_inputs,
)

logger = logging.getLogger("gpt2_story_app")

# Initialize engine singleton
engine = get_story_engine()

# Supported Font Options
SUPPORTED_FONTS = [
    "Classic Serif (Georgia)",
    "Modern Sans (Inter/Segoe UI)",
    "Typewriter (Courier New)",
    "Literary Novel (Palatino)",
    "Cursive Script (Handwriting)",
]

FONT_STYLE_MAP = {
    "Classic Serif (Georgia)": "Georgia, Cambria, 'Times New Roman', serif",
    "Modern Sans (Inter/Segoe UI)": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "Typewriter (Courier New)": "'Courier New', Courier, Consolas, Monaco, monospace",
    "Literary Novel (Palatino)": "'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif",
    "Cursive Script (Handwriting)": "'Brush Script MT', 'Lucida Handwriting', 'Segoe Script', cursive, serif",
}


def get_font_style_css(font_name: str) -> str:
    """Generate dynamic CSS to alter the font style across all story display components."""
    font_css = FONT_STYLE_MAP.get(font_name, FONT_STYLE_MAP["Classic Serif (Georgia)"])
    return f"""<style id="dynamic-story-font">
        .story-output textarea, .story-output {{
            font-family: {font_css} !important;
        }}
    </style>"""


CUSTOM_CSS = """
/* App container and typography */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header styling */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 26px 30px;
    margin-bottom: 24px;
    color: #f8fafc;
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.4);
}

.app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-header .subtitle {
    font-size: 1.15rem;
    font-weight: 500;
    color: #a5b4fc;
    margin-bottom: 10px;
}

.app-header .description {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.5;
    margin: 0;
}

/* Card containers */
.panel-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

/* Metric chip pills */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 12px 0;
}

.metric-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
    display: flex;
    flex-direction: column;
    min-width: 130px;
}

.metric-chip .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 2px;
}

.metric-chip .value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #38bdf8;
}

/* Comparison section */
.comparison-card {
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
    background: #111827;
}

.comparison-card.card-a {
    border-left: 4px solid #38bdf8;
}

.comparison-card.card-b {
    border-left: 4px solid #a855f7;
}

/* Primary generate button */
.btn-generate {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
}

.btn-generate:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
}

.btn-compare {
    background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
}

/* Story reading typography */
.story-output textarea {
    font-family: Georgia, Cambria, "Times New Roman", Times, serif;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    color: #f1f5f9 !important;
    background: #090d16 !important;
}
"""


def build_metric_html(
    new_tokens: int, prompt_tokens: int, total_tokens: int, style: str, temp: float, device: str
) -> str:
    """Format token counts and generation metadata into modern metric chips."""
    return f"""
    <div class="metric-row">
        <div class="metric-chip">
            <span class="label">Newly Generated Tokens</span>
            <span class="value" style="color: #38bdf8;">{new_tokens}</span>
        </div>
        <div class="metric-chip">
            <span class="label">Prompt Tokens</span>
            <span class="value" style="color: #94a3b8;">{prompt_tokens}</span>
        </div>
        <div class="metric-chip">
            <span class="label">Total Sequence</span>
            <span class="value" style="color: #a855f7;">{total_tokens}</span>
        </div>
        <div class="metric-chip">
            <span class="label">Style</span>
            <span class="value" style="color: #fcd34d; font-size: 1.05rem;">{style}</span>
        </div>
        <div class="metric-chip">
            <span class="label">Temperature</span>
            <span class="value" style="color: #34d399; font-size: 1.05rem;">{temp:.1f}</span>
        </div>
        <div class="metric-chip">
            <span class="label">Compute Device</span>
            <span class="value" style="color: #cbd5e1; font-size: 0.95rem;">{device.upper()}</span>
        </div>
    </div>
    """


def generate_story_handler(
    opening_sentence: str, style: str, temperature: float, max_new_tokens: int
):
    """
    Event handler for Generate Story button.
    Validates user input, queries the GPT2StoryEngine, and produces formatted output.
    """
    # Front-end validation
    is_valid, validation_msg = validate_story_inputs(
        opening_sentence, temperature, max_new_tokens=max_new_tokens
    )
    if not is_valid:
        empty_metrics = build_metric_html(0, 0, 0, style, float(temperature), engine.device)
        return (
            "",  # story text
            empty_metrics,
            f"⚠️ **Validation Error**: {validation_msg}",
        )

    # Trigger generation
    result = engine.generate_story(
        opening_sentence=opening_sentence,
        style=style,
        temperature=temperature,
        max_new_tokens=int(max_new_tokens),
    )

    if not result["success"]:
        empty_metrics = build_metric_html(0, 0, 0, style, float(temperature), engine.device)
        return (
            "",
            empty_metrics,
            f"❌ **Generation Error**: {result.get('error', 'Unknown generation error occurred.')}",
        )

    # Formulate metric display
    metric_html = build_metric_html(
        new_tokens=result["new_tokens"],
        prompt_tokens=result["prompt_tokens"],
        total_tokens=result["total_tokens"],
        style=result["style"],
        temp=result["temperature"],
        device=result.get("device", engine.device),
    )

    status_msg = (
        f"✅ Story generated successfully! ({result['new_tokens']} new tokens generated using "
        f"{result.get('device', engine.device).upper()})"
    )

    return (
        result["story"],
        metric_html,
        status_msg,
    )


def compare_temperatures_handler(
    opening_sentence: str,
    style: str,
    temp_a: float,
    temp_b: float,
    max_new_tokens: int,
):
    """
    Event handler for Compare Temperatures button.
    Runs twin generation at user-selected temp_a and temp_b with chosen token limit,
    displaying side-by-side outputs and an evidence-based summary.
    """
    is_valid_a, validation_msg_a = validate_story_inputs(
        opening_sentence, temperature=temp_a, max_new_tokens=max_new_tokens
    )
    if not is_valid_a:
        return (
            "",
            f"**Output A (T={temp_a:.1f})**: 0 tokens",
            "",
            f"**Output B (T={temp_b:.1f})**: 0 tokens",
            "",
            f"⚠️ **Validation Error**: {validation_msg_a}",
        )

    is_valid_b, validation_msg_b = validate_story_inputs(
        opening_sentence, temperature=temp_b, max_new_tokens=max_new_tokens
    )
    if not is_valid_b:
        return (
            "",
            f"**Output A (T={temp_a:.1f})**: 0 tokens",
            "",
            f"**Output B (T={temp_b:.1f})**: 0 tokens",
            "",
            f"⚠️ **Validation Error**: {validation_msg_b}",
        )

    cmp_result = engine.compare_temperatures(
        opening_sentence=opening_sentence,
        style=style,
        temp_a=float(temp_a),
        temp_b=float(temp_b),
        max_new_tokens=int(max_new_tokens),
        seed=42,
    )

    if not cmp_result["success"]:
        return (
            "",
            "",
            "",
            "",
            "",
            f"❌ **Comparison Error**: {cmp_result.get('error', 'Failed during comparison generation.')}",
        )

    res_a = cmp_result["output_a"]
    res_b = cmp_result["output_b"]

    stats_a = (
        f"**Output A Metrics (T={res_a['temperature']:.1f})**: {res_a['new_tokens']} newly generated tokens "
        f"({res_a['total_tokens']} total tokens)"
    )
    stats_b = (
        f"**Output B Metrics (T={res_b['temperature']:.1f})**: {res_b['new_tokens']} newly generated tokens "
        f"({res_b['total_tokens']} total tokens)"
    )

    status_msg = f"✅ Temperature comparison completed successfully at T={temp_a:.1f} vs T={temp_b:.1f}!"

    return (
        res_a["story"],
        stats_a,
        res_b["story"],
        stats_b,
        cmp_result["summary"],
        status_msg,
    )


def reset_handler():
    """Reset all inputs, outputs, and status banners back to initial defaults."""
    default_metrics = build_metric_html(0, 0, 0, "Fantasy", 0.8, engine.device)
    default_font = "Classic Serif (Georgia)"
    return (
        "",  # opening sentence
        "Fantasy",  # style dropdown
        0.8,  # temperature slider
        150,  # max new tokens slider
        default_font,  # font_dropdown
        get_font_style_css(default_font),  # dynamic_font_holder
        "",  # generated story output
        default_metrics,  # metric chips
        "",  # main status message
        0.5,  # temp_a
        1.2,  # temp_b
        150,  # cmp_max_tokens
        "",  # story a
        "",  # stats a
        "",  # story b
        "",  # stats b
        "",  # summary markdown
        "",  # comparison status
    )


def create_app() -> gr.Blocks:
    """Build and configure the complete Gradio UI."""
    with gr.Blocks(title="GPT-2 Interactive Story Generator") as demo:
        # Injected Stylesheet for Universal Compatibility
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # Dynamic Font Style Controller
        dynamic_font_holder = gr.HTML(get_font_style_css("Classic Serif (Georgia)"))

        # Header Section
        gr.HTML(
            """
            <div class="app-header">
                <h1>GPT-2 Interactive Story Generator</h1>
                <div class="subtitle">Turn a single sentence into an imaginative story.</div>
                <p class="description">
                    Explore neural text generation with OpenAI's causal GPT-2 model. Craft customized narratives
                    by steering genres, choosing custom reading fonts, customizing maximum generation token length,
                    and dynamically modulating the softmax temperature parameter to observe its impact on vocabulary
                    diversity, syntactic predictability, and narrative creativity.
                </p>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Story Input Panel
            with gr.Column(scale=5):
                gr.Markdown("### ✍️ Story Input Panel")

                opening_input = gr.Textbox(
                    label="Opening Sentence",
                    placeholder="The old house at the end of the road held a secret...",
                    lines=4,
                    info="Enter the beginning sentence or premise for your story (required).",
                )

                style_dropdown = gr.Dropdown(
                    choices=SUPPORTED_STYLES,
                    value="Fantasy",
                    label="Writing Style",
                    info="Guides narrative framing and thematic tone.",
                )

                with gr.Row():
                    temp_slider = gr.Slider(
                        minimum=0.2,
                        maximum=1.5,
                        value=0.8,
                        step=0.1,
                        label="Sampling Temperature",
                        info="Lower: predictable. Higher: varied.",
                        scale=1,
                    )
                    tokens_slider = gr.Slider(
                        minimum=20,
                        maximum=300,
                        value=150,
                        step=10,
                        label="Max New Tokens",
                        info="Target length limit.",
                        scale=1,
                    )

                with gr.Row():
                    btn_generate = gr.Button(
                        "✨ Generate Story",
                        variant="primary",
                        elem_classes=["btn-generate"],
                        scale=2,
                    )
                    btn_reset = gr.Button("🔄 Clear / Reset", variant="secondary", scale=1)

                status_banner = gr.Markdown(value="", visible=True)

            # Right Column: Generated Story Panel
            with gr.Column(scale=6):
                gr.Markdown("### 📖 Generated Story Panel")

                with gr.Row():
                    font_dropdown = gr.Dropdown(
                        choices=SUPPORTED_FONTS,
                        value="Classic Serif (Georgia)",
                        label="Reading Font Style",
                        info="Switch typography for all story display panels.",
                        scale=3,
                    )
                    btn_copy = gr.Button("📋 Copy Story", variant="secondary", size="sm", scale=1)

                story_output = gr.Textbox(
                    label="Generated Story",
                    placeholder="Your generated story will appear here once you click 'Generate Story'...",
                    lines=10,
                    interactive=False,
                    elem_classes=["story-output"],
                )

                initial_metric_html = build_metric_html(0, 0, 0, "Fantasy", 0.8, engine.device)
                metrics_display = gr.HTML(initial_metric_html)

        gr.Markdown("---")

        # Temperature Comparison Panel
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ### 🔬 Temperature Comparison
                    Select two custom temperatures to compare how the same GPT-2 model generates distinct narratives
                    from your exact opening sentence and writing style:
                    - **Temperature A:** Default 0.5 (Tighter probability distribution, predictable phrasing).
                    - **Temperature B:** Default 1.2 (Flattened probability distribution, exploratory vocabulary).
                    """
                )

                with gr.Row():
                    cmp_temp_a = gr.Slider(
                        minimum=0.2,
                        maximum=1.5,
                        value=0.5,
                        step=0.1,
                        label="Temperature A",
                        info="First comparison temperature (default: 0.5)",
                    )
                    cmp_temp_b = gr.Slider(
                        minimum=0.2,
                        maximum=1.5,
                        value=1.2,
                        step=0.1,
                        label="Temperature B",
                        info="Second comparison temperature (default: 1.2)",
                    )
                    cmp_tokens_slider = gr.Slider(
                        minimum=20,
                        maximum=300,
                        value=150,
                        step=10,
                        label="Comparison Max New Tokens",
                        info="Length for both outputs",
                    )

                btn_compare = gr.Button(
                    "⚡ Run Temperature Comparison",
                    variant="primary",
                    elem_classes=["btn-compare"],
                )
                cmp_status = gr.Markdown(value="", visible=True)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["comparison-card", "card-a"]):
                gr.Markdown("#### Output A")
                stats_a = gr.Markdown("**Tokens:** Not generated yet")
                story_a = gr.Textbox(
                    label="Story Continuation (Output A)",
                    lines=8,
                    interactive=False,
                    elem_classes=["story-output"],
                )
                btn_copy_a = gr.Button("📋 Copy Output A", variant="secondary", size="sm")

            with gr.Column(scale=1, elem_classes=["comparison-card", "card-b"]):
                gr.Markdown("#### Output B")
                stats_b = gr.Markdown("**Tokens:** Not generated yet")
                story_b = gr.Textbox(
                    label="Story Continuation (Output B)",
                    lines=8,
                    interactive=False,
                    elem_classes=["story-output"],
                )
                btn_copy_b = gr.Button("📋 Copy Output B", variant="secondary", size="sm")

        # Comparative Summary Section
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(
                    label="Comparison Summary",
                    value="*Click 'Run Temperature Comparison' above to inspect quantitative and qualitative tendencies.*",
                )

        # Wire Event Handlers
        btn_generate.click(
            fn=generate_story_handler,
            inputs=[opening_input, style_dropdown, temp_slider, tokens_slider],
            outputs=[story_output, metrics_display, status_banner],
        )

        btn_compare.click(
            fn=compare_temperatures_handler,
            inputs=[opening_input, style_dropdown, cmp_temp_a, cmp_temp_b, cmp_tokens_slider],
            outputs=[story_a, stats_a, story_b, stats_b, summary_output, cmp_status],
        )

        font_dropdown.change(
            fn=get_font_style_css,
            inputs=[font_dropdown],
            outputs=[dynamic_font_holder],
        )

        btn_reset.click(
            fn=reset_handler,
            inputs=[],
            outputs=[
                opening_input,
                style_dropdown,
                temp_slider,
                tokens_slider,
                font_dropdown,
                dynamic_font_holder,
                story_output,
                metrics_display,
                status_banner,
                cmp_temp_a,
                cmp_temp_b,
                cmp_tokens_slider,
                story_a,
                stats_a,
                story_b,
                stats_b,
                summary_output,
                cmp_status,
            ],
        )

        # Wire Copy Buttons via browser clipboard API
        copy_js = """(text) => {
            if (!text || text.trim() === '') {
                alert('No story text available to copy.');
                return;
            }
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(() => {
                    alert('Story copied to clipboard!');
                }).catch(() => {
                    alert('Clipboard access denied.');
                });
            } else {
                alert('Clipboard API not supported in this environment.');
            }
        }"""

        btn_copy.click(fn=None, inputs=[story_output], js=copy_js)
        btn_copy_a.click(fn=None, inputs=[story_a], js=copy_js)
        btn_copy_b.click(fn=None, inputs=[story_b], js=copy_js)

    return demo


if __name__ == "__main__":
    logger.info("Starting GPT-2 Interactive Story Generator application...")
    # Preload model in background on app startup
    preload_ok, preload_err = engine.load_model()
    if preload_ok:
        logger.info("GPT-2 model ready for inference.")
    else:
        logger.warning(f"GPT-2 model preload note: {preload_err}")

    demo = create_app()
    # Launch on localhost (auto-selects first available port starting from 7860)
    demo.launch(server_name="127.0.0.1")
