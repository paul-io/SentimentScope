import gradio as gr
import torch
from inference import load_model, predict

# Load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model("model", device)

# Prediction wrapper for Gradio
def run_predict(text: str) -> tuple[str, str]:
    if not text or not text.strip():
        return "", ""

    result = predict(text.strip(), model, tokenizer, device)
    label  = result["label"]
    conf   = result["confidence"]

    if label == "Positive":
        verdict = f"### 😊 Positive\n**{conf}% confidence**"
    else:
        verdict = f"### 😞 Negative\n**{conf}% confidence**"

    bar_filled = int(conf / 5)
    bar_empty  = 20 - bar_filled
    bar        = "█" * bar_filled + "░" * bar_empty
    meter      = f"`{bar}` {conf}%"

    return verdict, meter


# Custom CSS
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

* {
    font-family: 'DM Sans', sans-serif;
}

body, .gradio-container {
    background-color: #0e0e11 !important;
    color: #e8e6e0 !important;
}

.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    padding: 48px 24px !important;
}

/* Header */
#header {
    text-align: center;
    margin-bottom: 48px;
    border-bottom: 1px solid #2a2a2e;
    padding-bottom: 36px;
}

#title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem !important;
    font-weight: 400;
    color: #e8e6e0;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 12px !important;
}

#subtitle {
    font-size: 0.95rem !important;
    color: #6b6b75 !important;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Input */
#input-box textarea {
    background: #16161a !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 6px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 16px !important;
    resize: vertical !important;
    transition: border-color 0.2s ease;
}

#input-box textarea:focus {
    border-color: #c8b89a !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(200, 184, 154, 0.08) !important;
}

#input-box label span {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b6b75 !important;
}

/* Button */
#analyze-btn {
    background: #c8b89a !important;
    border: none !important;
    border-radius: 6px !important;
    color: #0e0e11 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    width: 100% !important;
    margin-top: 12px !important;
}

#analyze-btn:hover {
    background: #d9ccb8 !important;
    transform: translateY(-1px) !important;
}

#analyze-btn:active {
    transform: translateY(0) !important;
}

/* Output boxes */
#verdict-box, #meter-box {
    background: #16161a !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 6px !important;
    padding: 20px !important;
}

#verdict-box .prose, #meter-box .prose {
    color: #e8e6e0 !important;
}

#verdict-box label span, #meter-box label span {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b6b75 !important;
}

/* Verdict markdown — positive green, negative warm red */
#verdict-box .prose h3 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.6rem !important;
    font-weight: 400 !important;
    margin-bottom: 4px !important;
}

#verdict-box .prose strong {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #6b6b75 !important;
    font-weight: 500 !important;
}

/* Meter mono font */
#meter-box .prose code {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    background: transparent !important;
    color: #c8b89a !important;
    letter-spacing: 0.02em;
}

/* Examples section */
.examples-header {
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b6b75 !important;
    margin-top: 40px !important;
    margin-bottom: 12px !important;
}

.gr-samples-table {
    border: 1px solid #2a2a2e !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

.gr-samples-table td {
    background: #16161a !important;
    color: #9a9aa0 !important;
    font-size: 0.88rem !important;
    border-color: #2a2a2e !important;
    cursor: pointer !important;
    transition: color 0.15s ease, background 0.15s ease !important;
}

.gr-samples-table tr:hover td {
    background: #1e1e24 !important;
    color: #e8e6e0 !important;
}

/* Footer */
#footer {
    text-align: center;
    margin-top: 56px;
    padding-top: 24px;
    border-top: 1px solid #2a2a2e;
    font-size: 0.78rem;
    color: #3a3a42;
    letter-spacing: 0.04em;
}

/* Hide gradio footer */
footer { display: none !important; }
"""

# Layout
with gr.Blocks(css=css, title="SentimentScope") as demo:

    gr.HTML("""
        <div id="header">
            <div id="title">SentimentScope</div>
            <div id="subtitle">Transformer-based sentiment analysis &nbsp;·&nbsp; Built from scratch in PyTorch</div>
        </div>
    """)

    with gr.Column():
        input_box = gr.Textbox(
            lines=5,
            placeholder="Paste any text — a review, tweet, headline, or anything else...",
            label="Input Text",
            elem_id="input-box",
        )

        analyze_btn = gr.Button("Analyze", elem_id="analyze-btn")

        with gr.Row():
            verdict_box = gr.Markdown(label="Sentiment", elem_id="verdict-box")
            meter_box   = gr.Markdown(label="Confidence", elem_id="meter-box")

    gr.HTML('<div class="examples-header">Try an example</div>')

    gr.Examples(
        examples=[
            ["The performances were outstanding and the cinematography was breathtaking. One of the best films I've seen in years."],
            ["Slow, predictable, and a complete waste of two hours. I wanted to leave halfway through."],
            ["Shipped a fix for the critical bug — everything is back to normal and the deploy went smoothly."],
            ["Three hours on hold just to be told they can't help me. Absolutely appalling customer service."],
            ["Just finished reading it. Couldn't put it down — every chapter pulled me deeper in."],
        ],
        inputs=input_box,
        label="",
    )

    gr.HTML("""
        <div id="footer">
            SentimentScope &nbsp;·&nbsp; Transformer trained on 50,000 IMDB reviews &nbsp;·&nbsp; 75.6% test accuracy
        </div>
    """)

    analyze_btn.click(
        fn=run_predict,
        inputs=input_box,
        outputs=[verdict_box, meter_box],
    )

    input_box.submit(
        fn=run_predict,
        inputs=input_box,
        outputs=[verdict_box, meter_box],
    )

# Launch
if __name__ == "__main__":
    demo.launch()