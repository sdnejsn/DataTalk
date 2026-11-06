import json

import gradio as gr

from frontend_bridge import FrontendBridge


bridge = FrontendBridge()

STYLE = """
.gradio-container {max-width: 1440px !important; margin: auto !important;}
#hero {padding: 22px 4px 12px;}
#hero h1 {font-size: 36px; font-weight: 800; letter-spacing: -.035em; margin-bottom: 5px;}
#hero p {color: var(--body-text-color-subdued); font-size: 15px;}
#workspace {gap: 16px;}
#left-panel, #right-panel {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px;}
#run-button {min-height: 46px;}
#footer-note {font-size: 12px; color: var(--body-text-color-subdued);}
"""


def handle_upload(file_path):
    try:
        frame = bridge.read_csv(file_path)
    except (ValueError, OSError, UnicodeError) as exc:
        return None, None, bridge.schema(None), f"Upload error: {exc}", gr.update(interactive=False)
    label = f"Dataset ready | {len(frame):,} rows | {len(frame.columns):,} columns"
    return frame, frame.head(20), bridge.schema(frame), label, gr.update(interactive=True)


def handle_analysis(question, frame, mode, history):
    history = list(history or [])
    try:
        if mode == "Offline demo":
            result = bridge.offline_demo(question, frame)
        else:
            result = bridge.analyze(question, frame)
    except (ValueError, OSError, UnicodeError, RuntimeError, TypeError) as exc:
        result = {
            "success": False,
            "message": f"Request error: {exc}",
            "code": "",
            "plots": [],
            "status": "Failed | Request could not be completed",
            "attempts": [],
            "pass_at_1": None,
            "retry_count": None,
        }
    history.extend(
        [
            {"role": "user", "content": str(question or "")},
            {"role": "assistant", "content": result["message"]},
        ]
    )
    diagnostics = {
        "pass_at_1": result.get("pass_at_1"),
        "retry_count": result.get("retry_count"),
        "attempts": result.get("attempts", []),
    }
    return (
        history,
        history,
        result["message"],
        result["plots"],
        result["code"],
        result["status"],
        json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str),
    )


def reset_results():
    return [], "", [], "", "Ready", ""


def clear_dataset():
    return None, None, None, bridge.schema(None), "No dataset loaded", gr.update(interactive=False), *reset_results()


def build_app():
    with gr.Blocks(title="DataTalk | Data Analysis Workspace", css=STYLE, theme=gr.themes.Soft()) as demo:
        frame_state = gr.State(value=None)
        history_state = gr.State(value=[])

        gr.HTML(
            '<div id="hero"><h1>DataTalk</h1><p>Ask questions about your CSV data. '
            'Generate Python analysis and visualize results in one workspace.</p></div>'
        )

        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_id="left-panel"):
                gr.Markdown("### Dataset")
                file_upload = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
                file_status = gr.Markdown("No dataset loaded")
                with gr.Tabs():
                    with gr.Tab("Preview"):
                        preview = gr.Dataframe(label="First 20 rows", interactive=False, wrap=True)
                    with gr.Tab("Columns"):
                        schema = gr.Dataframe(label="Column metadata", interactive=False, wrap=True)
                gr.Markdown("### Query")
                question = gr.Textbox(
                    label="Ask a question",
                    placeholder="Example: Group sales by category and generate a bar chart.",
                    lines=3,
                    max_lines=6,
                )
                mode = gr.Radio(
                    choices=["Live backend", "Offline demo"],
                    value="Live backend",
                    label="Execution mode",
                    info="Offline demo shows local summary statistics and does not interpret the question.",
                )
                run_button = gr.Button("Analyze dataset", variant="primary", interactive=False, elem_id="run-button")
                clear_button = gr.Button("Reset workspace", variant="secondary")
            with gr.Column(scale=7, min_width=360, elem_id="right-panel"):
                gr.Markdown("### Analysis")
                run_status = gr.Markdown("Ready")
                result_text = gr.Textbox(label="Execution output", interactive=False, lines=7, max_lines=18)
                gallery = gr.Gallery(label="Generated charts", columns=2, object_fit="contain", height=340, preview=True)
                with gr.Accordion("Generated Python code", open=False):
                    code = gr.Code(label="Code", language="python", interactive=False, lines=12)
                with gr.Accordion("Conversation history", open=False):
                    chatbot = gr.Chatbot(label="Session questions and results", height=280)
                with gr.Accordion("Execution diagnostics", open=False):
                    diagnostics = gr.Code(label="Attempts and execution metadata", language="json", interactive=False, lines=10)

        gr.Markdown(
            "Local course prototype. Uploaded data is processed locally by the backend. "
            "Questions and schema metadata may be sent to the selected model provider in live mode.",
            elem_id="footer-note",
        )
        file_upload.change(
            handle_upload,
            inputs=[file_upload],
            outputs=[frame_state, preview, schema, file_status, run_button],
        )
        run_button.click(
            handle_analysis,
            inputs=[question, frame_state, mode, history_state],
            outputs=[chatbot, history_state, result_text, gallery, code, run_status, diagnostics],
            concurrency_limit=1,
            show_progress="full",
        )
        clear_button.click(
            clear_dataset,
            outputs=[file_upload, frame_state, preview, schema, file_status, run_button, chatbot, result_text, gallery, code, run_status, diagnostics],
        ).then(lambda: [], outputs=[history_state])
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(show_error=True)
