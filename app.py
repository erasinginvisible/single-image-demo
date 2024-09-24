import os
import gradio as gr
import numpy as np
import json
import redis
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
from kit import compute_performance, compute_quality
import dotenv
import pandas as pd

dotenv.load_dotenv()

CSS = """
.tabs button{
    font-size: 20px;
}
#download_btn {
    height: 91.6px;
}
#submit_btn {
    height: 91.6px;
}
#original_image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#uploaded_image {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
#leaderboard_plot {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 640px;  /* Adjust width as needed */
    height: 640px;  /* Adjust height as needed */
#leaderboard_table {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
"""

JS = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

QUALITY_POST_FUNC = lambda x: x / 4 * 8
PERFORMANCE_POST_FUNC = lambda x: abs(x - 0.5) * 2


# Connect to Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)


def save_to_redis(name, performance, quality):
    submission = {
        "name": name,
        "performance": performance,
        "quality": quality,
        "timestamp": datetime.now().isoformat(),
    }
    redis_client.lpush("submissions", json.dumps(submission))


def get_submissions_from_redis():
    submissions = redis_client.lrange("submissions", 0, -1)
    submissions = [json.loads(submission) for submission in submissions]
    for s in submissions:
        s["quality"] = QUALITY_POST_FUNC(s["quality"])
        s["performance"] = PERFORMANCE_POST_FUNC(s["performance"])
        s["score"] = np.sqrt(float(s["quality"]) ** 2 + float(s["performance"]) ** 2)
    return submissions


def update_plot(
    submissions,
    current_name=None,
):
    names = [sub["name"] for sub in submissions]
    performances = [float(sub["performance"]) for sub in submissions]
    qualities = [float(sub["quality"]) for sub in submissions]

    # Create scatter plot
    fig = go.Figure()

    for name, quality, performance in zip(names, qualities, performances):
        if name == current_name:
            marker = dict(symbol="star", size=15, color="orange")
        elif name.startswith("Baseline: "):
            marker = dict(symbol="square", size=8, color="blue")
        else:
            marker = dict(symbol="circle", size=10, color="green")

        fig.add_trace(
            go.Scatter(
                x=[quality],
                y=[performance],
                mode="markers+text",
                text=[name if not name.startswith("Baseline: ") else ""],
                textposition="top center",
                name=name,
                marker=marker,
                customdata=[
                    name if name.startswith("Baseline: ") else f"User: {name}",
                ],
                hovertemplate="<b>%{customdata}</b><br>"
                + "Performance: %{y:.3f}<br>"
                + "Quality: %{x:.3f}<br>"
                + "<extra></extra>",
            )
        )

    # Add circles
    circle_radii = np.linspace(0, 1, 5)
    for radius in circle_radii:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
                hovertemplate="Performance: %{x:.3f}<br>"
                + "Quality: %{y:.3f}<br>"
                + "<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Image Quality Degredation",
        yaxis_title="Watermark Detection Performance",
        xaxis=dict(
            range=[0, 1.1], titlefont=dict(size=16)  # Adjust this value as needed
        ),
        yaxis=dict(
            range=[0, 1.1], titlefont=dict(size=16)  # Adjust this value as needed
        ),
        width=640,
        height=640,
        showlegend=False,  # Remove legend
        modebar=dict(remove=["all"]),
    )
    fig.update_xaxes(title_font_size=20)
    fig.update_yaxes(title_font_size=20)

    return fig


def update_table(
    submissions,
    current_name=None,
):
    def tp(timestamp):
        return timestamp.replace("T", " ").split(".")[0]

    names = [
        (
            sub["name"][len("Baseline: ") :]
            if sub["name"].startswith("Baseline: ")
            else sub["name"]
        )
        for sub in submissions
    ]
    times = [
        (
            ""
            if sub["name"].startswith("Baseline: ")
            else (
                tp(sub["timestamp"]) + " (Current)"
                if sub["name"] == current_name
                else tp(sub["timestamp"])
            )
        )
        for sub in submissions
    ]
    performances = ["%.4f" % (float(sub["performance"])) for sub in submissions]
    qualities = ["%.4f" % (float(sub["quality"])) for sub in submissions]
    scores = ["%.4f" % (float(sub["score"])) for sub in submissions]
    df = pd.DataFrame(
        {
            "Name": names,
            "Submission Time": times,
            "Performance": performances,
            "Quality": qualities,
            "Score": scores,
        }
    ).sort_values(by=["Score"])
    df.insert(0, "Rank #", list(np.arange(len(names)) + 1), True)

    def highlight_null(s):
        con = s.copy()
        con[:] = None
        if s["Submission Time"] == "":
            con[:] = "background-color: darkgrey"
        return con

    return df.style.apply(highlight_null, axis=1)


def process_submission(name, image):
    original_image = Image.open("./image.png")
    progress = gr.Progress()
    progress(0, desc="Detecting Watermark")
    performance = compute_performance(image)
    progress(0.4, desc="Evaluating Image Quality")
    quality = compute_quality(image, original_image)
    progress(1.0, desc="Uploading Results")

    # Save unprocessed values but display processed values
    save_to_redis(name, performance, quality)
    quality = QUALITY_POST_FUNC(quality)
    performance = PERFORMANCE_POST_FUNC(performance)

    submissions = get_submissions_from_redis()
    leaderboard_table = update_table(submissions, current_name=name)
    leaderboard_plot = update_plot(submissions, current_name=name)

    # Calculate rank
    distances = [
        np.sqrt(float(s["quality"]) ** 2 + float(s["performance"]) ** 2)
        for s in submissions
    ]
    rank = sorted(distances).index(np.sqrt(quality**2 + performance**2)) + 1
    gr.Info(f"You ranked {rank} out of {len(submissions)}!")
    return (
        leaderboard_plot,
        leaderboard_table,
        f"{rank} out of {len(submissions)}",
        name,
        f"{performance:.3f}",
        f"{quality:.3f}",
        f"{np.sqrt(quality**2 + performance**2):.3f}",
    )


def upload_and_evaluate(name, image):
    if name == "":
        raise gr.Error("Please enter your name before submitting.")
    if image is None:
        raise gr.Error("Please upload an image before submitting.")
    return process_submission(name, image)


def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, js=JS) as demo:
        gr.Markdown(
            """
            # Erasing the Invisible (Demo of NeurIPS'24 competition)
            
            Welcome to the demo of the NeurIPS'24 competition [Erasing the Invisible: A Stress-Test Challenge for Image Watermarks](https://erasinginvisible.github.io/).

            You could use this demo to better understand the competition pipeline or just for fun! 🎮

            Here, we provide an image embedded with an invisible watermark. You only need to:

            1. **Download** the original watermarked image. 🌊
            2. **Remove** the invisible watermark using your preferred attack. 🧼
            3. **Upload** your image. We will evaluate and rank your attack. 📊

            That's it! 🚀

            *Note: This is just a demo. The watermark used here is not necessarily representative of those used for the competition. To officially participate in the competition, please follow the guidelines [here](https://erasinginvisible.github.io/).*
            """
        )

        with gr.Tabs(elem_classes=["tabs"]) as tabs:
            with gr.Tab("Original Watermarked Image", id="download"):
                with gr.Column():
                    original_image = gr.Image(
                        value="./image.png",
                        format="png",
                        label="Original Watermarked Image",
                        show_label=True,
                        height=512,
                        width=512,
                        type="filepath",
                        show_download_button=False,
                        show_share_button=False,
                        show_fullscreen_button=False,
                        container=True,
                        elem_id="original_image",
                    )
                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            "Download Watermarked Image",
                            value="./image.png",
                            elem_id="download_btn",
                        )
                        submit_btn = gr.Button(
                            "Submit Your Removal", elem_id="submit_btn"
                        )

            with gr.Tab(
                "Submit Watermark Removed Image",
                id="submit",
                elem_classes="gr-tab-header",
            ):

                with gr.Column():
                    uploaded_image = gr.Image(
                        label="Your Watermark Removed Image",
                        format="png",
                        show_label=True,
                        height=512,
                        width=512,
                        sources=["upload"],
                        type="pil",
                        show_download_button=False,
                        show_share_button=False,
                        show_fullscreen_button=False,
                        container=True,
                        placeholder="Upload your watermark removed image",
                        elem_id="uploaded_image",
                    )
                    with gr.Row():
                        name_input = gr.Textbox(
                            label="Your Name", placeholder="Anonymous"
                        )
                        upload_btn = gr.Button("Upload and Evaluate")

            with gr.Tab(
                "Evaluation Results",
                id="plot",
                elem_classes="gr-tab-header",
            ):
                gr.Markdown(
                    "The evaluation is based on two metrics, watermark performance ($$A$$) and image quality degradation ($$Q$$).",
                    latex_delimiters=[{"left": "$$", "right": "$$", "display": False}],
                )
                gr.Markdown(
                    "The lower the watermark performance and less quality degradation, the more effective the attack is. The overall score is $$\sqrt{Q^2+A^2}$$, the smaller the better.",
                    latex_delimiters=[{"left": "$$", "right": "$$", "display": False}],
                )
                gr.Markdown(
                    """
                    <p>
                        <span style="display: inline-block; width: 20px;"></span>🟦: Baseline attacks
                        <span style="display: inline-block; width: 20px;"></span>🟢: Users' submissions
                        <span style="display: inline-block; width: 20px;"></span>⭐: Your current submission
                    </p>
                    <p><em>Note: The performance and quality metrics differ from those in the competition (as only one image is used here), but they still give you an idea of how effective your attack is.</em></p>
                    """
                )
                with gr.Column():
                    leaderboard_plot = gr.Plot(
                        value=update_plot(get_submissions_from_redis()),
                        show_label=False,
                        elem_id="leaderboard_plot",
                    )
                    with gr.Row():
                        rank_output = gr.Textbox(label="Your Ranking")
                        name_output = gr.Textbox(label="Your Name")
                        performance_output = gr.Textbox(label="Watermark Performance")
                        quality_output = gr.Textbox(label="Quality Degredation")
                        overall_output = gr.Textbox(label="Overall Score")
            with gr.Tab(
                "Leaderboard",
                id="leaderboard",
                elem_classes="gr-tab-header",
            ):
                gr.Markdown("Find your ranking on the leaderboard!")
                gr.Markdown(
                    "Gray-shaded rows are baseline results provided by the organziers."
                )
                with gr.Column():
                    leaderboard_table = gr.Dataframe(
                        value=update_table(get_submissions_from_redis()),
                        show_label=False,
                        elem_id="leaderboard_table",
                    )
        submit_btn.click(lambda: gr.Tabs(selected="submit"), None, tabs)

        upload_btn.click(lambda: gr.Tabs(selected="plot"), None, tabs).then(
            upload_and_evaluate,
            inputs=[name_input, uploaded_image],
            outputs=[
                leaderboard_plot,
                leaderboard_table,
                rank_output,
                name_output,
                performance_output,
                quality_output,
                overall_output,
            ],
        )

        demo.load(
            lambda: [
                gr.Image(value="./image.png", height=512, width=512),
                gr.Plot(update_plot(get_submissions_from_redis())),
                gr.Dataframe(update_table(get_submissions_from_redis())),
            ],
            outputs=[original_image, leaderboard_plot, leaderboard_table],
        )

    return demo


# Create the demo object
demo = create_interface()

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
