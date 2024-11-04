import os
import io
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
from email_validator import validate_email, EmailNotValidError
import cloudinary
import cloudinary.uploader

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


# Connect to Cloudinary
cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_NAME"), 
    api_key = os.getenv("CLOUDINARY_KEY"), 
    api_secret = os.getenv("CLOUDINARY_SECRET"),
    secure=True
)


def save_to_redis(current_submission):
    redis_client.lpush("submissions", json.dumps(current_submission))
    return current_submission


def get_submissions_from_redis():
    submissions = redis_client.lrange("submissions", 0, -1)
    submissions = [json.loads(submission) for submission in submissions]
    for s in submissions:
        s["quality"] = s["quality"]
        s["performance"] = s["performance"]
        s["score"] = np.sqrt(float(QUALITY_POST_FUNC(s["quality"])) ** 2 + float(PERFORMANCE_POST_FUNC(s["performance"])) ** 2)
    return filter_submissions(submissions)


def filter_submissions(submissions):
    new_submissions = []
    for sub in submissions:
        flag = True
        for new_sub in new_submissions:
            if sub["name"] == new_sub["name"]:
                flag = False
                if sub["score"] < new_sub["score"]:
                    for key in sub.keys():
                        new_sub[key] = sub[key]
                break
        if flag:
            new_submissions.append(sub)
    return new_submissions


def update_plot(
    submissions,
    current_submission=None,
):
    names = [sub["name"] for sub in submissions]
    performances = [float(PERFORMANCE_POST_FUNC(sub["performance"])) for sub in submissions]
    qualities = [float(QUALITY_POST_FUNC(sub["quality"])) for sub in submissions]
    descriptions = [sub["description"] for sub in submissions]

    # Create scatter plot
    fig = go.Figure()

    if current_submission is not None:
        fig.add_trace(
            go.Scatter(
                x=[QUALITY_POST_FUNC(current_submission["quality"])],
                y=[PERFORMANCE_POST_FUNC(current_submission["performance"])],
                mode="markers+text",
                #text=[name if not name.startswith("Baseline: ") else ""],
                #textposition="top center",
                name=current_submission["name"],
                marker=dict(symbol="star", size=15, color="orange"),
                customdata=[current_submission["name"]],
                hovertemplate = "<b>%{customdata}</b><br>" + "Performance: %{y:.3f}<br>" + "Quality: %{x:.3f}<br>" + f"Description: {current_submission['description'] if current_submission['description'] != '' else 'N/A'}" + "<extra></extra>",
            )
        )

    for name, quality, performance, description in zip(names, qualities, performances, descriptions):
        if name.startswith("Baseline: "):
            marker = dict(symbol="square", size=8, color="blue")
        else:
            marker = dict(symbol="circle", size=10, color="green")

        fig.add_trace(
            go.Scatter(
                x=[quality],
                y=[performance],
                mode="markers+text",
                #text=[name if not name.startswith("Baseline: ") else ""],
                #textposition="top center",
                name=name,
                marker=marker,
                customdata=[name if name.startswith("Baseline: ") else f"User: {name}",],
                hovertemplate = "<b>%{customdata}</b><br>" 
                + "Performance: %{y:.3f}<br>" 
                + "Quality: %{x:.3f}<br>" 
                + f"Description: {description if description != '' else 'N/A'}" 
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
                hovertemplate = "Performance: %{x:.3f}<br>" 
                + "Quality: %{y:.3f}<br>" 
                + "<extra></extra>"
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
    )
    fig.update_xaxes(title_font_size=20)
    fig.update_yaxes(title_font_size=20)

    return fig


def update_table(
    submissions,
    current_submission=None,
):
    def tp(timestamp):
        return timestamp.replace("T", " ").split('.')[0]
    def get_name(name, is_published, url_image):
        text = name[len("Baseline: "):] if name.startswith("Baseline: ") else name
        if not is_published or url_image == "":
            return text
        else:
            return f"[{text}]({url_image})"
    names = [get_name(sub["name"], sub["is_published"], sub["url_image"]) for sub in submissions]
    emails = [sub["email"] for sub in submissions]
    descriptions = [sub["description"] for sub in submissions]
    times = ["" if sub["name"].startswith("Baseline: ") else tp(sub["timestamp"]) for sub in submissions]
    performances = ["%.4f" % (float(PERFORMANCE_POST_FUNC(sub["performance"]))) for sub in submissions]
    qualities = ["%.4f" % (float(QUALITY_POST_FUNC(sub["quality"]))) for sub in submissions]
    scores = ["%.4f" % (float(sub["score"])) for sub in submissions]

    if current_submission is not None:
        names.append(get_name(current_submission["name"], current_submission["is_published"], current_submission["url_image"]))
        emails.append(current_submission["email"])
        descriptions.append(current_submission["description"])
        times.append(current_submission["timestamp"]+" (Current)")
        performances.append("%.4f" % (float(PERFORMANCE_POST_FUNC(current_submission["performance"]))))
        qualities.append("%.4f" % (float(QUALITY_POST_FUNC(current_submission["quality"]))))
        scores.append("%.4f" % (float(np.sqrt(float(QUALITY_POST_FUNC(current_submission["quality"])) ** 2 + float(PERFORMANCE_POST_FUNC(current_submission["performance"])) ** 2))))

    df = pd.DataFrame(
        {
            "Name":names,
            "Email":emails,
            "Description":descriptions,
            "Submission Time":times, 
            "Performance":performances, 
            "Quality": qualities,
            "Score": scores,
        }
    ).sort_values(
        by=["Score"]
    )
    df.insert(0, "Rank #", list(np.arange(len(names))+1), True)
    def highlight_null(s):
        con = s.copy()
        con[:] = None
        if s['Submission Time'] == '':
            con[:] = 'background-color: lightgrey'
        return con
    return df.style.apply(highlight_null, axis=1)


def process_submission(name, email, description, is_published, image):
    submissions = get_submissions_from_redis()

    original_image = Image.open("./image.png")
    progress = gr.Progress()
    progress(0, desc="Detecting Watermark")
    performance = compute_performance(image)
    progress(0.4, desc="Evaluating Image Quality")
    quality = compute_quality(image, original_image)
    progress(1.0, desc="Uploading Results")
    b = io.BytesIO()
    image.save(b, 'png')
    im_bytes = b.getvalue()
    upload_result = cloudinary.uploader.upload(im_bytes, public_id=email)
    url_image = upload_result["secure_url"]
    
    current_submission = {
        "name": name,
        "performance": performance,
        "quality": quality,
        "timestamp": datetime.now().isoformat(),
        "email": email,
        "description": description,
        "is_published": is_published,
        "url_image": url_image,
    }
    
    leaderboard_table = update_table(submissions, current_submission=current_submission)
    leaderboard_plot = update_plot(submissions, current_submission=current_submission)
    
    # Calculate rank
    distances = [
        np.sqrt(float(QUALITY_POST_FUNC(s["quality"])) ** 2 + float(PERFORMANCE_POST_FUNC(s["performance"])) ** 2)
        for s in submissions+[current_submission]
    ]
    rank = (
        sorted(distances, reverse=True).index(
            np.sqrt(float(QUALITY_POST_FUNC(quality))**2 + float(PERFORMANCE_POST_FUNC(performance))**2)
        ) + 1
    )
    gr.Info(f"You ranked {rank} out of {len(submissions)+1}!")

    save_to_redis(current_submission)
    
    return (
        leaderboard_plot,
        leaderboard_table,
        f"{rank} out of {len(submissions)}",
        name,
        f"{performance:.3f}",
        f"{quality:.3f}",
        f"{np.sqrt(quality**2 + performance**2):.3f}",
    )


def upload_and_evaluate(name, email, description, is_published, image):
    if name == "":
        raise gr.Error("Please enter your name before submitting.")
    try:
        email = validate_email(email)["email"]
    except EmailNotValidError as e:
        raise gr.Error(f"Please enter a valid email before submitting.")
    if image is None:
        raise gr.Error("Please upload an image before submitting.")
    return process_submission(name, email, description, is_published, image)


def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, js=JS) as demo:
        gr.Markdown(
            """
            # Erasing the Invisible (Demo of NeurIPS'24 competition)
            ### Welcome to the demo of the NeurIPS'24 competition [Erasing the Invisible: A Stress-Test Challenge for Image Watermarks](https://erasinginvisible.github.io/).
        
            ### You could use this demo to better understand the competition pipeline or just for fun! 🎮
            
            ### Here, we provide a image embedded with invisible watermark, you only need to:
            
            ### Step 1: **Download** the original watermarked image. 🌊
            
            ### Step 2: **Remove** the invisible watermark using your preferred attack. 🧼
            
            ### Step 3: **Upload** your image. We will evaluate and rank your attack. 📊
            
            ### That's it! 🚀
            
            ### *Note: This is just a demo. The watermark used here is not necessarily representative of those used for the competition. To officially participate in the competition, please follow the guidelines [here](https://erasinginvisible.github.io/).*
            """
        )

        with gr.Tabs(elem_classes=["tabs"]) as tabs:
            with gr.Tab(
                "Original Watermarked Image", 
                id="download"
            ):
                # gr.Markdown(
                #     """
                #     TODO: Add descriptions
                #     """
                # )
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
                # gr.Markdown(
                #     """
                #     TODO: Add descriptions
                #     """
                # )
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
                        with gr.Column():
                            description_input = gr.Textbox(
                                label="Method Description (optional)", placeholder="You could provide here a brief description of the attack", lines=6
                            )
                            is_published_input = gr.Checkbox(label="Would you like to publish your image?")
                        with gr.Column():
                            name_input = gr.Textbox(
                                label="Your Name", placeholder="Anonymous"
                            )
                            email_input = gr.Textbox(
                                label="Your Email", placeholder="Anonymous"
                            )
                            upload_btn = gr.Button("Upload and Evaluate")

            with gr.Tab(
                "Evaluation Results",
                id="plot",
                elem_classes="gr-tab-header",
            ):
                gr.Markdown(
                    """
                    <h3> The evaluation is based on two metrics, watermark performance (A) and image quality degradation (Q).
                    The lower the watermark performance and less quality degradation, the more effective the attack is.
                    The overall score is $$\large \sqrt{Q^2+A^2}$$, the smaller the better.

                    🟦: Baseline attacks
                    
                    🟢: Users' submissions
                    
                    ⭐: Your current submission

                    Note: The performance and quality metrics differ from those in the competition (as only one image is used here), but they still give you an idea of how effective your attack is.
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
                        performance_output = gr.Textbox(
                            label="Watermark Performance (lower is better)"
                        )
                        quality_output = gr.Textbox(
                            label="Quality Degredation (lower is better)"
                        )
                        overall_output = gr.Textbox(
                            label="Overall Score (lower is better)"
                        )
            with gr.Tab(
                "Leaderboard",
                id="leaderboard",
                elem_classes="gr-tab-header",
            ):
                gr.Markdown(
                    """
                    <h3> Find your ranking on the leaderboard!

                    <h3> Gray-shaded rows are baseline results provided by the organziers.

                    <h3> To check the pulished attacked images, click on the links in the "Name" column.
                    
                    <h3> For multiple submissions with the same name, only the best (lowest) score is shown.
                    """
                )
                with gr.Column():
                    leaderboard_table = gr.Dataframe(
                        value=update_table(get_submissions_from_redis()),
                        datatype=["str", "markdown", "str", "str", "str", "str", "str"],
                        show_label=False,
                        elem_id="leaderboard_table",
                    )

        submit_btn.click(lambda: gr.Tabs(selected="submit"), None, tabs)

        upload_btn.click(lambda: gr.Tabs(selected="plot"), None, tabs).then(
            upload_and_evaluate,
            inputs=[name_input, email_input, description_input, is_published_input, uploaded_image],
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
                gr.Dataframe(
                    update_table(get_submissions_from_redis()),
                    datatype=["str", "markdown", "str", "str", "str", "str", "str"]
                ),
            ],
            outputs=[original_image, leaderboard_plot, leaderboard_table],
        )

    return demo


# Create the demo object
demo = create_interface()

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
