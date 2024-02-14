import streamlit as st
from octoai.client import Client
from octoai.errors import OctoAIClientError, OctoAIServerError
from io import BytesIO
from base64 import b64encode, b64decode
import requests
from PIL import Image, ExifTags
import os
import time
import json

from streamlit_image_select import image_select

OCTOMERGE_ENDPOINT_URL = os.environ["OCTOMERGE_ENDPOINT_URL"]
OCTOAI_TOKEN = os.environ["OCTOAI_TOKEN"]

# OctoAI client
oai_client = Client(OCTOAI_TOKEN)

def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64


def octomerge(my_upload, payload):
    # Wrap all of this in a try block
    try:
        start = time.time()

        # convert images to base64 and update payload
        input_images = [read_image(Image.open(image)) for image in my_upload]
        payload["transfer_images"] = {"trigger": input_images}
        
        progress_text = "OctoMerge in action..."
        percent_complete = 0
        progress_bar = st.progress(percent_complete, text=progress_text)

        # Query endpoint async
        octomerge_futures = oai_client.infer_async(
            f"{OCTOMERGE_ENDPOINT_URL}/generate/sdxl",
            payload
        )

        # Poll on completion - target 30s completion - hence the 0.25 time step
        time_step = 0.25
        while not oai_client.is_future_ready(octomerge_futures):
            time.sleep(time_step)
            percent_complete = min(99, percent_complete + 1)
            if percent_complete == 99:
                progress_text = "OctoMerge is taking longer than usual, hang tight!"
            progress_bar.progress(percent_complete, text=progress_text)

        # Process results
        end = time.time()
        progress_bar.empty()
        st.write("OctoMerged images in {:.2f}s :star2:".format(end - start))
        result = oai_client.get_future_result(octomerge_futures)
        cols = st.columns(payload["num_images"])
        for idx, col in enumerate(cols):
            octomerged_image = Image.open(BytesIO(b64decode(result["images"][idx]["image_b64"])))
            col.image(octomerged_image)

    except OctoAIClientError as e:
        progress_bar.empty()
        st.write(
            "Oops something went wrong (client error)! Please hit OctoMerge again or [report the issue if this is a recurring problem](https://forms.gle/vWVAXa8CU7wXPGcq6)! Join our discord [here](https://discord.com/invite/rXTPeRBcG7) and hop on to the #octoshop channel to provide feedback or ask questions."
        )
        print(e)

    except OctoAIServerError as e:
        progress_bar.empty()
        st.write(
            "Oops something went wrong (server error)! Please hit OctoMerge again or [report the issue if this is a recurring problem](https://forms.gle/vWVAXa8CU7wXPGcq6)! Join our discord [here](https://discord.com/invite/rXTPeRBcG7) and hop on to the #octoshop channel to provide feedback or ask questions."
        )
        print(e)

    except Exception as e:
        progress_bar.empty()
        st.write(
            "Oops something went wrong (unexpected error)! Please hit OctoMerge again or [report the issue if this is a recurring problem](https://forms.gle/vWVAXa8CU7wXPGcq6)! Join our discord [here](https://discord.com/invite/rXTPeRBcG7) and hop on to the #octoshop channel to provide feedback or ask questions."
        )
        print(e)


st.set_page_config(layout="wide", page_title="OctoMerge", initial_sidebar_state="collapsed")

col_0, col_1 = st.columns([1, 2])
col_0.image("assets/octoai_electric_blue.png", width=300)
col_1.write("")
col_1.write("# OctoMerge Preview")
col_1.write("\n :warning: Internal Version (OctoML Only)")

with open("./examples.json", "r") as f:
    EXAMPLES = json.load(f)

if not st.session_state:
    st.session_state.update(EXAMPLES["Dissolve"])
    st.session_state["payload"] = EXAMPLES["Dissolve"]["payload"]
    st.session_state["previous_example_index"] = 0

example_image_pattern = "assets/{}_small.png"
example_image_fp = [
    example_image_pattern.format(name.lower()) for name in list(EXAMPLES.keys())[:4]
]
index_selected = image_select(
    "",
    images=example_image_fp,
    captions=list(EXAMPLES.keys())[:4],
    index=0,
    return_value="index",
)
if index_selected != st.session_state["previous_example_index"]:
    name_selected = list(EXAMPLES.keys())[index_selected]
    st.session_state.update(EXAMPLES[name_selected].copy())
    st.session_state["payload"] = EXAMPLES[name_selected]["payload"]
    st.session_state["previous_example_index"] = index_selected

st.write("")

st.write("")

    
st.session_state["images"] = st.file_uploader("Upload photos", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.session_state["images"]:
    form = st.form(key="form_settings")
    total_images = len(st.session_state["images"])
    cols = form.columns(total_images)
    for col, image in zip(cols, st.session_state["images"]):
        input_img = Image.open(image)
        col.image(input_img, width=128)
    if form.form_submit_button("OctoMerge!"):
        octomerge(st.session_state["images"], st.session_state["payload"])


st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.markdown(
    ":thumbsup: :thumbsdown: Give us your [feedback](https://forms.gle/7sfoQDjXt2SNjmp86) to help us improve OctoMerge! Or join our discord [here](https://discord.com/invite/rXTPeRBcG7) and hop on to the #octomerge channel to provide feedback or ask questions."
)
col2.markdown(
    ":bug: Report bugs, issues, or problematic content [here](https://forms.gle/vWVAXa8CU7wXPGcq6)!"
)
col3.markdown(
    ":warning: **Disclaimer** OctoMerge is built on the foundation of CLIP Interrogator, SDXL, LLAMA2, and is therefore likely to carry forward the potential dangers inherent in these base models."
)
