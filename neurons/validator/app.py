import hmac
import time
from os import listdir

import streamlit as st

st.set_page_config(layout="wide")

credentials = open("streamlit_credentials.txt", "r").read()
credentials_split = credentials.split("\n")

hashkey = credentials_split[0].split("hashkey=")[1]
username = credentials_split[1].split("username=")[1]
password = credentials_split[2].split("password=")[1]

css = """
<style>
    section.main > div {max-width:97%;}
    button[title="View fullscreen"] {display: None !important;}
</style>
"""
# img {max-height: 400px; max-width: 400px}
st.markdown(css, unsafe_allow_html=True)


def constant_time_compare(val1, val2):
    """
    Returns True if the two strings are equal, False otherwise.
    The time taken is constant and independent of the number of characters
    that match.
    """
    if not isinstance(val1, bytes):
        val1 = val1.encode()
    if not isinstance(val2, bytes):
        val2 = val2.encode()

    key: str = hashkey
    if not isinstance(key, bytes):
        key = key.encode()

    # Use an arbitrary key to prevent the values being leaked via timing.
    hmac1 = hmac.new(key, msg=val1, digestmod="sha256").digest()
    hmac2 = hmac.new(key, msg=val2, digestmod="sha256").digest()

    return hmac.compare_digest(hmac1, hmac2)


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username_correct = constant_time_compare(
            st.session_state["username"], username
        )
        password_correct = constant_time_compare(
            st.session_state["password"], password
        )

        st.session_state["password_correct"] = (
            username_correct and password_correct
        )

        # Don't store the username or password in the session
        del st.session_state["password"]
        del st.session_state["username"]

    # Username + password validated and all ok
    if st.session_state.get("password_correct", False):
        return True

    # Return to login
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

directory = r"neurons/validator/images"
st.markdown("# TensorAlchemy Human Validation")
st.markdown(
    "# Select what you think is the best image "
    + "from the batch of images below within 10 seconds ..."
)

prompt_text = st.empty()
empty_image_text = "# AWAITING NEXT BATCH ..."
# debug = st.empty()

col1, col2, col3, col4 = st.columns(4)

if "vote_1" not in st.session_state:
    st.session_state.vote_1 = False
if "vote_2" not in st.session_state:
    st.session_state.vote_2 = False
if "vote_3" not in st.session_state:
    st.session_state.vote_3 = False
if "vote_4" not in st.session_state:
    st.session_state.vote_4 = False
if "vote_5" not in st.session_state:
    st.session_state.vote_5 = False
if "vote_6" not in st.session_state:
    st.session_state.vote_6 = False
if "vote_7" not in st.session_state:
    st.session_state.vote_7 = False
if "vote_8" not in st.session_state:
    st.session_state.vote_8 = False
if "vote_9" not in st.session_state:
    st.session_state.vote_9 = False
if "vote_10" not in st.session_state:
    st.session_state.vote_10 = False
if "vote_11" not in st.session_state:
    st.session_state.vote_11 = False
if "vote_12" not in st.session_state:
    st.session_state.vote_12 = False


def input_callback():
    if st.session_state.vote_1:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("1")
            st.session_state.vote_1 = False
    elif st.session_state.vote_2:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("2")
            st.session_state.vote_2 = False
    elif st.session_state.vote_3:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("3")
            st.session_state.vote_3 = False
    elif st.session_state.vote_4:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("4")
            st.session_state.vote_4 = False
    elif st.session_state.vote_5:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("5")
            st.session_state.vote_5 = False
    elif st.session_state.vote_6:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("6")
            st.session_state.vote_6 = False
    elif st.session_state.vote_7:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("7")
            st.session_state.vote_7 = False
    elif st.session_state.vote_8:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("8")
            st.session_state.vote_8 = False
    elif st.session_state.vote_9:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("9")
            st.session_state.vote_9 = False
    elif st.session_state.vote_10:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("10")
            st.session_state.vote_10 = False
    elif st.session_state.vote_11:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("11")
            st.session_state.vote_11 = False
    elif st.session_state.vote_12:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("12")
            st.session_state.vote_12 = False


with col1:
    placeholder_1 = st.empty()
    vote_1 = st.checkbox("Image 1", key="vote_1", on_change=input_callback)
    placeholder_5 = st.empty()
    vote_5 = st.checkbox("Image 5", key="vote_5", on_change=input_callback)
    placeholder_9 = st.empty()
    vote_9 = st.checkbox("Image 9", key="vote_9", on_change=input_callback)

with col2:
    placeholder_2 = st.empty()
    vote_2 = st.checkbox("Image 2", key="vote_2", on_change=input_callback)
    placeholder_6 = st.empty()
    vote_6 = st.checkbox("Image 6", key="vote_6", on_change=input_callback)
    placeholder_10 = st.empty()
    vote_10 = st.checkbox("Image 10", key="vote_10", on_change=input_callback)

with col3:
    placeholder_3 = st.empty()
    vote_3 = st.checkbox("Image 3", key="vote_3", on_change=input_callback)
    placeholder_7 = st.empty()
    vote_7 = st.checkbox("Image 7", key="vote_7", on_change=input_callback)
    placeholder_11 = st.empty()
    vote_11 = st.checkbox("Image 11", key="vote_11", on_change=input_callback)

with col4:
    placeholder_4 = st.empty()
    vote_4 = st.checkbox("Image 4", key="vote_4", on_change=input_callback)
    placeholder_8 = st.empty()
    vote_8 = st.checkbox("Image 8", key="vote_8", on_change=input_callback)
    placeholder_12 = st.empty()
    vote_12 = st.checkbox("Image 12", key="vote_12", on_change=input_callback)


image_list = [
    placeholder_1,
    placeholder_2,
    placeholder_3,
    placeholder_4,
    placeholder_5,
    placeholder_6,
    placeholder_7,
    placeholder_8,
    placeholder_9,
    placeholder_10,
    placeholder_11,
    placeholder_12,
]

IMAGE_WIDTH = 1024
blacked_out = False
while True:
    images = [
        image
        for image in listdir(directory)
        if (".png" in image) and (image != "black.png")
    ]

    errored = False

    if images:
        blacked_out = False
        try:
            prompt = open(f"{directory}/prompt.txt", "r").read()
            prompt = prompt.replace('"', "")
            prompt_text.markdown(f"# Prompt: {prompt}")
            for i in range(0, len(image_list)):
                if len(images) > i:
                    image_list[i].image(
                        f"{directory}/{images[i]}",
                        width=IMAGE_WIDTH,
                        use_column_width=True,
                    )
                else:
                    image_list[i].image(
                        f"{directory}/black.png",
                        width=IMAGE_WIDTH,
                        use_column_width=True,
                    )
        except Exception:
            errored = True

    if not blacked_out and (not images or errored):
        for i in range(0, len(image_list)):
            prompt_text.markdown(empty_image_text)
            image_list[i].image(
                f"{directory}/black.png",
                width=IMAGE_WIDTH,
                use_column_width=True,
            )
        blacked_out = True
    time.sleep(0.1)
