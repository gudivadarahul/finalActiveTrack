import streamlit as st
import requests

from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="ActiveTrack",
)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Get rid of default menu bar and footer from streamlit
sidebar = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """

st.markdown(sidebar, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.write("# Welcome to ActiveTrack! 👋")
st.subheader("Let's have a great workout!")

# select difficulty level
conf_level = st.slider('select confidence level for difficulty', 0.0, 1.0, 0.5)
st.write("Your selected level is ", conf_level*100, "percent confidence")
st.write("Higher means the difficulty will increase because it will recognize you more accurately")
st.sidebar.success("Select an exercise above.")


lottie_diagram_url = 'https://assets10.lottiefiles.com/packages/lf20_vxnelydc.json'
lottie_diagram = load_lottieurl(lottie_diagram_url)
st_lottie(lottie_diagram, key='diagram')