# Standard Imports
import streamlit as st

# Local Imports
from src.configs.settings import PROJECT_PATH

st.set_page_config(
    page_title="Delivery Time Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    pages = st.navigation(
        pages=[
            st.Page(
                page=PROJECT_PATH.joinpath("src", "streamlit", "playground.py"),
                title="Predições",
                icon="🎯",
            ),
            st.Page(
                page=PROJECT_PATH.joinpath(
                    "src", "streamlit", "business_understanding.py"
                ),
                title="1. Business Understanding",
                icon="📚",
            ),
            st.Page(
                page=PROJECT_PATH.joinpath("src", "streamlit", "data_understanding.py"),
                title="2. Data Understanding",
                icon="📊",
            ),
            st.Page(
                page=PROJECT_PATH.joinpath("src", "streamlit", "data_preparation.py"),
                title="3. Data Preparation",
                icon="🧹",
            ),
            st.Page(
                page=PROJECT_PATH.joinpath(
                    "src", "streamlit", "modelling_and_evaluation.py"
                ),
                title="4. Modelling and Evaluation",
                icon="🤖",
            ),
            st.Page(
                page=PROJECT_PATH.joinpath("src", "streamlit", "deployment.py"),
                title="5. Deployment",
                icon="🚀",
            ),
        ],
        position="top",
    )

    pages.run()


if __name__ == "__main__":
    main()
