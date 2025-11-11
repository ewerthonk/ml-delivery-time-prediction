# Standard Imports
import streamlit as st
import pandas as pd
from textwrap import dedent
from pathlib import Path

# Local Imports
from src.configs import settings

def main():
    st.markdown(
        body=dedent(
            f"""
            ## Deployment

            O pipeline com modelo foi salvo usando a biblioteca `joblib` e está exposto em Streamlit.
            A aplicação foi containerizada e implementada em Google Cloud Run e está exposta ao público.
            
            A arquitetura pode ser vista abaixo:
            """
        ),
        unsafe_allow_html=True,
    )

    st.image(
        image=settings.RESOURCES_PATH.joinpath("architecture.drawio.png"),
    )
    

if __name__ == "__main__":
    main()