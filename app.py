from __future__ import annotations

import streamlit as st

from panels import apply_app_style, render_app_body


def render_app() -> None:
    st.set_page_config(page_title="Raw Representation DNA Storage", page_icon="🧬", layout="wide")
    apply_app_style()
    render_app_body()


if __name__ == "__main__":
    render_app()
