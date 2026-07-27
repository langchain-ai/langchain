"""Simple Streamlit UI for the Content Growth Agent (MVP).

This UI accepts a YouTube or Instagram URL and displays generated outputs.
"""
from __future__ import annotations

import sys
import pathlib

# Ensure top-level 'src' is on sys.path when running the app file directly
_APP_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import streamlit as st
import asyncio

from content_growth_agent.models.input_models import VideoInput
from content_growth_agent.api import process_url
from content_growth_agent.tools.logger import get_logger

logger = get_logger("streamlit_app")


def main() -> None:
    st.set_page_config(page_title="AI Content Growth Agent", layout="wide")
    st.title("AI Content Growth Agent (MVP)")

    with st.form("input_form"):
        url = st.text_input("YouTube / Instagram URL")
        language = st.text_input("Language (en)", value="en")
        include_competitor = st.checkbox("Include competitor analysis (slower)")
        submitted = st.form_submit_button("Generate")

    if submitted:
        if not url:
            st.error("Please provide a URL.")
            return
        try:
            video = VideoInput(url=url, language=language, include_competitor_analysis=include_competitor)
        except Exception as exc:
            st.error(f"Invalid input: {exc}")
            return

        with st.spinner("Processing... this may take 10-30s depending on the model"):
            # Run async processing
            try:
                result = asyncio.run(process_url(video))
            except Exception as exc:
                logger.exception("Processing failed")
                st.error(f"Processing failed: {exc}")
                return

        st.header("Summary")
        st.write(result.summary)

        st.header("SEO Titles")
        for t in result.seo.titles:
            st.markdown(f"- {t}")

        st.header("SEO Descriptions")
        for d in result.seo.descriptions:
            st.markdown(d)

        st.header("Thumbnail Texts")
        for txt in result.thumbnail.texts:
            st.markdown(f"- {txt}")

        st.header("Instagram Caption")
        st.write(result.captions.instagram)

        st.header("LinkedIn Post")
        st.write(result.captions.linkedin)

        st.header("X Thread")
        for line in result.captions.x_thread:
            st.write(line)

        st.header("Hashtags")
        st.write(", ".join(result.hashtags))


if __name__ == "__main__":
    main()
