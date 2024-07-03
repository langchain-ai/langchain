import os

import streamlit as st

# from chatchat.webui_pages.loom_view_client import update_store
# from chatchat.webui_pages.openai_plugins import openai_plugins_page
from streamlit_option_menu import option_menu

from tests.assistant.client import ZhipuAIPluginsClient
from tests.assistant.dialogue import dialogue_page
from tests.assistant.utils import get_img_base64

api = ZhipuAIPluginsClient(base_url="http://127.0.0.1:10000")


if __name__ == "__main__":
    st.set_page_config(
        "assistant",
        get_img_base64(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "chatchat_icon_blue_square_v2.png",
            )
        ),
        initial_sidebar_state="expanded",
        menu_items={},
        layout="wide",
    )

    # use the following code to set the app to wide mode and the html markdown to increase the sidebar width
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 600px;
            margin-left: -600px;
        }
         
        """,
        unsafe_allow_html=True,
    )
    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
    }
    with st.sidebar:
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            menu_title="",
            key="selected_page",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](client=api)
