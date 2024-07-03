import os
import sys

if __name__ == "__main__":
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.py")
    try:
        # for streamlit >= 1.12.1
        from streamlit.web import bootstrap
    except ImportError:
        from streamlit import bootstrap

    flag_options = {"server_address": "127.0.0.1", "server_port": 8501}
    args = []
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(script_dir, False, args, flag_options)
