#!/bin/zsh
cd "$(dirname "$0")"
source .venv_streamlit/bin/activate
streamlit run valmet_app.py
