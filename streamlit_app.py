"""
DDD cross-validation experiment viewer (entry point).

  streamlit run streamlit_app.py

Implementation lives in ``streamlit_ui/viewer_app.py``.
For the full analysis UI, use ``streamlit run streamlit_analysis_app.py``.
"""
from streamlit_ui.viewer_app import main

if __name__ == "__main__":
    main()
