"""
Streamlit UI package for DDD cross-validation analysis.

Entry points (project root):
  streamlit run streamlit_analysis_app.py   # full analysis app
  streamlit run streamlit_app.py            # lightweight viewer

Note: package name is ``streamlit_ui`` (not ``streamlit``) to avoid shadowing
the ``streamlit`` PyPI library when the repo root is on ``sys.path``.

Pipeline CLIs live under ``scripts/run/`` (see ``scripts/README.md``); the
Experiments tab launches those, not code inside ``streamlit_ui/``.
"""
