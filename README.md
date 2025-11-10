# Data Analyst Assistant with AI
# ========================================

# This Streamlit app lets you upload a CSV and ask questions about the dataset using
# Google Gemini via langchain-google-genai.

# Requirements
# ------------
# See `requirements.txt`.

# Setup
# -----
# 1. Create and activate a virtual environment and install dependencies:
#    Windows (PowerShell):
#      python -m venv .venv
#      .venv\Scripts\Activate.ps1
#      pip install -r requirements.txt

#    Windows (cmd):
#      python -m venv .venv
#      .venv\Scripts\activate
#      pip install -r requirements.txt

#    macOS / Linux:
#      python -m venv .venv
#      source .venv/bin/activate
#      pip install -r requirements.txt

# 2. Provide your API key. Preferred: set the environment variable in the terminal
#    before running Streamlit. Examples:

#    - PowerShell:
#      $env:GEMINI_API_KEY = "your_api_key_here"

#    - Command Prompt (cmd):
#      set GEMINI_API_KEY=your_api_key_here

#    - Bash (macOS / Linux):
#      export GEMINI_API_KEY="your_api_key_here"

#    Note: `python-dotenv` can load a `.env` file, but exporting the variable in the terminal
#    ensures the running process sees it.

# 3. Run the app:
#      streamlit run analista.py

# Notes
# -----
# - If GEMINI_API_KEY is not set, the AI assistant will be disabled and the UI will show a warning.
# - If you had code referencing a deleted folder named "Gráficos", search for that string
#   and comment those lines with `#` before deleting the folder from the repository.