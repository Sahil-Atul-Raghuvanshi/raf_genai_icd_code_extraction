@echo off
echo Starting Streamlit app from testing folder...
cd /d "%~dp0\.."
streamlit run testing\app.py
pause
