@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python -m streamlit run dash3.py
pause
