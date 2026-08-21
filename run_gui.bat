@echo off
echo =========================================
echo  Starting Lignin Removal Predictor...
echo  (Please wait while the server starts)
echo =========================================
cd /d "%~dp0"
python -m streamlit run scripts\predict_gui.py
pause
