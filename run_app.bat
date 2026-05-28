@echo off
cls
echo ========================================
echo ASD Screening Prediction System
echo ========================================
echo.
echo Installing/Updating dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting Streamlit application...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
timeout /t 2

streamlit run streamlit_app.py --server.fileWatcherType none

pause
