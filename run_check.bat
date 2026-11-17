@echo off
cd /d "C:\Users\NIRANJAN\.cline\data\workspaces\chat\SentimentAnalysisML"
"C:\Users\NIRANJAN\AppData\Local\Python\pythoncore-3.15-64\python.exe" -m pip install --target ./numpy_fix numpy==2.5.5 --force-reinstall --no-deps > check_output.txt 2>&1