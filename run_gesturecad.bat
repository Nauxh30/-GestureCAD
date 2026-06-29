@echo off
cd /d "%~dp0"
python gesture_cad.py
if errorlevel 1 pause
