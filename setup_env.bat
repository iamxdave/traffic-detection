@echo off
SET ENV_NAME=venv

echo 🚀 Setting up virtual environment: %ENV_NAME%

:: Remove old environment if it exists
IF EXIST %ENV_NAME% (
    echo ⚠️ Removing existing environment...
    rmdir /s /q %ENV_NAME%
)

:: Create virtual environment
python -m venv %ENV_NAME%
echo ✅ Virtual environment created!

:: Activate environment
call %ENV_NAME%\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt

echo ✅ All dependencies installed!
echo 🎯 To activate the environment, run: call %ENV_NAME%\Scripts\activate
