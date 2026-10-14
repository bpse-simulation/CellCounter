@echo off
rem ============================================================
rem  CellCounter launcher
rem  Picks the first Python that can import BOTH cellpose and PySide6.
rem  Order: .venv -> conda env "cellpose" -> py launcher -> python on PATH
rem  "CellCounter.bat debug" runs with a console window to show errors.
rem ============================================================
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%src;%PYTHONPATH%"
set "MODE=%~1"
set "EXE="

call :try "%HERE%.venv\Scripts\python.exe"
if not defined EXE call :try "%USERPROFILE%\miniconda3\envs\cellpose\python.exe"
if not defined EXE call :try "%USERPROFILE%\anaconda3\envs\cellpose\python.exe"
if not defined EXE call :try "%LOCALAPPDATA%\miniconda3\envs\cellpose\python.exe"
if not defined EXE call :try "%ProgramData%\miniconda3\envs\cellpose\python.exe"
if not defined EXE call :trycmd py
if not defined EXE call :trycmd python

if not defined EXE goto notfound

echo Using: %EXE%
if /i "%MODE%"=="debug" (
    "%EXE%" -m cellcounter gui
    pause
    exit /b
)
if defined EXEW (
    start "" "%EXEW%" -m cellcounter gui
) else (
    start "" "%EXE%" -m cellcounter gui
)
exit /b 0

rem ---- full path candidate ------------------------------------
:try
if not exist "%~1" exit /b 0
"%~1" -c "import PySide6" >nul 2>nul || (echo [skip] %~1 : PySide6 not installed & exit /b 0)
"%~1" -c "import cellpose" >nul 2>nul || (echo [skip] %~1 : cellpose not installed & exit /b 0)
set "EXE=%~1"
set "EXEW="
if exist "%~dp1pythonw.exe" set "EXEW=%~dp1pythonw.exe"
exit /b 0

rem ---- command on PATH (py / python) ---------------------------
:trycmd
where %1 >nul 2>nul || exit /b 0
%1 -c "import PySide6" >nul 2>nul || (echo [skip] %1 : PySide6 not installed & exit /b 0)
%1 -c "import cellpose" >nul 2>nul || (echo [skip] %1 : cellpose not installed & exit /b 0)
set "EXE=%1"
set "EXEW="
if /i "%1"=="py" (where pyw >nul 2>nul && set "EXEW=pyw")
if /i "%1"=="python" (where pythonw >nul 2>nul && set "EXEW=pythonw")
exit /b 0

:notfound
echo.
echo No Python environment has both cellpose and PySide6.
echo Install the missing package into the environment you use, e.g.
echo     py -m pip install cellpose PySide6
echo   or (conda)
echo     conda activate cellpose ^&^& pip install PySide6
echo.
pause
exit /b 1
