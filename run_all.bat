@echo off
REM simple batch file to run the pipeline
REM it runs everything in the readme except install and cleanup

SETLOCAL ENABLEDELAYEDEXPANSION

REM move to script folder (root of project if not already)
CD /D %~dp0

ECHO running data collection...
python src\collect_data.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO collect_data failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO running preprocess...
python src\preprocess.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO preprocess failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO training classifier (svm)...
python src\classify.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO classify failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO training classifier (bert)...
python src\classify_bert.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO classify_bert failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO testing retrieval...
python src\retrieval.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO retrieval failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO updating judgments...
python src\update_judgments.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO update_judgments failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO running evaluation...
python src\evaluate.py
IF %ERRORLEVEL% NEQ 0 (
  ECHO evaluate failed with code %ERRORLEVEL%
  GOTO :end
)

ECHO all done.
GOTO :eof

:end
ECHO stopped because a step failed.
EXIT /B %ERRORLEVEL%
