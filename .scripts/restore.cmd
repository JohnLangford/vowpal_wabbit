@ECHO OFF
IF DEFINED DebugBuildScripts (
    @ECHO ON
)

SETLOCAL

CALL %~dp0init.cmd
PUSHD %~dp0

ECHO Restoring "%vwRoot%\vowpalwabbit\vw.sln"
"%nugetPath%" restore "%vwRoot%\vowpalwabbit\vw.sln"
ECHO.

POPD

ENDLOCAL