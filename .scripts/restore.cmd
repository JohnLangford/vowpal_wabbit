@ECHO OFF
IF DEFINED DebugBuildScripts (
    @ECHO ON
)

SETLOCAL

CALL %~dp0init.cmd
PUSHD %~dp0

REM TODO: This really should be out-of-source
vcpkg install flatbuffers:x64-windows

ECHO Restoring "%vwRoot%\cs\common\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\common\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\cs\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\cs\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\cs_console\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\cs_console\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\cs_json\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\cs_json\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\cs_parallel\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\cs_parallel\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\examples\simulator\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\examples\simulator\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\setup_bundle\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\setup_bundle\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\cs\unittest\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\cs\unittest\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\vowpalwabbit\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\vowpalwabbit\packages.config"
ECHO.

ECHO Restoring "%vwRoot%\vowpalwabbit\slim\test\packages.config"
"%nugetPath%" restore -o "%vwRoot%\vowpalwabbit\packages" "%vwRoot%\vowpalwabbit\slim\test\packages.config"
ECHO.

POPD

ENDLOCAL
