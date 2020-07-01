@ECHO OFF
IF DEFINED DebugBuildScripts (
    @ECHO ON
)

SETLOCAL

REM There are some issues around AV in more locked-down environments not releasing certain files in time
REM This interferes destructively with the MSBuild 2017+ propensity to try to build all possible chains of
REM dependencies at the same time, causing the clr projects to build multiple times simultaneously.
REM The fix is to force a particular build order at the solution level. In the meanwhile, to unblock local
REM command-line builds, set the ForceLinearBuild switch. 
SET __MULTIBUILD_SWITCH="/m"
IF DEFINED ForceLinearBuild (
    SET __MULTIBUILD_SWITCH=""
)

CALL %~dp0init.cmd
PUSHD %~dp0

REM TODO: Figure out how to parametrize this script?! (is there a standard, or do we actually need parse args?)
ECHO Building "%vwRoot%\vowpalwabbit\vw.sln" for Release x64
"%msbuildPath%" /bl:"%vwRoot%\vowpalwabbit\vw.binlog" /v:normal %__MULTIBUILD_SWITCH% /nr:false /p:Configuration=Release;Platform=x64 "%vwRoot%\vowpalwabbit\vw.sln" %1

POPD

ENDLOCAL
