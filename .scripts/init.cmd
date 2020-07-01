REM Integration points for toolchain customization
IF NOT DEFINED nugetPath (
    SET nugetPath=nuget
)

IF NOT DEFINED msbuildPath (
    CALL %~dp0find-vs.cmd
)

IF NOT DEFINED vstestPath (
    CALL %~dp0find-vs.cmd
)

IF NOT DEFINED msbuildPath (
    IF EXIST "%VsInstallDir%\MSBuild\15.0\Bin\MSBuild.exe" (
        SET "msBuildPath=%VsInstallDir%\MSBuild\15.0\Bin\MSBuild.exe"
    ) ELSE (
        IF EXIST "%VsInstallDir%\MSBuild\Current\Bin\MSBuild.exe" (
            SET "msBuildPath=%VsInstallDir%\MSBuild\Current\Bin\MSBuild.exe"
        ) ELSE (
            ECHO Failed to find vstest.console.exe in "%VsInstallDir%\MSBuild\Current\Bin\MSBuild.exe"
            ECHO ERROR: MsBuild couldn't be found
            EXIT /b 1
        )
    )
)

IF NOT DEFINED vstestPath (
    IF EXIST "%VsInstallDir%\Common7\IDE\CommonExtensions\Microsoft\TestWindow\vstest.console.exe" (
        SET "vstestPath=%VsInstallDir%\Common7\IDE\CommonExtensions\Microsoft\TestWindow\vstest.console.exe"
    ) ELSE (
        ECHO Failed to find vstest.console.exe in "%VsInstallDir%\Common7\IDE\CommonExtensions\Microsoft\TestWindow\vstest.console.exe"
        ECHO ERROR: vstest.console couldn't be found
        EXIT /b 1
    )
)

REM Repo-specific paths
IF NOT DEFINED vwRoot (
    SET vwRoot=%~dp0..
)
