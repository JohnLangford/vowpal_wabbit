IF NOT DEFINED VsInstallDir (
    REM Try to find VS Install
    FOR /f "usebackq tokens=*" %%i IN (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -prerelease -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) DO (
        SET "VsInstallDir=%%i"
    )
)