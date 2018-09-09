SET BIN_PATH=bin\win32
IF %PROCESSOR_ARCHITECTURE%==AMD64 SET SET BIN_PATH=bin\x64

%BIN_PATH%\tiled_shading_demo_Release.exe data/crysponza/sponza.obj

pause