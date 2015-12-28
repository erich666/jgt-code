# Microsoft Developer Studio Project File - Name="_glui library" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=_glui library - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "_glui library.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "_glui library.mak" CFG="_glui library - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "_glui library - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "_glui library - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "_glui library - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /YX /FD /c
# ADD BASE RSC /l 0x409
# ADD RSC /l 0x409
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"Release\glui.lib"

!ELSEIF  "$(CFG)" == "_glui library - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /Z7 /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /YX /FD /c
# ADD CPP /nologo /GX /Z7 /Od /I "..\.." /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /YX /FD /c
# ADD BASE RSC /l 0x409
# ADD RSC /l 0x409
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"lib\glui32.lib"

!ENDIF 

# Begin Target

# Name "_glui library - Win32 Release"
# Name "_glui library - Win32 Debug"
# Begin Source File

SOURCE=..\ALGEBRA3.CPP
# End Source File
# Begin Source File

SOURCE=..\ARCBALL.CPP
# End Source File
# Begin Source File

SOURCE=..\glui.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_add_controls.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_bitmap_img_data.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_bitmaps.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_button.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_checkbox.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_column.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_control.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_edittext.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_listbox.cpp
# End Source File
# Begin Source File

SOURCE=..\Glui_mouse_iaction.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_node.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_panel.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_radio.cpp
# End Source File
# Begin Source File

SOURCE=..\Glui_rollout.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_rotation.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_separator.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_spinner.cpp
# End Source File
# Begin Source File

SOURCE=..\glui_statictext.cpp
# End Source File
# Begin Source File

SOURCE=..\Glui_translation.cpp
# End Source File
# Begin Source File

SOURCE=..\quaternion.cpp
# End Source File
# End Target
# End Project
