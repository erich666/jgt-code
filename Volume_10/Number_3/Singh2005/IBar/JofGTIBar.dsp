# Microsoft Developer Studio Project File - Name="JofGTIBar" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=JofGTIBar - Win32 Release
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "JofGTIBar.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "JofGTIBar.mak" CFG="JofGTIBar - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "JofGTIBar - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "JofGTIBar - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "JofGTIBar - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "../../../objs/JofGTIBar/Win32R"
# PROP Intermediate_Dir "../../../objs/JofGTIBar/Win32R"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "../../../include" /I "../../../../ExternalLibraries/include" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 fltk.lib fltkgl.lib fltkforms.lib wsock32.lib comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386 /out:"../../../bin/Win32R/JofGTIBarR.exe" /libpath:"../../../../ExternalLibraries/lib/Win32R"

!ELSEIF  "$(CFG)" == "JofGTIBar - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /I "../../../../ExternalLibraries/include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 fltkgld.lib fltkd.lib wsock32.lib OpenGL32.lib comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /out:"Debug/JofGTIBarDB.exe" /pdbtype:sept /libpath:"../../../../ExternalLibraries/lib/Win32DB"

!ENDIF 

# Begin Target

# Name "JofGTIBar - Win32 Release"
# Name "JofGTIBar - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\CAMIBar.cpp
# End Source File
# Begin Source File

SOURCE=.\JofGTIBar.cpp
# End Source File
# Begin Source File

SOURCE=.\main.cpp
# End Source File
# Begin Source File

SOURCE=.\OGLObjsCamera.cpp
# End Source File
# Begin Source File

SOURCE=.\OGLObjsCamera_Keyboard.cpp
# End Source File
# Begin Source File

SOURCE=.\R2Line.cpp
# End Source File
# Begin Source File

SOURCE=.\R2Line_seg.cpp
# End Source File
# Begin Source File

SOURCE=.\UserInterface.cpp
# End Source File
# Begin Source File

SOURCE=.\UserInterface.h
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Cam_IBar.H
# End Source File
# Begin Source File

SOURCE=.\JofGTIBar.h
# End Source File
# Begin Source File

SOURCE=.\OGLObjs_Camera.H
# End Source File
# Begin Source File

SOURCE=.\OGLObjs_Camera_i.H
# End Source File
# Begin Source File

SOURCE=.\R2_Line.H
# End Source File
# Begin Source File

SOURCE=.\R2_Line_i.H
# End Source File
# Begin Source File

SOURCE=.\R2_Line_seg.H
# End Source File
# Begin Source File

SOURCE=.\R2_Line_seg_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Affine1.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Affine2.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Affine3.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Affine4.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Binary1_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Binary2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Binary3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Binary4_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_CoVector1_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_CoVector2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_CoVector3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_CoVector4_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Defs.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Io1_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Io2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Io3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Io4_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Matrix2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Matrix3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Matrix4_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Point1_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Point2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Point3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Point4_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Unary1_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Unary2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Unary3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Unary4_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Vector1_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Vector2_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Vector3_i.H
# End Source File
# Begin Source File

SOURCE=.\Rn_Vector4_i.H
# End Source File
# Begin Source File

SOURCE=.\S4_quaternion.H
# End Source File
# Begin Source File

SOURCE=.\StdAfx.H
# End Source File
# Begin Source File

SOURCE=.\WINSystemDefines.H
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
