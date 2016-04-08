# Microsoft Developer Studio Generated NMAKE File, Based on tsplat.dsp
!IF "$(CFG)" == ""
CFG=tsplat - Win32 Debug
!MESSAGE No configuration specified. Defaulting to tsplat - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "tsplat - Win32 Release" && "$(CFG)" != "tsplat - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "tsplat.mak" CFG="tsplat - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "tsplat - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "tsplat - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "tsplat - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

ALL : "$(OUTDIR)\tsplat.exe"


CLEAN :
	-@erase "$(INTDIR)\main.obj"
	-@erase "$(INTDIR)\OSUmatrix.obj"
	-@erase "$(INTDIR)\parser.obj"
	-@erase "$(INTDIR)\tsplat.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\viewer.obj"
	-@erase "$(INTDIR)\volume.obj"
	-@erase "$(OUTDIR)\tsplat.exe"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /ML /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /Fp"$(INTDIR)\tsplat.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\tsplat.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=nvparse.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:no /pdb:"$(OUTDIR)\tsplat.pdb" /machine:I386 /out:"$(OUTDIR)\tsplat.exe" 
LINK32_OBJS= \
	"$(INTDIR)\main.obj" \
	"$(INTDIR)\OSUmatrix.obj" \
	"$(INTDIR)\parser.obj" \
	"$(INTDIR)\tsplat.obj" \
	"$(INTDIR)\viewer.obj" \
	"$(INTDIR)\volume.obj"

"$(OUTDIR)\tsplat.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "tsplat - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

ALL : "$(OUTDIR)\tsplat.exe"


CLEAN :
	-@erase "$(INTDIR)\main.obj"
	-@erase "$(INTDIR)\OSUmatrix.obj"
	-@erase "$(INTDIR)\parser.obj"
	-@erase "$(INTDIR)\tsplat.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\viewer.obj"
	-@erase "$(INTDIR)\volume.obj"
	-@erase "$(OUTDIR)\tsplat.exe"
	-@erase "$(OUTDIR)\tsplat.ilk"
	-@erase "$(OUTDIR)\tsplat.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP_PROJ=/nologo /MLd /W3 /Gm /GX /ZI /Od /I "C:\Program Files\NVIDIA Corporation\NVSDK5.1\OpenGLSDK\include\glh" /I "C:\Program Files\NVIDIA Corporation\NVSDK5.1\OpenGLSDK\include\shared" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /Fp"$(INTDIR)\tsplat.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\tsplat.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=nvparse.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:yes /pdb:"$(OUTDIR)\tsplat.pdb" /debug /machine:I386 /out:"$(OUTDIR)\tsplat.exe" /pdbtype:sept 
LINK32_OBJS= \
	"$(INTDIR)\main.obj" \
	"$(INTDIR)\OSUmatrix.obj" \
	"$(INTDIR)\parser.obj" \
	"$(INTDIR)\tsplat.obj" \
	"$(INTDIR)\viewer.obj" \
	"$(INTDIR)\volume.obj"

"$(OUTDIR)\tsplat.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("tsplat.dep")
!INCLUDE "tsplat.dep"
!ELSE 
!MESSAGE Warning: cannot find "tsplat.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "tsplat - Win32 Release" || "$(CFG)" == "tsplat - Win32 Debug"
SOURCE=.\main.cpp

"$(INTDIR)\main.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\OSUmatrix.cpp

"$(INTDIR)\OSUmatrix.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\parser.cpp

"$(INTDIR)\parser.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\tsplat.cpp

"$(INTDIR)\tsplat.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\viewer.cpp

"$(INTDIR)\viewer.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\volume.cpp

"$(INTDIR)\volume.obj" : $(SOURCE) "$(INTDIR)"



!ENDIF 

