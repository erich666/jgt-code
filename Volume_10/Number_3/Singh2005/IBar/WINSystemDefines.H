#ifndef _PLATFORM_DEPENDENT_DEFS_H
#define _PLATFORM_DEPENDENT_DEFS_H

/**@name Platform dependent nonsense.

   This amazingly ugly file contains all of the
   definitions/includes/workarounds needed to get my source code to
   compile on the SGI (CC compiler) sun (gcc or native compiler) and
   PCs (linux gcc or windows). Some things to note: <br>

   I don't use min/max/bool, instead I use WINmin, WINmax, and
   WINbool, which are my own definitions (or default to the
   appropriate definition on the given architecture). <br>

   I include all the system files I might need here
   (string/file/math/stdio/memory). This slows down compile time a
   bit, but it sure beats chasing down all of those include files when
   you try to get it to compile on another platform. <br>

   There's some syntactic differences between the compilers, the worst
   being those compilers which define for(int i ..) with the int i
   defined inside vs. outside the for loop. There's also some funny
   stuff with template and inline, so there's #defines for those as
   well. And of course verify/assert. <br>

   And there's the weird list of functions which appear on some
   platforms but not others. <br>

   And last, but not least, Windows does not have a set binary call on
   their streams (you have to set the binary flag at open time) so
   there's a couple routines to guarantee that a given file is opened
   in binary mode. There's a variable in here (WINDATAHOME and the
   corresponding function WINDataHome) which point to a data
   directory. Since windows and suns have different maps to the root
   directory...

   */
//@{

/// Declare in main program. Points to top of data tree.
extern char g_strDataHome[];

#ifdef sgi
  #include <strings.h>
  #include <iostream.h>
  #include <fstream.h>
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
  #include <float.h>

  #define EmptyTemplate
  #define tinline inline
  #define tempbrackets
  #define FORINT 
  
  #ifndef ASSERT
    #include <assert.h>
    #define ASSERT(f) assert(f)
  #endif

  #ifdef DEBUG
    #define VERIFY(f) assert(f)
  #else
    #define VERIFY(f) f
  #endif

  #define TRACE printf

  typedef  void (*HHSIG_PF)(int);

  typedef enum {
     FALSE = 0,
     TRUE = 1
  } WINbool;

  // Disable warning about possible loss of precision in conversions.
  // Single-precision operators not available.
   inline float sqrtf(float f) { return (float)sqrt(f); }
   inline float fabsf(float f) { return (float)fabs(f); }
   inline float sinf(float f) { return (float)sin(f); }
   inline float cosf(float f) { return (float)cos(f); }
   inline float tanf(float f) { return (float)tan(f); }
   inline float acosf(float f) { return (float)acos(f); }
   inline float ceilf(float f) { return (float)ceil(f); }
   inline float floorf(float f) { return (float)floor(f); }

  extern const char* CTime();

#endif

#ifdef WIN32
  #define __WIN32

  #define NOMINMAX
  #define VC_EXTRALEAN		// Exclude rarely-used stuff from Windows headers
  #define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

  #ifdef _AFXDLL
    #include <afx.h>
    #include <afxwin.h>
  #else
    #include <assert.h>
    #include <crtdbg.h>
    #define ASSERT _ASSERTE
    #define TRACE printf
    #define VERIFY(x) if ( !(x) ) ASSERT(false);
    #ifndef FALSE
      #define FALSE false
      #define TRUE true
    #endif
  #endif
  #include <string>
  #include <iostream>
  #include <iomanip>
  #include <fstream>
  #include <memory>             // memcpy()
  #include <float.h>
  #include <math.h>
  using namespace std;

  // I have not thought this through.  Look at WSAEWOULDBLOCK...
  #define EWOULDBLOCK EAGAIN

  #define EmptyTemplate template<>
  #define tinline inline
  #define tempbrackets

  #define FORINT 

  #ifndef assert
    #define assert ASSERT
  #endif

  typedef void (__cdecl *HHSIG_PF)(int sig,...);
  // bcopy() not available.
  inline void bcopy(const void* src, void* dst, int nbytes)
  {
      memcpy(dst,src,nbytes);
  }
  // sleep() available as Sleep()
  extern "C" __declspec(dllimport) void __stdcall Sleep(unsigned long millsecs);
  inline void sleep(int sec) { Sleep(sec*1000); }

  typedef bool WINbool;

#endif

#ifdef sun
  #include <strings.h>
  #include <iostream.h>
  #include <fstream.h>
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
  #include <float.h>

  #define UINT    int

  #define EmptyTemplate template<>
  #define tinline 
  #define tempbrackets
#ifdef SOLGCC
  #define FORINT int
#endif
#ifdef SOLCC42
  #define FORINT 
#endif
#ifdef SOLCC50
  #define FORINT int
#endif

  #ifndef ASSERT
    #include <assert.h>
    #define ASSERT(f) assert(f)
  #endif

  #ifdef DEBUG
    #define VERIFY(f) assert(f)
  #else
    #define VERIFY(f) f
  #endif

  #define TRACE printf

  // Disable warning about possible loss of precision in conversions.
  // Single-precision operators not available.
   inline float sqrtf(float f) { return (float)sqrt(f); }
   inline float fabsf(float f) { return (float)fabs(f); }
   inline float sinf(float f) { return (float)sin(f); }
   inline float cosf(float f) { return (float)cos(f); }
   inline float tanf(float f) { return (float)tan(f); }
   inline float acosf(float f) { return (float)acos(f); }
   inline float ceilf(float f) { return (float)ceil(f); }
   inline float floorf(float f) { return (float)floor(f); }

  extern const char* CTime();

  typedef void (*HHSIG_PF)(int);

#ifndef FALSE
  typedef enum {
   FALSE = 0,
   TRUE = 1
  } WINbool;
#else
typedef int WINbool
#endif

  #define bool int
  #define false 0
  #define true 1
  #define HAS_BOOL_TYPE 
 
#endif


#ifdef linux
  #include <strings.h>
  #include <iostream.h>
  #include <fstream.h>
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
  #include <float.h>
  #include <string.h>
  #include <netinet/in.h>
  #include <unistd.h>

  #define UINT    int

  #define EmptyTemplate template<>
  #define tinline 
  #define tempbrackets <>
 
  #ifndef ASSERT
    #include <assert.h>
    #define ASSERT(f) assert(f)
  #endif

  #define FORINT int

  #ifdef DEBUG
    #define VERIFY(f) assert(f)
  #else
    #define VERIFY(f) f
  #endif

  #define TRACE printf

  extern const char* CTime();


  typedef void (*HHSIG_PF)(int);
  typedef bool WINbool;

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#endif

/// A kinda clever way to get around undeclared variable warnings in release mode
  /// (use for, e.g., intput index variables that are not touched/used in release mode)
#ifdef _DEBUG
#define DECLAREVAR(v) v
#else
#define DECLAREVAR(v)
#endif
  ///
inline WINbool WINBoolRead( istream &in )
{
    char c;
    in >> c;
    if ( c == 't' ) return TRUE;
    else if ( c == 'f' ) return FALSE;
    else ASSERT(FALSE);

    return FALSE;
}

  ///

inline void WINBoolWrite( ostream &out, const WINbool &in_b )
{
    if ( in_b == FALSE ) {
        out << " f ";
    } else {
        out << " t ";
    }
}



/**@name Open files in binary mode. Checks result. */
//@{
///
inline void WINWriteBinary(const char *name, ofstream &out)
{ 
    out.clear();

#ifdef sgi
    out.open(name, ios::out); 
#else
    out.open(name, ios::out | ios::binary); 
#endif
    if (!out.good()) {
        cerr << "Bad file " << name << "\n";
        ASSERT(FALSE); 
    }
}
///
inline WINbool WINWriteBinaryTest(const char *name, ofstream &out)
{ 
    out.clear();

#ifdef sgi
    out.open(name, ios::out); 
#else
    out.open(name, ios::out | ios::binary); 
#endif
    if (!out.good()) {
        cerr << "Bad file " << name << "\n";
        return FALSE; 
    }
    return TRUE;
}


///
inline void WINAppendBinary(const char *name, ofstream &out)
{ 
    out.clear();
#ifdef sgi
    out.open(name, ios::app); 
#else
    out.open(name, ios::app | ios::binary); 
#endif

    if (!out.good()) {
        cerr << "Bad file " << name << "\n";
        ASSERT(FALSE); 
    }
}
///
inline void WINReadBinary(const char *name, ifstream &in)
{ 
    in.clear();
#ifdef sgi
    in.open(name, ios::in); 
#else
    in.open(name, ios::in | ios::binary); 
#endif

    if (!in.good()) {
        cerr << "Bad file " << name << "\n";
        ASSERT(FALSE); 
    }
}
///
inline WINbool WINReadBinaryTest(const char *name, ifstream &in)
{ 
    in.clear();
#ifdef sgi
    in.open(name, ios::in); 
#else
    in.open(name, ios::in | ios::binary); 
#endif

    if (!in.good()) {
        cerr << "Bad file " << name << "\n";
        return FALSE; 
    }
    return TRUE;
}


/** Generate a file name pointing to the data directory.
    Usually /map/gfx1/common/ManifoldsProject/data.
    WARNING: Do not use twice in one call if you don't pass in strings
    for the routines to put the result.
    @param in_strAppend The file to open
    @param out_str A place to put the glued-together name */
inline char *WINDataHome(const char *in_strAppend, char *out_str = NULL) 
{
    static char str[512];
    if ( out_str == NULL )
        out_str = str;
    
    sprintf(out_str, "%s%s", g_strDataHome, in_strAppend);
    return out_str;
}
//@}

/**@name Min/max, WINbool.
   I'm sick of trying to work around everyone's personal definitions
   of bool, min, and max. So I made my own :).
   WINbool is an enumerated type on suns/sgi, type bool on windows. */
//@{
///
template<class T> inline T WINmin(T a, T b) { return a<b?a:b; }
///
template<class T> inline T WINmax(T a, T b) { return a>b?a:b; }
///
template<class T> inline T WINminmax(T a, T tMin, T tMax) 
{ 
    return (a > tMax) ? tMax : ( (a < tMin) ? tMin : a ); 
}
//@}
//@}

#endif




