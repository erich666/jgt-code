/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#ifndef _Win32ApiWrapper_h_
#define _Win32ApiWrapper_h_

// Windows headers generates heaps of warnings on higher warning levels so we take this opportunity to
// switch to level 3. Warnings from windows.h aren't really going to help us as it is not likely we will be
// able to fix them anyhow.
#pragma warning(push, 3)

// Next we will use some defines to turn off a stack of windows APIs that we wont need
// this is not strictly neccessary but helps compilation speed.

// These two defines turn off some of the more rarely used APIs, we wont be needing them.
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN

// In addition the below defines turn off specific APIs, lets just jettison the lot.
#define NOGDICAPMASKS //     - CC_*, LC_*, PC_*, CP_*, TC_*, RC_
#define NOWINMESSAGES //     - WM_*, EM_*, LB_*, CB_*
#define NOWINSTYLES //       - WS_*, CS_*, ES_*, LBS_*, SBS_*, CBS_*
#define NOSYSMETRICS //      - SM_*
#define NOMENUS //           - MF_*
#define NOICONS //           - IDI_*
#define NOKEYSTATES //       - MK_*
#define NOSYSCOMMANDS //     - SC_*
#define NORASTEROPS //       - Binary and Tertiary raster ops
#define NOSHOWWINDOW //      - SW_*
#define OEMRESOURCE //       - OEM Resource values
#define NOATOM //            - Atom Manager routines
#define NOCLIPBOARD //       - Clipboard routines
#define NOCOLOR //           - Screen colors
#define NOCTLMGR //          - Control and Dialog routines
#define NODRAWTEXT //        - DrawText() and DT_*
#define NOGDI //             - All GDI defines and routines
#define NOKERNEL //          - All KERNEL defines and routines
#define NONLS //             - All NLS defines and routines
#define NOMEMMGR //          - GMEM_*, LMEM_*, GHND, LHND, associated routines
#define NOMETAFILE //        - typedef METAFILEPICT
#define NOMINMAX //          - Macros min(a,b) and max(a,b)
#define NOMSG //             - typedef MSG and associated routines
#define NOOPENFILE //        - OpenFile(), OemToAnsi, AnsiToOem, and OF_*
#define NOSCROLL //          - SB_* and scrolling routines
#define NOSERVICE //         - All Service Controller routines, SERVICE_ equates, etc.
#define NOSOUND //           - Sound driver routines
#define NOTEXTMETRIC //      - typedef TEXTMETRIC and associated routines
#define NOWH //              - SetWindowsHook and WH_*
#define NOWINOFFSETS //      - GWL_*, GCL_*, associated routines
#define NOCOMM //            - COMM driver routines
#define NOKANJI //           - Kanji support stuff.
#define NOHELP //            - Help engine interface.
#define NOPROFILER //        - Profiler interface.
#define NODEFERWINDOWPOS //  - DeferWindowPos routines
#define NOMCX //             - Modem Configuration Extensions

// we use GetAsyncKeyState & VK for input
// #define NOUSER //            - All USER defines and routines
// #define NOVIRTUALKEYCODES // - VK_*
// #define NOMB //              - MB_* and MessageBox()

#include <Windows.h>

// we also want to undefine some of the more insidious macros defined in the windows API (at times at least...)
#undef max
#undef min
#undef near
#undef far
#pragma warning(pop)


#endif // _Win32ApiWrapper_h_
