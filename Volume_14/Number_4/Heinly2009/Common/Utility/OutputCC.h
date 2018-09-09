
/*
 *  Copyright 2009, 2010 Grove City College
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef Common_Utility_OutputCC_h
#define Common_Utility_OutputCC_h

#include <cstdlib>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
using std::flush;

// #define USE_THREADS
#ifdef USE_THREADS
#include <pthread.h>

static pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;

#define EnterThreadSafe() pthread_mutex_lock(&output_mutex)
#define ExitThreadSafe()  pthread_mutex_unlock(&output_mutex)
#else
#define EnterThreadSafe()
#define ExitThreadSafe()
#endif // USE_THREADS

#define SetOutputFlag(flag) \
  { \
    cout.setf(flag); \
  }

#ifdef NDEBUG
#define Debug(msg)
#else
#define Debug(msg)                                                     \
  {                                                                    \
    EnterThreadSafe();                                                 \
    cerr << endl;                                                      \
    cerr << "Debug:  " << __FILE__ << ", line " << __LINE__ << ":  "   \
         << endl                                                       \
         << "  " << msg                                                \
         << flush;                                                     \
    ExitThreadSafe();                                                  \
  }
#endif

#define Error(msg)                                                   \
  {                                                                  \
    EnterThreadSafe();                                               \
    cerr << endl;                                                    \
    cerr << "Error:  " << __FILE__ << ", line " << __LINE__ << ":  " \
         << endl                                                     \
         << "  " << msg                                              \
         << flush;                                                   \
    ExitThreadSafe();                                                \
  }

#define FatalError(msg)                                                 \
  {                                                                     \
    EnterThreadSafe();                                                  \
    cerr << endl;                                                       \
    cerr << "Fatal error:  " << __FILE__ << ", line " << __LINE__ << ":  " \
         << endl                                                        \
         << "  " << msg                                                 \
         << endl                                                        \
         << endl                                                        \
         << flush;                                                      \
    ExitThreadSafe();                                                   \
    exit(1);                                                            \
  }

#define Output(msg)                             \
  {                                             \
    EnterThreadSafe();                          \
    cout << msg << flush;                       \
    ExitThreadSafe();                           \
  }

#define Warning(msg)                            \
  {                                             \
    EnterThreadSafe();                          \
    cerr << endl;                                                       \
    cerr << "Warning:  " << __FILE__ << ", line " << __LINE__ << ":  "  \
         << endl                                                        \
         << "  " << msg << flush;                                       \
    ExitThreadSafe();                                                   \
  }

#endif // Common_Utility_OutputCC_h
