
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

#ifndef tangere_Node_h
#define tangere_Node_h

#include <Common/Types.h>

#include <tangere/BBox.t>

namespace tangere
{
  template<typename T>
  struct Node
  {
    BBox<T> box;     // 24 bytes
    uint    index;   //  4 bytes
    ushort  numTris; //  2 bytes
    uchar   axis;    //  1 byte
    uchar   pad;     //  1 byte
  };

} // namespace tangere

#endif // tangere_Node_h
