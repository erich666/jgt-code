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
#ifndef _chag_Path_h_
#define _chag_Path_h_

#include <string>


namespace chag
{


/**
 * Ensures all path separators are '/'.
 */
inline std::string normalizePath(std::string path)
{
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}



/**
 * For example given: path/path/path.blah, it returns path/path/
 */
inline std::string getBasePath(const std::string &filePath)
{
  size_t pos = filePath.find_last_of("/\\");
  if (pos != std::string::npos)
  {
    return filePath.substr(0, pos + 1);
  }
  return std::string();
}

}; // namespace chag



#endif // _chag_Path_h_
