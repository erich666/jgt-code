/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson, Erik Sintorn, Ulf Assarsson, Markus Billeter
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
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <float.h>
#include "GL/glew.h"
#include "GL/glut.h"
#include <IL/il.h>
#include <IL/ilut.h>
#include "OBJModel.h"
#include <stdlib.h>
#include <utils/CheckGLError.h>
#include <utils/Path.h>

#ifdef _MSC_VER
#	define FORCE_INLINE __forceinline
#	define FLATTEN 
#elif defined(__GNUC__) && __GNUC__ > 4
#	define FORCE_INLINE __attribute__((always_inline))
#	define FLATTEN __attribute__((flatten))
#else 
#	define FORCE_INLINE inline
#	define FLATTEN
#endif //_MSC_VER || __GNUC__ > 4


using namespace std;
using namespace chag;

OBJModel::OBJModel(void) :
m_numVerts(0)
{
#if 1
  glGenTextures(1, &m_defaultTextureOne);
  glBindTexture(GL_TEXTURE_2D, m_defaultTextureOne);
  float4 pixel = { 1.0f, 1.0f, 1.0f, 1.0f };
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_FLOAT, &pixel);
  glBindTexture(GL_TEXTURE_2D, 0);
  CHECK_GL_ERROR();

  glGenTextures(1, &m_defaultNormalTexture);
  glBindTexture(GL_TEXTURE_2D, m_defaultNormalTexture);
  float4 pixelZ = { 0.5f, 0.5f, 1.0f, 1.0f };
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, &pixelZ);
  glBindTexture(GL_TEXTURE_2D, 0);
  CHECK_GL_ERROR();  
#endif
}

OBJModel::~OBJModel(void)
{
}

bool OBJModel::load(std::string fileName)
{
  std::ifstream file;

  // ensure only one type of slashes, (not strictly necessary)...
  fileName = normalizePath(fileName);

  file.open(fileName.c_str(), std::ios::binary);
  if (!file)
  {
    cerr << "Error in openening file '" << fileName << "'" << endl;
    return false;
  }
  cout << "Loading OBJ file: '" << fileName << "'..." << endl;
  std::string basePath = getBasePath(fileName);
  bool result = loadOBJ(file, basePath);
  cout << " verts: " << getNumVerts() << endl;
  return result;
}



size_t OBJModel::getNumVerts()
{
  return m_numVerts;
}


struct ObjTri
{
  int v[3];
  int t[3];
  int n[3];
};



// Helper class to tokeize the obj file input.
class ObjLexer
{
public:
  enum { s_bufferLength = 512 };

  ObjLexer(istream* input) : 
  m_input(input), 
    m_bufferPos(0), 
    m_bufferEnd(0) 
  { 
  }

  enum Tokens
  {
    T_Eof      = 0,
    T_MtlLib   = 'm' << 8 | 't',
    T_UseMtl   = 'u' << 8 | 's',
    T_Face     = 'f' << 8 | ' ', // line starts with 'f' followed by ' '
    T_Face2    = 'f' << 8 | '\t', // line starts with 'f' followed by '\t'
    T_Vertex   = 'v' << 8 | ' ', // line starts with 'v' followed by ' '
    T_Vertex2  = 'v' << 8 | '\t', // line starts with 'v' followed by '\t'
    T_Normal   = 'v' << 8 | 'n',
    T_TexCoord = 'v' << 8 | 't',
  };

  FORCE_INLINE int fillBuffer()
  {
    if (m_bufferPos >= m_bufferEnd)
    {
      m_input->read(m_buffer, s_bufferLength);
      m_bufferEnd = int(m_input->gcount());
      m_bufferPos = 0;
    }
    return m_bufferEnd != 0;
  }

  FORCE_INLINE int nextChar()
  {
    if (fillBuffer())
    {
      return m_buffer[m_bufferPos++];
    }
    return 0;
  }

  int firstLine()
  {
    // read the first line token.
    return nextChar() << 8 | nextChar();
  }

  FORCE_INLINE int nextLine()
  {
    // scan to end of line...
    while('\n' != nextChar())
    {
    }
    while(matchChar('\n') || matchChar('\r'))
    {
    }
    // Or: convert next 2 chars to token (16 bit), can be mt, us, vn, vt, v , v\t, f , f\t, unknown
    return nextChar() << 8 | nextChar();
  }


  FORCE_INLINE bool match(const char s[], const size_t l)
  {
    for (int i = 0; fillBuffer() && i < int(l) - 1; ++i)
    {
      if (s[i] != m_buffer[m_bufferPos])
      {
        return false;
      }
      else
      {
        ++m_bufferPos;
      }
    }
    return true;
  }
  FORCE_INLINE bool matchString(std::string &str)
  {
    while (fillBuffer() && !isspace(m_buffer[m_bufferPos]))
    {
      str.push_back(m_buffer[m_bufferPos++]);
    }
    return !str.empty();
  }

  FORCE_INLINE bool matchFloat(float &result)
  {
    bool found = false;
    result = 0.0f;
    float sign = 1.0f;
    if (matchChar('-'))
    {
      sign = -1.0f;
      found = true;
    }
    char c;
    while (fillBuffer() && myIsDigit(c = m_buffer[m_bufferPos]))
    {
      result = result * 10.0f + float(c - '0'); 
      ++m_bufferPos;
      found = true;
    }
    float frac = 0.1f;
    if (matchChar('.'))
    {
      char c;
      while (fillBuffer() && myIsDigit(c = m_buffer[m_bufferPos]))
      {
        result += frac * float(c - '0'); 
        ++m_bufferPos;
        frac *= 0.1f;
      }
      found = true;
    }
    if (matchChar('e') || matchChar('E'))
    {
      float expSign = matchChar('-') ? -1.0f : 1.0f;
      int exp = 0;
      if (matchInt(exp))
      {
        result = result * powf(10.0f, float(exp) * expSign);
      }
    }
    result *= sign;
    return found;
  }

  FORCE_INLINE bool myIsDigit(char c)
  {
    return ((unsigned int)(c) - (unsigned int)('0') < 10U);
  }

  FORCE_INLINE bool matchInt(int &result)
  {
    bool found = false;
    result = 0;
    char c;
    while (fillBuffer() && myIsDigit(c = m_buffer[m_bufferPos]))// isdigit(m_buffer[m_bufferPos]))
    {
      result = result * 10 + int(c - '0'); 
      ++m_bufferPos;
      found = true;
    }
    return found;
  }
  FORCE_INLINE bool matchChar(int matchTo)
  {
    if (fillBuffer() && m_buffer[m_bufferPos] == matchTo)
    {
      m_bufferPos++;
      return true;
    }
    return false;
  }

  FORCE_INLINE bool matchWs(bool optional = false)
  {
    bool found = false;
    while (fillBuffer() && 
      (m_buffer[m_bufferPos] == ' ' || m_buffer[m_bufferPos] == '\t'))
    {
      found = true;
      m_bufferPos++;
    }
    return found || optional;
  }
  istream* m_input;
  char m_buffer[s_bufferLength];
  int m_bufferPos;
  int m_bufferEnd;
};

FORCE_INLINE static bool parseFaceIndSet(ObjLexer &lexer, ObjTri &t, int v)
{
  t.v[v] = -1;
  t.t[v] = -1;
  t.n[v] = -1;

  if(lexer.matchWs(true)
    && lexer.matchInt(t.v[v])
    &&   lexer.matchChar('/')
    &&   (lexer.matchInt(t.t[v]) || true)  // The middle index is optional!
    &&   lexer.matchChar('/')
    &&   lexer.matchInt(t.n[v]))
  {
    // need to adjust for silly obj 1 based indexing
    t.v[v] -= 1;
    t.t[v] -= 1;
    t.n[v] -= 1;
    return true;
  }
  return false;
}





bool FLATTEN OBJModel::loadOBJ(ifstream &file, std::string basePath) 
{
  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<float2> uvs;
  std::vector<ObjTri> tris;

  positions.reserve(256 * 1024);
  normals.reserve(256 * 1024);
  uvs.reserve(256 * 1024);
  tris.reserve(256 * 1024);

  std::vector<std::pair<std::string, size_t> > materialChunks;

  cout << "  Reading data..." << endl << flush;

  ObjLexer lexer(&file);
  for(int token = lexer.firstLine(); token != ObjLexer::T_Eof; token = lexer.nextLine())
  {
    switch(token)
    {
    case ObjLexer::T_MtlLib: 
      {
        string materialFile;
        if(lexer.match("llib", sizeof("llib")) && lexer.matchWs() && lexer.matchString(materialFile))
        {
          loadMaterials(basePath + materialFile, basePath);
        }
        break;
      }
    case ObjLexer::T_UseMtl: 
      {
        string materialName;
        if(lexer.match("emtl", sizeof("emtl")) && lexer.matchWs() && lexer.matchString(materialName))
        {
          if (materialChunks.size() == 0 || (*materialChunks.rbegin()).first != materialName)
          {
            materialChunks.push_back(std::make_pair(materialName, tris.size()));
          }
        }
      }
      break;
    case ObjLexer::T_Vertex:
    case ObjLexer::T_Vertex2:
      {
        float3 p;
        if (lexer.matchWs(true)
          && lexer.matchFloat(p.x)
          && lexer.matchWs()
          && lexer.matchFloat(p.y)
          && lexer.matchWs()
          && lexer.matchFloat(p.z))
        {
          positions.push_back(p);
        }
      }
      break;
    case ObjLexer::T_Normal: 
      {
        float3 n = { 0.0f, 0.0f, 0.0f };
        if (lexer.matchWs(true)
          && lexer.matchFloat(n.x)
          && lexer.matchWs()
          && lexer.matchFloat(n.y)
          && lexer.matchWs()
          && lexer.matchFloat(n.z))
        {
          normals.push_back(n);
        }
      }
      break;
    case ObjLexer::T_TexCoord: 
      {
        float2 t;
        if (lexer.matchWs(true)
          && lexer.matchFloat(t.x)
          && lexer.matchWs()
          && lexer.matchFloat(t.y))
        {
          uvs.push_back(t);
        }
      }
      break;
    case ObjLexer::T_Face: 
    case ObjLexer::T_Face2:
      {
        ObjTri t;
        // parse vert 0 and 1
        if (parseFaceIndSet(lexer, t, 0) && parseFaceIndSet(lexer, t, 1))
        {
          // let any more produce one more triangle
          while(parseFaceIndSet(lexer, t, 2))
          {
            // kick tri,
            tris.push_back(t);
            // the make last vert second (this also keeps winding the same).
            t.n[1] = t.n[2];
            t.t[1] = t.t[2];
            t.v[1] = t.v[2];
          }
        }
      }
      break;
    default:
      break;
    };
  }

  cout << "  done." << endl;

  m_aabb = make_aabb(&positions[0], positions.size());

  m_numVerts = tris.size() * 3;

  m_normals.resize(m_numVerts);
  m_positions.resize(m_numVerts);
  m_uvs.resize(m_numVerts, make_vector(0.0f, 0.0f));
  m_tangents.resize(m_numVerts, make_vector(0.0f, 1.0f, 0.0f));
  m_bitangents.resize(m_numVerts, make_vector(1.0f, 0.0f, 0.0f));

  for (size_t i = 0; i < materialChunks.size(); ++i)
  {
    Chunk chunk;
    chunk.material = &m_materials[materialChunks[i].first];
		
		// Note, no alpha supported at present.
		chunk.renderFlags = 0;
		if (chunk.material->textureId.opacity != -1)
		{
			chunk.renderFlags |= RF_AlphaTested;
		}
		else
		{
			chunk.renderFlags |= RF_Opaque;
		}

    const size_t start = materialChunks[i].second;
    const size_t end = i + 1 < materialChunks.size() ? materialChunks[i + 1].second : tris.size();

    chunk.offset = uint32_t(start * 3);
    chunk.count = uint32_t((end - start) * 3);

    for (size_t k = start; k < end; ++k)
    {
      for (int j = 0; j < 3; ++j)
      {
        m_normals[k * 3 + j] = (normals[tris[k].n[j]]);
        m_positions[k * 3 + j] = (positions[tris[k].v[j]]);
        if(tris[k].t[j] != -1)
        {
          m_uvs[k * 3 + j] = (uvs[tris[k].t[j]]);
        }
      }
      // work out tangent space..
      if(tris[k].t[0] != -1)
      {
        float3 *p = &m_positions[k * 3];
        float3 e0 = p[1] - p[0];
        float3 e1 = p[2] - p[0];

        float2 *uv = &m_uvs[k * 3];
        float2 uve0 = uv[1] - uv[0];
        float2 uve1 = uv[2] - uv[0];

        // tangent for triangle
        float3 triTangent = make_vector(uve1.y * e0.x - uve0.y * e1.x, uve1.y * e0.y - uve0.y * e1.y, uve1.y * e0.z - uve0.y * e1.z);
        for (int j = 0; j < 3; ++j)
        {
          float3 normal = m_normals[k * 3 + j];
          float3 bitangent = normalize(cross(triTangent, normal));
          float3 tangent = normalize(cross(normal, bitangent));

          m_tangents[k * 3 + j] = tangent;
          m_bitangents[k * 3 + j] = bitangent;
        }
      }
    }
    m_chunks.push_back(chunk);
  }
  cout << "done." << endl;

  glGenVertexArrays(1, &m_vaob); 
  glBindVertexArray(m_vaob);
  glGenBuffers(1, &m_positions_bo); 
  glBindBuffer(GL_ARRAY_BUFFER, m_positions_bo);
  glBufferData(GL_ARRAY_BUFFER, m_positions.size() * sizeof(float3), &m_positions[0].x, GL_STATIC_DRAW); 
  glVertexAttribPointer(AA_Position, 3, GL_FLOAT, false, 0, 0);	
  glEnableVertexAttribArray(AA_Position);

  glGenBuffers(1, &m_normals_bo); 
  glBindBuffer(GL_ARRAY_BUFFER, m_normals_bo);
  glBufferData(GL_ARRAY_BUFFER, m_normals.size() * sizeof(float3), &m_normals[0].x, GL_STATIC_DRAW); 
  glVertexAttribPointer(AA_Normal, 3, GL_FLOAT, false, 0, 0);	
  glEnableVertexAttribArray(AA_Normal);

  glGenBuffers(1, &m_uvs_bo); 
  glBindBuffer(GL_ARRAY_BUFFER, m_uvs_bo);
  glBufferData(GL_ARRAY_BUFFER, m_uvs.size() * sizeof(float2), &m_uvs[0].x, GL_STATIC_DRAW); 
  glVertexAttribPointer(AA_TexCoord, 2, GL_FLOAT, false, 0, 0);	
  glEnableVertexAttribArray(AA_TexCoord);

  glGenBuffers(1, &m_tangents_bo); 
  glBindBuffer(GL_ARRAY_BUFFER, m_tangents_bo);
  glBufferData(GL_ARRAY_BUFFER, m_tangents.size() * sizeof(float3), &m_tangents[0].x, GL_STATIC_DRAW); 
  glVertexAttribPointer(AA_Tangent, 3, GL_FLOAT, false, 0, 0);	
  glEnableVertexAttribArray(AA_Tangent);

  glGenBuffers(1, &m_bitangents_bo); 
  glBindBuffer(GL_ARRAY_BUFFER, m_bitangents_bo);
  glBufferData(GL_ARRAY_BUFFER, m_bitangents.size() * sizeof(float3), &m_bitangents[0].x, GL_STATIC_DRAW); 
  glVertexAttribPointer(AA_Bitangent, 3, GL_FLOAT, false, 0, 0);	
  glEnableVertexAttribArray(AA_Bitangent);


  // store material uniforms in buffer object.
  std::vector<MaterialProperties_Std140> tmpMaterials;
  GLint uniformBufferAlignment;
  glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniformBufferAlignment);
  //printf("uniformBufferAlignment: %d\n", uniformBufferAlignment);
  ASSERT((sizeof(MaterialProperties_Std140) % uniformBufferAlignment) == 0);

  for (std::map<std::string, Material>::iterator it = m_materials.begin(); it != m_materials.end(); ++it)
  {
    Material &m = (*it).second;
    MaterialProperties_Std140 tmp;
    tmp.diffuse_color = m.color.diffuse;
    tmp.specular_color = m.color.specular;
    tmp.emissive_color = m.color.emissive;
    tmp.specular_exponent = m.specularExponent;

    m.offset = tmpMaterials.size();

    tmpMaterials.push_back(tmp);
  }
  m_materialPropertiesBuffer.init(tmpMaterials.size(), &tmpMaterials[0], GL_STATIC_DRAW);

  return true;
}

// helper to bind texture...
static void bindTexture(int texUnit, int textureId, GLuint defaultTexture)
{
  glActiveTexture(GL_TEXTURE0 + texUnit);
  glBindTexture(GL_TEXTURE_2D, textureId != -1 ? textureId : defaultTexture);
}


void OBJModel::render(uint32_t renderFlags)
{
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  glBindVertexArray(m_vaob);

  for (size_t i = 0; i < m_chunks.size(); ++i)
  {
    Chunk &chunk = m_chunks[i];

		if (renderFlags & chunk.renderFlags)
		{
			bindTexture(TU_Diffuse, chunk.material->textureId.diffuse, m_defaultTextureOne);
			bindTexture(TU_Opacity, chunk.material->textureId.opacity, m_defaultTextureOne);
			bindTexture(TU_Specular, chunk.material->textureId.specular, m_defaultTextureOne);
			bindTexture(TU_Normal, chunk.material->textureId.normal, m_defaultNormalTexture);

			m_materialPropertiesBuffer.bindSlotRange(GL_UNIFORM_BUFFER, UBS_MaterialProperties, (uint32_t)chunk.material->offset);

			glDrawArrays(GL_TRIANGLES, chunk.offset, chunk.count);
		}
  }
  glPopAttrib();

  // deactivate texture units...
  for (int i = TU_Max - 1; i >= 0; --i)
  {
    glActiveTexture(GL_TEXTURE0 + i);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
}

bool OBJModel::loadMaterials(std::string fileName, std::string basePath )
{
  ifstream file;
  file.open(fileName.c_str());
  if (!file)
  {
    cerr << "Error in openening file '" << fileName << "'" << std::endl;
    return false;
  }
  std::string currentMaterial("");

  std::string lineread;
  while(std::getline(file, lineread)) // Read line by line
  {
    //cout << lineread << endl; 
    stringstream ss(lineread); 
    std::string firstword; 
    ss >> firstword;
    if(firstword == "newmtl")
    {
      ss >> currentMaterial;
      Material m = { { { 0.5f, 0.5f, 0.5f }, { 0.5f, 0.5f, 0.5f }, 
      { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 0.0f } }, -1 };
      m_materials[currentMaterial] = m;
      m_materials[currentMaterial].textureId.diffuse = -1;
      m_materials[currentMaterial].textureId.specular = -1;
      m_materials[currentMaterial].textureId.normal = -1;
      m_materials[currentMaterial].textureId.opacity = -1;
    }
    else if(firstword == "Ka")
    {
      float3 color;
      ss >> color.x >> color.y >> color.z;
      m_materials[currentMaterial].color.ambient = color;
    }
    else if(firstword == "Ns")
    {
      ss >> m_materials[currentMaterial].specularExponent;
    }
    else if(firstword == "Kd")
    {
      float3 color;
      ss >> color.x >> color.y >> color.z;
      m_materials[currentMaterial].color.diffuse = color;
    }
    else if(firstword == "Ks")
    {
      float3 color;
      ss >> color.x >> color.y >> color.z;
      m_materials[currentMaterial].color.specular = color;
    }
    else if(firstword == "Ke")
    {
      float3 color;
      ss >> color.x >> color.y >> color.z;
      m_materials[currentMaterial].color.emissive = color;
    }
    else if(firstword == "map_Kd")
    {
      std::string fileName; 
      ss >> fileName;
      m_materials[currentMaterial].textureId.diffuse = (int) loadTexture(basePath + fileName, basePath, true);
    }
    else if(firstword == "map_Ks")
    {
      std::string fileName; 
      ss >> fileName;
      m_materials[currentMaterial].textureId.specular = (int) loadTexture(basePath + fileName, basePath, true);
    }
    else if(firstword == "map_bump")
    {
      std::string fileName; 
      ss >> fileName;
      // read optional arg -bs 0.1234 (bump height i perhaps?)
      if (fileName == "-bm")
      {
        float tmp;
        ss >> tmp;
        // re-read file name
        ss >> fileName;
      }
      m_materials[currentMaterial].textureId.normal = (int) loadTexture(basePath + fileName, basePath, false);
    }
    else if (firstword == "map_d")
    {
      std::string fileName; 
      ss >> fileName;
      m_materials[currentMaterial].textureId.opacity = (int) loadTexture(basePath + fileName, basePath, false);
    }
  }

  // hackfix specular and diffuse colors which are set to 0 by some clever exporters
  // when there is a texture present...
  for (MatrialMap::iterator i = m_materials.begin(); i != m_materials.end(); ++i)
  {
    Material &m = (*i).second;
    if (m.textureId.specular != -1 && lengthSquared(m.color.specular) < 0.00001f)
    {
      m.color.specular = make_vector(1.0f, 1.0f, 1.0f);
    }
    if (m.textureId.diffuse != -1 && lengthSquared(m.color.diffuse) < 0.00001f)
    {
      m.color.diffuse = make_vector(1.0f, 1.0f, 1.0f);
    }
  }
  return true;
}

unsigned int OBJModel::loadTexture(std::string fileName, std::string /*basePath*/, bool srgb )
{
  fileName = normalizePath(fileName);

  ILuint image = ilGenImage();
  ilBindImage(image);
  CHECK_GL_ERROR();

  if(ilLoadImage(fileName.c_str()) == IL_FALSE)   
  {
    std::cout << "Failed to load texture: '" << fileName << "'" << std::endl;
    ILenum Error;
    while ((Error = ilGetError()) != IL_NO_ERROR) 
    {
      printf("  %d: %s\n", Error, iluErrorString(Error));
    }
    ilDeleteImage(image);
    return -1;
  }
  CHECK_GL_ERROR();
  // why not?
  if (ilTypeFromExt(fileName.c_str()) == IL_PNG || ilTypeFromExt(fileName.c_str()) == IL_JPG)
  {
    iluFlipImage();
  }
  ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);
  // super hack to work on my crappy laptop!
  //iluScale(128, 128, 1);
  GLuint texid;
  glGenTextures(1, &texid);
  glActiveTexture(GL_TEXTURE0);
  CHECK_GL_ERROR();
  glBindTexture(GL_TEXTURE_2D, texid);
  CHECK_GL_ERROR();
  int width = ilGetInteger(IL_IMAGE_WIDTH);
  int height = ilGetInteger(IL_IMAGE_HEIGHT);
  glTexImage2D(GL_TEXTURE_2D, 0, srgb ? GL_SRGB_ALPHA : GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ilGetData());
  CHECK_GL_ERROR();
  cout << "    Loaded texture '" << fileName << "', (" << ilGetInteger(IL_IMAGE_WIDTH) << "x" << ilGetInteger(IL_IMAGE_HEIGHT) << ")" << endl; 
  ilDeleteImage(image);
  CHECK_GL_ERROR();
  glGenerateMipmap(GL_TEXTURE_2D);
  CHECK_GL_ERROR();
  
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16);
  CHECK_GL_ERROR();
  glBindTexture(GL_TEXTURE_2D, 0);
  CHECK_GL_ERROR();
  return texid;
}
