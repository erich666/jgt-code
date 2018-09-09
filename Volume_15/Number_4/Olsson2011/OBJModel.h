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
#ifndef __OBJModel_h_
#define __OBJModel_h_

#include "GL/glew.h"
#include "GL/glut.h"
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <linmath/float2.h>
#include <linmath/float3.h>
#include <linmath/float4.h>
#include <linmath/Aabb.h>
#include <utils/GlBufferObject.h>


class OBJModel
{
public:
	enum RenderFlags
	{
		RF_Opaque = 1, /**< Draw chunks that are fully opaque, i.e. do not have transparency or an opacity map */
		RF_AlphaTested = 1 << 1,  /**< Draw chunks that have an opacity map */
		RF_Transparent = 1 << 2,  /**< Draw chunks that have an alpha value != 1.0f */
		RF_All = RF_Opaque | RF_AlphaTested | RF_Transparent,  /**< Draw everything. */
	};

	OBJModel(void);
	~OBJModel(void);
	/**
	 */
	void render(uint32_t renderFlags = RF_All);
	/**
	 */
	bool load(std::string fileName); 
  /**
   */
  const chag::Aabb &getAabb() const { return m_aabb; }

  // used open GL texture units, ensure they are bound appropriately for the shaders.
  enum TextureUnits
  {
    TU_Diffuse = 0,
    TU_Opacity,
    TU_Specular,
    TU_Normal,
    TU_Max,
  };
  // used in open gl, ensure they are bound appropriately for the shaders.
  enum AttributeArrays
  {
    AA_Position = 0,
    AA_Normal,
    AA_TexCoord,
    AA_Tangent,
    AA_Bitangent,
    AA_Max,
  };

  /**
   * Matrial properties are packed into an uniform buffer, should be declared thus (in the shader):
   *  layout(std140) uniform MaterialProperties
   *  {
   *    vec3 material_diffuse_color; 
   *    vec3 material_specular_color; 
   *    vec3 material_emissive_color; 
   *    float material_specular_exponent;
   *  };
   */
  enum UniformBufferSlots
  {
    UBS_MaterialProperties = 0,
    UBS_Max,
  };

protected:

	size_t getNumVerts();

	bool loadOBJ(std::ifstream &file, std::string basePath);
	bool loadMaterials(std::string fileName, std::string basePath);
	unsigned int loadTexture(std::string fileName, std::string basePath, bool srgb);

	struct Material
	{
    struct Color
    {
    chag::float3 diffuse;
		chag::float3 ambient;
		chag::float3 specular;
		chag::float3 emissive;
    } color;
    float specularExponent;
    struct TextureId
    {
      int diffuse;
      int opacity;
      int specular;
      int normal;
    } textureId;

    size_t offset;
	};


  typedef std::map<std::string, Material> MatrialMap;
	MatrialMap m_materials;

  // maps to layout of uniforms under std140 layout.
  struct MaterialProperties_Std140
  {
    chag::float3 diffuse_color; 
    float pad0;
    chag::float3 specular_color; 
    float pad1;
    chag::float3 emissive_color; 
    float specular_exponent;
    // this meets the alignment required for uniform buffer offsets on NVidia GTX280/480, also 
    // compatible with AMD integrated Radeon HD 3100.
    float alignPad[52];
  };
  GlBufferObject<MaterialProperties_Std140> m_materialPropertiesBuffer;

	struct Chunk
	{
		Material *material;
    uint32_t offset;
    uint32_t count;
		uint32_t renderFlags;
	};

  size_t m_numVerts;
	// Data on host
	std::vector<chag::float3> m_positions;
	std::vector<chag::float3> m_normals;
	std::vector<chag::float2> m_uvs; 
	std::vector<chag::float3> m_tangents;
	std::vector<chag::float3> m_bitangents;
  // Data on GPU
	GLuint	m_positions_bo; 
	GLuint	m_normals_bo; 
	GLuint	m_uvs_bo; 
	GLuint	m_tangents_bo; 
	GLuint	m_bitangents_bo; 
	// Vertex Array Object
	GLuint	m_vaob;

	std::vector<Chunk> m_chunks;

  chag::Aabb m_aabb;
  GLuint m_defaultTextureOne; /**< all 1, single pixel texture to use when no texture is loaded. */
  GLuint m_defaultNormalTexture;  /**< { 0.5, 0.5, 1, 1 }, single pixel float texture to use when no normal texture is loaded. */
};

#endif // __OBJModel_h_
