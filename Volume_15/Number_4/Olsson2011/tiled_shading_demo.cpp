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
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <IL/il.h>
#include <IL/ilu.h>

#include <stdlib.h>
#include <stdio.h>
#include <linmath/float4x4.h>
#include <utils/Rendering.h>
#include <utils/SimpleCamera.h>
#include <utils/PerformanceTimer.h>
#include <utils/CheckGlError.h>
#include <utils/GlBufferObject.h>
#include <utils/Random.h>
#include <utils/GLTimerQuery.h>
#include <utils/SimpleShader.h>
#include <utils/ComboShader.h>
#include "OBJModel.h"

#include "Light.h"
#include "LightGrid.h"

using namespace chag;

OBJModel *g_model;

const int g_startWidth  = 1280;//1920;//256;//512;//800;//3840;//1024;//128;//60;//
const int g_startHeight = 720 ;//1080;//256;//512;//600;//2160;//1024;//128;//34;//

int g_width  = g_startWidth;
int g_height = g_startHeight;

static float g_near = 0.1f;
static float g_far = 10000.0f;
const float g_fov = 45.0f;
const chag::float3 g_ambientLight = { 0.05f, 0.05f, 0.05f };


LightGrid g_lightGrid;
std::vector<Light> g_lights;
SimpleCamera g_camera;
static PerformanceTimer g_appTimer;

ComboShader *g_simpleShader = 0; 
ComboShader *g_deferredShader = 0;
ComboShader *g_tiledDeferredShader = 0;
ComboShader *g_tiledForwardShader = 0;
SimpleShader *g_downSampleMinMaxShader = 0;

// Note: GL_RGBA32F and 16x aa doesn't appear to be supported on less than Fermi, 
//       Not tested on high-end AMD cards.
GLuint g_rgbaFpFormat = GL_RGBA16F;// GL_RGBA32F;// GL_R11F_G11F_B10F;//

static bool g_showLights = false;
static bool g_showLightGrid = false;
static int g_showGBuffer = 0;
static bool g_showInfo = false;
static bool g_enablePreZ = false;
static bool g_enableDepthRangeTest = true;

static uint32_t g_numMsaaSamples = 0;
static uint32_t g_maxMsaaSamples = MAX_ALLOWED_MSAA_SAMPLES;


static std::string g_sceneFileName = "house.obj";

enum RenderMethod
{
  RM_TiledDeferred,
  RM_TiledForward,
  RM_Simple,
  RM_Max,
};
static RenderMethod g_renderMethod = RM_TiledDeferred;
static const char *g_renderMethodNames[RM_Max] = 
{
  "TiledDeferred",
  "TiledForward",
  "Simple",
};


enum TiledDeferredUniformBufferSlots
{
  TDUBS_Globals = OBJModel::UBS_Max, // use buffers that are not used by obj model.
  TDUBS_LightGrid,
  TDUBS_LightPositionsRanges,
  TDUBS_LightColors,
  TDUBS_Max,
};


enum DeferredRenderTargetIndices
{
  DRTI_Diffuse,
  DRTI_SpecularShininess,
  DRTI_Normal,
  DRTI_Ambient,
	DRTI_Depth,
  DRTI_Max,
};

enum TiledDeferredTextureUnits
{
	TDTU_LightIndexData = OBJModel::TU_Max, // avoid those used by objmodel, will make life easier for tiled forward
	TDTU_Diffuse,
  TDTU_SpecularShininess,
  TDTU_Normal,
  TDTU_Ambient,
  TDTU_Depth,
  TDTU_Max,
};

static const char *g_tiledDeferredTextureUnitNames[TDTU_Max - TDTU_LightIndexData] =
{
	"tileLightIndexListsTex",
  "diffuseTex",
  "specularShininessTex",
  "normalTex",
  "ambientTex",
  "depthTex",
};


// deferred render target.
GLuint g_deferredFbo = 0;
GLuint g_renderTargetTextures[DRTI_Max] = { 0 };
//GLuint g_depthTargetTexture = 0;
// forward MSAA render target (depth target is shared with deferred).
GLuint g_forwardFbo = 0;
GLuint g_forwardTargetTexture = 0;
// Contains depth min/max per tile, downsampled from frame buffer
GLuint g_minMaxDepthFbo = 0;
GLuint g_minMaxDepthTargetTexture = 0;
// Buffer texture that contains all the tile light lists.
GLuint g_tileLightIndexListsTexture = 0;
GlBufferObject<int> g_tileLightIndexListsBuffer;
// Buffers for grid and light data, bound to uniform blocks
GlBufferObject<int4> g_gridBuffer;
GlBufferObject<float4> g_lightPositionRangeBuffer;
GlBufferObject<float4> g_lightColorBuffer;


struct ShaderGlobals_Std140
{
	chag::float4x4 viewMatrix; 
	chag::float4x4 viewProjectionMatrix; 
	chag::float4x4 inverseProjectionMatrix;
	chag::float4x4 normalMatrix; 
	chag::float3 worldUpDirection;
  float pad0;
	chag::float3 ambientGlobal;
  float pad1;
	chag::float2 invFbSize;
	chag::int2 fbSize;
} g_shaderGlobals;


GlBufferObject<ShaderGlobals_Std140> g_shaderGlobalsGl;

static void createFbos(int width, int height);
static void createShaders();
static void bindLightGridConstants(const LightGrid &lightGrid);
static void printInfo(float fps, float gridBuildTime);
static void printString(int x, int y, const char *fmt, ...);
static void checkFBO(uint32_t fbo);
static void downSampleDepthBuffer(std::vector<float2> &depthRanges);


GLTimerQuery *g_glTimer = 0;


inline GLenum getRenderTargetTextureTargetType()
{
	return g_numMsaaSamples == 0 ? GL_TEXTURE_2D : GL_TEXTURE_2D_MULTISAMPLE;
}



void printMemStats()
{
  if (GLEW_NVX_gpu_memory_info)
  {
    int dedMem = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &dedMem);
    int totMem = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &totMem);
    int currMem = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &currMem);
    int evictCount = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX, &evictCount);
    int evictMem = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX, &evictMem);

    printf("Dedicated: %0.2fGb, total %0.2fGb, avail: %0.2fGb, evict: %0.2fGb, evictCount: %d\n", float(dedMem) / (1024.0f * 1024.0f), float(totMem) / (1024.0f * 1024.0f), float(currMem) / (1024.0f * 1024.0f), float(evictMem) / (1024.0f * 1024.0f), evictCount);
  }
}



void drawQuad2D(float x, float y, float w, float h)
{
  glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(x, y);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(x + w, y);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(x + w, y + h);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(x, y + h);
  glEnd();
}



static void glVertex3(const float3 &v)
{
  glVertex3fv(&v.x);
}



static float3 hueToRGB(float hue)
{
  const float s = hue * 6.0f;
  float r0 = clamp(s - 4.0f, 0.0f, 1.0f);
  float g0 = clamp(s - 0.0f, 0.0f, 1.0f);
  float b0 = clamp(s - 2.0f, 0.0f, 1.0f);

  float r1 = clamp(2.0f - s, 0.0f, 1.0f);
  float g1 = clamp(4.0f - s, 0.0f, 1.0f);
  float b1 = clamp(6.0f - s, 0.0f, 1.0f);

  // annoying that it wont quite vectorize...
  return make_vector(r0 + r1, g0 * g1, b0 * b1);
}



static void renderTiledDeferred(LightGrid &grid, const float4x4 &projectionMatrix)
{
  {
    glViewport(0, 0, g_width, g_height);
    CHECK_GL_ERROR();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    bindLightGridConstants(grid);

    g_tiledDeferredShader->begin(false);

    glBegin(GL_QUADS);
      glVertex2f(-1.0f, -1.0f);
      glVertex2f(1.0f, -1.0f);
      glVertex2f(1.0f, 1.0f);
      glVertex2f(-1.0f, 1.0f);
    glEnd();

    g_tiledDeferredShader->end();

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glDepthMask(GL_TRUE);

    glPopAttrib();
  }
}

static void updateShaderGlobals(const chag::float4x4 &viewMatrix, const chag::float4x4 &projectionMatrix, const chag::float3 &worldUpDirection, int width, int height)
{
	g_shaderGlobals.viewMatrix = viewMatrix;
	g_shaderGlobals.viewProjectionMatrix = projectionMatrix * viewMatrix;
	g_shaderGlobals.inverseProjectionMatrix = inverse(projectionMatrix);
	g_shaderGlobals.normalMatrix = transpose(inverse(viewMatrix));
	g_shaderGlobals.worldUpDirection = worldUpDirection;
	g_shaderGlobals.ambientGlobal = g_ambientLight;
	g_shaderGlobals.invFbSize = make_vector(1.0f / float(width), 1.0f / float(height));
	g_shaderGlobals.fbSize = make_vector(width, height);

	// copy to Gl
	g_shaderGlobalsGl.copyFromHost(&g_shaderGlobals, 1);
	// bind buffer.
  g_shaderGlobalsGl.bindSlot(GL_UNIFORM_BUFFER, TDUBS_Globals);
}


static void drawPreZPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, g_forwardFbo);
	glViewport(0,0, g_width, g_height);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);
	if (g_numMsaaSamples != 0)
	{
		glEnable(GL_MULTISAMPLE);
	}
	glDepthFunc(GL_LEQUAL);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	g_simpleShader->begin(false);
	g_model->render(OBJModel::RF_Opaque);
	g_simpleShader->end();
	g_simpleShader->begin(true);
	g_model->render(OBJModel::RF_AlphaTested);
	g_simpleShader->end();

	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}


static void onGlutDisplay()
{
  CHECK_GL_ERROR();
  g_glTimer->start();
  CHECK_GL_ERROR();

  float4x4 projection = perspectiveMatrix(g_fov, float(g_width) / float(g_height), g_near, g_far);
  float4x4 modelView = lookAt(g_camera.getPosition(), g_camera.getPosition() + g_camera.getDirection(), g_camera.getUp());

	updateShaderGlobals(modelView, projection, g_camera.getWorldUp(), g_width, g_height);

  PerformanceTimer buildTimer;
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  switch(g_renderMethod)
  {
    case RM_TiledDeferred:
    {
      CHECK_GL_ERROR();
      // 0. pre-z pass
      if (g_enablePreZ)
      {
				drawPreZPass();
      }
      // 1. deferred render model.
      CHECK_GL_ERROR();
      glBindFramebuffer(GL_FRAMEBUFFER, g_deferredFbo);

        CHECK_GL_ERROR();
        glViewport(0,0, g_width, g_height);
        CHECK_GL_ERROR();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        CHECK_GL_ERROR();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        CHECK_GL_ERROR();

        g_deferredShader->begin(false);
				g_model->render(OBJModel::RF_Opaque);
        g_deferredShader->end();
				g_deferredShader->begin(true);
				g_model->render(OBJModel::RF_AlphaTested);
        g_deferredShader->end();
      CHECK_GL_ERROR();

			buildTimer.start();
      // 2. build grid
			std::vector<float2> depthRanges;
			if (g_enableDepthRangeTest)
			{
				downSampleDepthBuffer(depthRanges);
			}
      g_lightGrid.build(
        make_vector<uint32_t>(LIGHT_GRID_TILE_DIM_X, LIGHT_GRID_TILE_DIM_Y), 
        make_vector<uint32_t>(g_width, g_height),
        g_lights,
        modelView,
        projection,
        g_near,
        depthRanges
        );
      buildTimer.stop();

      // 3. apply tiled deferred lighting
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0,0, g_width, g_height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | (g_enablePreZ ? 0 : GL_DEPTH_BUFFER_BIT));

        renderTiledDeferred(g_lightGrid, projection);
    }
    break;
    case RM_TiledForward:
    {
      // 0. Pre-Z pass
      if (g_enablePreZ)
      {
				drawPreZPass();
      }

      // 1. build grid
			buildTimer.start();
			std::vector<float2> depthRanges;
			if (g_enableDepthRangeTest && g_enablePreZ)
			{
				downSampleDepthBuffer(depthRanges);
			}

      g_lightGrid.build(
        make_vector<uint32_t>(LIGHT_GRID_TILE_DIM_X, LIGHT_GRID_TILE_DIM_Y), 
        make_vector<uint32_t>(g_width, g_height),
        g_lights,
        modelView,
        projection,
        g_near,
        depthRanges
        );
      buildTimer.stop();
      CHECK_GL_ERROR();

      CHECK_GL_ERROR();

      // 2. Render scene using light info from grid.
      bindLightGridConstants(g_lightGrid);
      CHECK_GL_ERROR();
			if (g_numMsaaSamples != 0 || g_enablePreZ)
      {
        glBindFramebuffer(GL_FRAMEBUFFER, g_forwardFbo);
				if (g_numMsaaSamples != 0)
				{
					glEnable(GL_MULTISAMPLE);
				}
      }
      else
      {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
      }

      CHECK_GL_ERROR();
      glViewport(0,0, g_width, g_height);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

			// if pre-z is enabled, we don't want to re-clear the frame buffer.
			glClear(GL_COLOR_BUFFER_BIT | (g_enablePreZ ? 0 : GL_DEPTH_BUFFER_BIT));
			
      CHECK_GL_ERROR();

      g_tiledForwardShader->begin(false);
			g_model->render(OBJModel::RF_Opaque);
      g_tiledForwardShader->end();
			g_tiledForwardShader->begin(true);
			g_model->render(OBJModel::RF_AlphaTested);
      g_tiledForwardShader->end();

      CHECK_GL_ERROR();
      
      if (g_numMsaaSamples != 0 || g_enablePreZ)
      {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, g_forwardFbo);
        CHECK_GL_ERROR();
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, g_width, g_height, 0, 0, g_width, g_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        CHECK_GL_ERROR();
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        CHECK_GL_ERROR();
      }
    }
    break;
    case RM_Simple:
    {
			buildTimer.start();
      buildTimer.stop();
 
      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      glViewport(0,0, g_width, g_height);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      g_simpleShader->begin(false);
			g_model->render(OBJModel::RF_Opaque);
      g_simpleShader->end();
			g_simpleShader->begin(true);
			g_model->render(OBJModel::RF_AlphaTested);
      g_simpleShader->end();
    }
    break;
  };
  glPopAttrib();

  if (g_showLights)
  {
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(&projection.c1.x);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(&modelView.c1.x);
    glDisable(GL_LIGHTING);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    for (size_t i = 0; i < g_lights.size(); ++i)
    {
      const Light &l = g_lights[i];
      glColor4f(l.color.x, l.color.y, l.color.z, 0.1f);
      glPushMatrix();
        glTranslatef(l.position.x, l.position.y, l.position.z);
        glutSolidSphere(l.range, 8, 8);
      glPopMatrix();
      //glBegin(GL_LINES);
      //  glVertex3fv(&l.position.x);
      //  float3 origo = { 0.0f, 0.0f, 0.0f };
      //  glVertex3fv(&origo.x);
      //glEnd();
    }

    glPopAttrib();
  }

  if (g_showLightGrid)
  {
    glViewport(0, 0, g_width, g_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, float(g_width), 0.0, float(g_height), -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_LIGHTING);

    glBegin(GL_QUADS);
    const int border = 2;
    uint32_t offset = 0;
    const LightGrid &lightGrid = g_lightGrid;
    const uint2 tileSize = lightGrid.getTileSize();
    float maxCount = float(lightGrid.getMaxTileLightCount());
    //printf("%d - %d\n", int(maxCount), lightGrid.getTotalTileLightIndexListLength());

    for (uint32_t y = 0; y < lightGrid.getGridDim().y; ++y)
    {
      for (uint32_t x = 0; x < lightGrid.getGridDim().x; ++x)
      {
        int count = lightGrid.tileLightCount(x, y);
        if (count)
        {
          float depth = 0.0f;//lightGrid.minMaxGridValid() ? lightGrid.getTileMinMax(x,y).x : 0.0f;
          const int *lightIds = lightGrid.tileLightIndexList(x, y); 

          float3 c = make_vector(0.0f, 0.0f, 0.0f);
          for (int i = 0; i < count; ++i)
          {
            int id = lightIds[i];
            ASSERT(size_t(id) < lightGrid.getViewSpaceLights().size());
            Light l = lightGrid.getViewSpaceLights()[id];
            //glColor4f(1.0f, 0.5f, 0.5f, float(count) / maxCount);
            c += max(l.color / float(maxCount), make_vector3(1.0f / 255.0f));
          }
          c = c * 0.9f + 0.1f;
          glColor4f(c.x, c.y, c.z, 1.0f / maxCount);
          //glColor4f(1,1,1,1);

          glVertex3f(float(x * tileSize.x + border), float(y * tileSize.y + border), depth);
          glVertex3f(float((x + 1) * tileSize.x - border), float(y * tileSize.y + border), depth);

          glVertex3f(float((x + 1) * tileSize.x - border), float(y * tileSize.y + border), depth);
          glVertex3f(float((x + 1) * tileSize.x - border), float((y + 1)  * tileSize.y - border), depth);

          glVertex3f(float((x + 1) * tileSize.x - border), float((y + 1)  * tileSize.y - border), depth);
          glVertex3f(float(x * tileSize.x + border), float((y + 1)  * tileSize.y - border), depth);

          glVertex3f(float(x * tileSize.x + border), float((y + 1)  * tileSize.y - border), depth);
          glVertex3f(float(x * tileSize.x + border), float(y * tileSize.y + border), depth);
        }
      }
    }

    glEnd();

    glDepthMask(GL_TRUE);
    glPopAttrib();
    CHECK_GL_ERROR();
  }

  // blit gbuffer...
  if (g_showGBuffer != 0)
  {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, g_deferredFbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0 + g_showGBuffer - 1);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, g_width, g_height, 0, 0, g_width, g_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    CHECK_GL_ERROR();
  }

  g_glTimer->stop();

  printInfo(1000.0f / g_glTimer->getAvgMs(), float(buildTimer.getElapsedTime() * 1000.0));
 
  glutSwapBuffers();
}

static void onGlutIdle()
{
  float dt = float(g_appTimer.getElapsedTime());
  g_appTimer.restart();
  g_camera.update(dt);
  glutPostRedisplay();
}


static std::vector<Light> generateLights(const Aabb &aabb, int num)
{
  std::vector<Light> result;
  // divide volume equally amongst lights
  const float lightVol = aabb.getVolume() / float(num);
  // set radius to be the cube root of volume for a light
  const float lightRad = pow(lightVol, 1.0f / 3.0f);
  // and allow some overlap
  const float maxRad = lightRad;// * 2.0f;
  const float minRad = lightRad;// / 2.0f;

  for (int i = 0; i < num; ++i)
  {
    float rad = randomRange(minRad, maxRad);
    float3 col = hueToRGB(randomUnitFloat()) * randomRange(0.4f, 0.7f);
    //float3 pos = { randomRange(aabb.min.x + rad, aabb.max.x - rad), randomRange(aabb.min.y + rad, aabb.max.y - rad), randomRange(aabb.min.z + rad, aabb.max.z - rad) };
    const float ind =  rad / 8.0f;
    float3 pos = { randomRange(aabb.min.x + ind, aabb.max.x - ind), randomRange(aabb.min.y + ind, aabb.max.y - ind), randomRange(aabb.min.z + ind, aabb.max.z - ind) };
    Light l = { pos, col, rad };

    result.push_back(l);
  }
  return result;
}


static void onGlutKeyboard(unsigned char key, int, int)
{
  g_camera.setMoveVelocity(length(g_model->getAabb().getHalfSize()) / 10.0f);

  if (g_camera.handleKeyInput(key, true))
  {
    return;
  }

  switch(tolower(key))
  {
  case 27:
    glutLeaveMainLoop();
    break;
  case 'r':
    g_lights = generateLights(g_model->getAabb(), int(g_lights.size()));
    break;
  case 't':
		g_showGBuffer = (g_showGBuffer + 1) % (DRTI_Depth + 1);
    break;
  case 'l':
    createShaders();
    break;
  case 'p':
    g_showLights = !g_showLights;
    break;
  case 'g':
    g_showLightGrid = !g_showLightGrid;
    break;
  case 'm':
    g_renderMethod = RenderMethod((g_renderMethod + 1) % RM_Max);
    break;
  case 'c':
		{
			if (g_numMsaaSamples == 0)
			{
				g_numMsaaSamples = 2;
			}
			else
			{
				g_numMsaaSamples <<= 1;
			}
			if (g_numMsaaSamples > MAX_ALLOWED_MSAA_SAMPLES)
			{
				g_numMsaaSamples = 0;
			}
			// now we must recompile shaders, and re-create FBOs
			createShaders();
			createFbos(g_width, g_height);
		}
    break;
  case 'u':
    {
      float3 pos = g_camera.getPosition();
      static int upAxis = 0;
      static float3 ups[]  = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };
      static float3 fwds[] = { { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f } };
      g_camera.init(ups[upAxis], fwds[upAxis], pos);
      upAxis = (upAxis + 1) % 3;
    }
  case 'o':
    g_enablePreZ = !g_enablePreZ;
    break;
  case 'z':
    g_enableDepthRangeTest = !g_enableDepthRangeTest;
    break;
	case '+':
    {
      int newCount = min(int(NUM_POSSIBLE_LIGHTS), int(g_lights.size()) + NUM_POSSIBLE_LIGHTS / 8);
      g_lights = generateLights(g_model->getAabb(), newCount);
    }
    break;
  case '-':
    {
      int newCount = max(0, int(g_lights.size()) - NUM_POSSIBLE_LIGHTS / 8);
      g_lights = generateLights(g_model->getAabb(), newCount);
    }
    break;
  }
}



static void onGlutKeyboardUp(unsigned char _key, int, int)
{
  if (g_camera.handleKeyInput(_key, false))
  {
    return;
  }
}



static void onGlutSpecial(int key, int, int)
{
  if (key == GLUT_KEY_F1)
  {
    g_showInfo = !g_showInfo;
  }
}



static void onGlutMouse(int, int, int, int)
{
  g_camera.resetMouseInputPosition();
}



static void onGlutMotion(int x, int y)
{
  g_camera.handleMouseInput(make_vector(x,y));
}



static void onGlutReshape(int width, int height)
{
  if (g_width != width || g_height != height)
  {
    createFbos(width, height);
  }
  g_width = width;
  g_height = height;
 }



int main(int argc, char* argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  glutInitWindowSize(g_width, g_height);
  glutCreateWindow("Tiled Shading Demo");

  glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glutSwapBuffers();

  glewInit();

  printf("--------------------------------------\nOpenGL\n  Vendor: %s\n  Renderer: %s\n  Version: %s\n--------------------------------------\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION));

  if (!GLEW_VERSION_3_3)
  {
    fprintf(stderr, "This demo assumes that Open GL 3.3 is present.\n"
      "It may possibly run with less, so feel free to remove\n"
      "the check, and see what happens.\n");
    return 1;
  }

  int v = 0;
  glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &v);
  printf("GL_MAX_UNIFORM_BLOCK_SIZE: %d\n", v);
  glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &v);
  printf("GL_MAX_COLOR_TEXTURE_SAMPLES: %d\n", v);
  glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &v);
  printf("GL_MAX_COLOR_TEXTURE_SAMPLES: %d\n", v);
	g_maxMsaaSamples = min<uint32_t>(g_maxMsaaSamples, MAX_ALLOWED_MSAA_SAMPLES);
  printf("--------------------------------------\n");

  ilInit();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);
  glDisable(GL_DITHER);
  glDisable(GL_ALPHA_TEST);
  
  wglSwapIntervalEXT(0);

  glutDisplayFunc(onGlutDisplay);
  glutIdleFunc(onGlutIdle);
  glutKeyboardFunc(onGlutKeyboard);
  glutSpecialFunc(onGlutSpecial);
  glutKeyboardUpFunc(onGlutKeyboardUp);
  glutMouseFunc(onGlutMouse);
  glutMotionFunc(onGlutMotion);
  glutReshapeFunc(onGlutReshape);

  if (argc > 1)
  {
    g_sceneFileName = argv[1];
  }

  createFbos(g_width, g_height);
  createShaders();

  g_gridBuffer.init(LIGHT_GRID_MAX_DIM_X * LIGHT_GRID_MAX_DIM_Y, 0);

  g_lightPositionRangeBuffer.init(NUM_POSSIBLE_LIGHTS);
  g_lightColorBuffer.init(NUM_POSSIBLE_LIGHTS);

  glGenTextures(1, &g_tileLightIndexListsTexture);
  // initial size is 1, because glTexBuffer failed if it was empty, we'll shovel in data later.
  g_tileLightIndexListsBuffer.init(1);
  glBindTexture(GL_TEXTURE_BUFFER, g_tileLightIndexListsTexture);
  glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, g_tileLightIndexListsBuffer);
  CHECK_GL_ERROR();

	g_shaderGlobalsGl.init(1, 0, GL_DYNAMIC_DRAW);

  g_model = new OBJModel();
  if (!g_model->load(g_sceneFileName))
  {
    fprintf(stderr, "The file: '%s' could not be loaded.\n", g_sceneFileName.c_str());
    return 1;
  }
  //g_camera.init(make_vector(0.0f, 0.0f, 1.0f), make_vector(0.0f, 1.0f, 0.0f));
  g_camera.init(make_vector(0.0f, 1.0f, 0.0f), make_vector(0.0f, 0.0f, 1.0f));

	g_far = length(g_model->getAabb().getHalfSize()) * 3.0f;
	g_near = g_far / 1000.0f;


  g_lights = generateLights(g_model->getAabb(), NUM_POSSIBLE_LIGHTS);

  g_glTimer = new GLTimerQuery;

  g_appTimer.start();
  glutMainLoop();

	return 0;
}




// helper function to create and attach a frame buffer target object. 
static GLuint attachTargetTextureToFBO(GLuint fbo, GLenum attachment, int width, int height, GLenum internalFormat, GLenum format, GLenum type = GL_FLOAT, int msaaSamples = 0)
{
  GLuint targetTexture;
  glGenTextures(1, &targetTexture);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  CHECK_GL_ERROR();

  if (msaaSamples == 0)
  {
    glBindTexture(GL_TEXTURE_2D, targetTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, NULL);

    CHECK_GL_ERROR();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, targetTexture, 0);
    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, 0);
  }
  else
  {
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, targetTexture);
    CHECK_GL_ERROR();

    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, msaaSamples, internalFormat, width, height, false);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D_MULTISAMPLE, targetTexture, 0);

    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
  }
  glBindTexture(GL_TEXTURE_2D, 0);


  return targetTexture;
}




static void checkFBO(uint32_t fbo)
{
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
  {
    switch(glCheckFramebufferStatus(GL_FRAMEBUFFER))
    {
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
          printf("FRAMEBUFFER_INCOMPLETE_ATTACHMENT\n");
          break;
      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
          printf("FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT\n");
          break;
      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
          printf("FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER\n");
          break;
      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
          printf("FRAMEBUFFER_INCOMPLETE_READ_BUFFER\n");
          break;
      case GL_FRAMEBUFFER_UNSUPPORTED:
          printf("FRAMEBUFFER_UNSUPPORTED\n");
          break;
      default:
          printf("Unknown framebuffer problem %d\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
          break;
    }
    printf("Error: bad frame buffer config\n");
    DBG_BREAK();
    exit(-1);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


static void deleteTextureIfUsed(GLuint texId)
{
  if (texId != 0)
  {
    glDeleteTextures(1, &texId);
  }
}

static void createFbos(int width, int height)
{
  int maxSamples = 0;
  glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);

  for (int i = 0; i < DRTI_Max; ++i)
  {
    deleteTextureIfUsed(g_renderTargetTextures[i]);
  }

  // deferred render target
  if (!g_deferredFbo)
  {
    // only create if not already created.
    glGenFramebuffers(1, &g_deferredFbo);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, g_deferredFbo);
	for (int i = 0; i < DRTI_Depth; ++i)
  {
    g_renderTargetTextures[i] = attachTargetTextureToFBO(g_deferredFbo, GL_COLOR_ATTACHMENT0 + i, width, height, g_rgbaFpFormat, GL_RGBA, GL_FLOAT, g_numMsaaSamples);
  }
  g_renderTargetTextures[DRTI_Depth] = attachTargetTextureToFBO(g_deferredFbo, GL_DEPTH_ATTACHMENT, width, height, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, g_numMsaaSamples);

  CHECK_GL_ERROR();
  GLenum bufs[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
  glDrawBuffers(sizeof(bufs) / sizeof(bufs[0]), bufs);
	glReadBuffer(GL_NONE); 
  
  printMemStats();

  checkFBO(g_deferredFbo);

  // forward shading render target
  if (!g_forwardFbo)
  {
    // only create if not already created.
    glGenFramebuffers(1, &g_forwardFbo);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, g_forwardFbo);
  g_forwardTargetTexture = attachTargetTextureToFBO(g_forwardFbo, GL_COLOR_ATTACHMENT0, width, height, GL_RGBA, GL_RGBA, GL_FLOAT, g_numMsaaSamples);
  // Shared with deferred
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, getRenderTargetTextureTargetType(), g_renderTargetTextures[DRTI_Depth], 0);


  glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glReadBuffer(GL_COLOR_ATTACHMENT0); 
  
  checkFBO(g_forwardFbo);

  if (!g_minMaxDepthFbo)
  {
    // only create if not already created.
    glGenFramebuffers(1, &g_minMaxDepthFbo);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, g_minMaxDepthFbo);

	uint2 tileSize = make_vector<uint32_t>(LIGHT_GRID_TILE_DIM_X, LIGHT_GRID_TILE_DIM_Y);
  uint2 resolution = make_vector<uint32_t>(width, height);
	uint2 gridRes = (resolution + tileSize - 1) / tileSize;

	g_minMaxDepthTargetTexture = attachTargetTextureToFBO(g_minMaxDepthFbo, GL_COLOR_ATTACHMENT0, gridRes.x, gridRes.y, GL_RG32F, GL_RGBA, GL_FLOAT, 0);

  glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glReadBuffer(GL_COLOR_ATTACHMENT0); 
  
  checkFBO(g_minMaxDepthFbo);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void bindObjModelAttributes(ComboShader *shader)
{
  shader->bindAttribLocation(OBJModel::AA_Position, "position");
  shader->bindAttribLocation(OBJModel::AA_Normal, "normalIn");
  shader->bindAttribLocation(OBJModel::AA_TexCoord, "texCoordIn");
  shader->bindAttribLocation(OBJModel::AA_Tangent, "tangentIn");
  shader->bindAttribLocation(OBJModel::AA_Bitangent, "bitangentIn");
}

static void setObjModelUniformBindings(ComboShader *shader)
{
  shader->begin(false);
    shader->setUniform("diffuse_texture", OBJModel::TU_Diffuse);
    shader->setUniform("opacity_texture", OBJModel::TU_Opacity);
    shader->setUniform("specular_texture", OBJModel::TU_Specular);
    shader->setUniform("normal_texture", OBJModel::TU_Normal);
  shader->end();
  shader->setUniformBufferSlot("MaterialProperties", OBJModel::UBS_MaterialProperties);
 
	shader->setUniformBufferSlot("Globals", TDUBS_Globals);
}


static void setTiledLightingUniformBindings(ComboShader *shader)
{
  shader->setUniformBufferSlot("Globals", TDUBS_Globals);
  shader->setUniformBufferSlot("LightGrid", TDUBS_LightGrid);
  shader->setUniformBufferSlot("LightPositionsRanges", TDUBS_LightPositionsRanges);
  shader->setUniformBufferSlot("LightColors", TDUBS_LightColors);

	shader->begin(false);

	for (int i = TDTU_LightIndexData; i < TDTU_Max; ++i)
	{
    shader->setUniform(g_tiledDeferredTextureUnitNames[i - TDTU_LightIndexData], i);
	}
  shader->end();
}
template <typename T>
static void deleteIfThere(T *&shader)
{
	if (shader)
	{
		delete shader;
		shader = 0;
	}
}

static void createShaders()
{
  SimpleShader::Context shaderCtx;
  shaderCtx.setPreprocDef("NUM_POSSIBLE_LIGHTS", NUM_POSSIBLE_LIGHTS);
  shaderCtx.setPreprocDef("LIGHT_GRID_TILE_DIM_X", LIGHT_GRID_TILE_DIM_X);
  shaderCtx.setPreprocDef("LIGHT_GRID_TILE_DIM_Y", LIGHT_GRID_TILE_DIM_Y);
  shaderCtx.setPreprocDef("LIGHT_GRID_MAX_DIM_X", LIGHT_GRID_MAX_DIM_X);
  shaderCtx.setPreprocDef("LIGHT_GRID_MAX_DIM_Y", LIGHT_GRID_MAX_DIM_Y);
  shaderCtx.setPreprocDef("NUM_MSAA_SAMPLES", int(g_numMsaaSamples));

	deleteIfThere(g_simpleShader);
  g_simpleShader = new ComboShader("shaders/simple_vertex.glsl", "shaders/simple_fragment.glsl", shaderCtx);
    bindObjModelAttributes(g_simpleShader);
    g_simpleShader->bindFragDataLocation(0, "resultColor");
  g_simpleShader->link();
  setObjModelUniformBindings(g_simpleShader);

  // deferred shader
	deleteIfThere(g_deferredShader);
  g_deferredShader = new ComboShader("shaders/deferred_vertex.glsl", "shaders/deferred_fragment.glsl", shaderCtx);
    bindObjModelAttributes(g_deferredShader);

    g_deferredShader->bindFragDataLocation(DRTI_Diffuse, "outDiffuse");
    g_deferredShader->bindFragDataLocation(DRTI_SpecularShininess, "outSpecularShininess");
    g_deferredShader->bindFragDataLocation(DRTI_Normal, "outNormal");
    g_deferredShader->bindFragDataLocation(DRTI_Ambient, "outAmbient");
  g_deferredShader->link();
  setObjModelUniformBindings(g_deferredShader);

  // tiled deferred shader
	deleteIfThere(g_tiledDeferredShader);
  g_tiledDeferredShader = new ComboShader("shaders/tiled_deferred_vertex.glsl", "shaders/tiled_deferred_fragment.glsl", shaderCtx);
    g_tiledDeferredShader->bindFragDataLocation(0, "resultColor");
  g_tiledDeferredShader->link();
  setTiledLightingUniformBindings(g_tiledDeferredShader);

  // tiled forward shader
	deleteIfThere(g_tiledForwardShader);
  g_tiledForwardShader = new ComboShader("shaders/tiled_forward_vertex.glsl", "shaders/tiled_forward_fragment.glsl", shaderCtx);
    bindObjModelAttributes(g_tiledForwardShader);
    g_tiledForwardShader->bindFragDataLocation(0, "resultColor");
  g_tiledForwardShader->link();
  setObjModelUniformBindings(g_tiledForwardShader);
  setTiledLightingUniformBindings(g_tiledForwardShader);

  // downsample min/max shader
	deleteIfThere(g_downSampleMinMaxShader);
  g_downSampleMinMaxShader = new SimpleShader("shaders/tiled_deferred_vertex.glsl", "shaders/downsample_minmax_fragment.glsl", shaderCtx);
    g_downSampleMinMaxShader->bindFragDataLocation(0, "resultMinMax");
  g_downSampleMinMaxShader->link();
  g_downSampleMinMaxShader->setUniformBufferSlot("Globals", TDUBS_Globals);
}

// helper to bind texture...
static void bindTexture(GLenum type, int texUnit, int textureId)
{
  glActiveTexture(GL_TEXTURE0 + texUnit);
  glBindTexture(type, textureId);
}

static void bindLightGridConstants(const LightGrid &lightGrid)
{
  // pack grid data in int4 because this will work on AMD GPUs, where constant registers are 4-vectors.
  static chag::int4 tmp[LIGHT_GRID_MAX_DIM_X * LIGHT_GRID_MAX_DIM_Y];
  {
    const int *counts = lightGrid.tileCountsDataPtr();
    const int *offsets = lightGrid.tileDataPtr();
    for (int i = 0; i < LIGHT_GRID_MAX_DIM_X * LIGHT_GRID_MAX_DIM_Y; ++i)
    {
      tmp[i] = chag::make_vector(counts[i], offsets[i], 0, 0);
    }
  }
  g_gridBuffer.copyFromHost(tmp, LIGHT_GRID_MAX_DIM_X * LIGHT_GRID_MAX_DIM_Y);

  if (lightGrid.getTotalTileLightIndexListLength())
  {
    g_tileLightIndexListsBuffer.copyFromHost(lightGrid.tileLightIndexListsPtr(), lightGrid.getTotalTileLightIndexListLength());
    // This should not be neccessary, but for amd it seems to be (HD3200 integrated)
    glBindTexture(GL_TEXTURE_BUFFER, g_tileLightIndexListsTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, g_tileLightIndexListsBuffer);
    CHECK_GL_ERROR();
  }
	bindTexture(GL_TEXTURE_BUFFER, TDTU_LightIndexData, g_tileLightIndexListsTexture);

  const Lights &lights = lightGrid.getViewSpaceLights();

  const size_t maxLights = NUM_POSSIBLE_LIGHTS;
  static float4 light_position_range[maxLights];
  static float4 light_color[maxLights];

  memset(light_position_range, 0, sizeof(light_position_range));
  memset(light_color, 0, sizeof(light_color));

  for (size_t i = 0; i < std::min(maxLights, lights.size()); ++i)
  {
    light_position_range[i] = make_vector4(lights[i].position, lights[i].range);
    light_color[i] = make_vector4(lights[i].color, 1.0f);
  }
  g_lightPositionRangeBuffer.copyFromHost(light_position_range, NUM_POSSIBLE_LIGHTS);
  g_lightColorBuffer.copyFromHost(light_color, NUM_POSSIBLE_LIGHTS);

  g_gridBuffer.bindSlot(GL_UNIFORM_BUFFER, TDUBS_LightGrid);
  g_lightPositionRangeBuffer.bindSlot(GL_UNIFORM_BUFFER, TDUBS_LightPositionsRanges);
  g_lightColorBuffer.bindSlot(GL_UNIFORM_BUFFER, TDUBS_LightColors);

  for (int i = 0; i < DRTI_Max; ++i)
  {
		bindTexture(getRenderTargetTextureTargetType(), i + TDTU_Diffuse, g_renderTargetTextures[i]);
  }
}


static void printString(int x, int y, const char *fmt, ...)
{
	char text[256];
	va_list		ap;
	va_start(ap, fmt);
  vsnprintf( text, 256, fmt, ap );
	va_end(ap);

  glRasterPos2i(x, y);
	for(size_t i = 0; i < strlen(text); ++i)
  {
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, text[i]); 
	}
}


static void printInfo(float fps, float gridBuildTime)
{
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  glViewport(0, 0, g_width, g_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, float(g_width), float(g_height), 0.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDepthMask(GL_FALSE);
  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);

  glDisable(GL_TEXTURE_2D);

  if (g_showInfo)
  {
    int yOffset = 10;
    int yStep = 13;
    glColor3f(1.0f, 1.0f, 1.0f);
    printString(10, yOffset, "Toggle info: <F1>");
    printString(10, yOffset += yStep, "Method ('m'): %s", g_renderMethodNames[g_renderMethod]);
    printString(10, yOffset += yStep, "FPS: %6.2f", fps);
    printString(10, yOffset += yStep, "Grid Build: %6.2fms", gridBuildTime);
		char msaaBuffer[64];
		sprintf(msaaBuffer, "%dx", g_numMsaaSamples);
		printString(10, yOffset += yStep, "MSAA Level ('c'): %s (Max: %dx)", g_numMsaaSamples == 0 ? "Off" : msaaBuffer, g_maxMsaaSamples);
    printString(10, yOffset += yStep, "Show Lights ('p'): %s", g_showLights ? "On" : "Off");
    printString(10, yOffset += yStep, "Show Light Grid ('g'): %s", g_showLightGrid ? "On" : "Off");
    printString(10, yOffset += yStep, "Toggle PreZ ('o'): %s", g_enablePreZ ? "On" : "Off");
    printString(10, yOffset += yStep, "Toggle Depth Range Test ('z'): %s", g_enableDepthRangeTest ? "On" : "Off");
    printString(10, yOffset += yStep, "Add/Remove %d Lights '+/-'", NUM_POSSIBLE_LIGHTS / 8);
    const char *gBufferNames[] = 
    {
      "Off",
      "Diffuse",
      "SpecularShininess",
      "Normal",
      "Ambient",
    };
    printString(10, yOffset += yStep, "Shown G-Buffer ('t'): %s", gBufferNames[g_showGBuffer]);
    printString(10, yOffset += yStep, "Re-load Shaders ('l')");
    printString(10, yOffset += yStep, "Cycle Up direction ('u')");
    printString(10, yOffset += yStep, "Scene file name: %s", g_sceneFileName.c_str());
    printString(10, yOffset += yStep, "G-Buffer Format: %s", g_rgbaFpFormat == GL_RGBA16F ? "GL_RGBA16F" : (g_rgbaFpFormat == GL_RGBA32F ? "GL_RGBA32F" : "Unknown"));


    const uint2 tileSize = g_lightGrid.getTileSize();
    printString(10, yOffset += yStep, "Tile Size: {%d,%d}", tileSize.x, tileSize.y);
    const uint2 gridRes = g_lightGrid.getGridDim();
    printString(10, yOffset += yStep, "Grid Resolution: {%d,%d}", gridRes.x, gridRes.y);
    printString(10, yOffset += yStep, "Max Tile Light Count: %d", g_lightGrid.getMaxTileLightCount());
    printString(10, yOffset += yStep, "Total Light List Length: %d", g_lightGrid.getTotalTileLightIndexListLength());
    printString(10, yOffset += yStep, "Num Lights: %d", int(g_lights.size()));
  }
  else
  {
    glColor3f(0.4f, 0.4f, 0.4f);
    printString(10, 10, "Toggle info: <F1> - FPS: %6.2f", fps);
  }
  glPopAttrib();
}


static void downSampleDepthBuffer(std::vector<float2> &depthRanges)
{
	glBindFramebuffer(GL_FRAMEBUFFER, g_minMaxDepthFbo);

		uint2 tileSize = make_vector<uint32_t>(LIGHT_GRID_TILE_DIM_X, LIGHT_GRID_TILE_DIM_Y);
		uint2 resolution = make_vector<uint32_t>(g_width, g_height);
		uint2 gridRes = (resolution + tileSize - 1) / tileSize;
	
		glViewport(0, 0, gridRes.x, gridRes.y);
		glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		CHECK_GL_ERROR();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
    
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		g_downSampleMinMaxShader->begin();
		if (g_numMsaaSamples == 0)
		{
			g_downSampleMinMaxShader->setTexture2D("depthTex", g_renderTargetTextures[DRTI_Depth], 0);
		}
		else
		{
			g_downSampleMinMaxShader->setTexture2DMS("depthTex", g_renderTargetTextures[DRTI_Depth], 0);
		}

		glBegin(GL_QUADS);
			glVertex2f(-1.0f, -1.0f);
			glVertex2f(1.0f, -1.0f);
			glVertex2f(1.0f, 1.0f);
			glVertex2f(-1.0f, 1.0f);
		glEnd();
    CHECK_GL_ERROR();

		g_downSampleMinMaxShader->end();

    CHECK_GL_ERROR();
		glDepthMask(GL_TRUE);
		glPopAttrib();
	  CHECK_GL_ERROR();

		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		depthRanges.resize(gridRes.x * gridRes.y);
		glReadPixels(0, 0, gridRes.x, gridRes.y, GL_RG, GL_FLOAT, &depthRanges[0]);
    CHECK_GL_ERROR();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}