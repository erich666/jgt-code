
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

#ifndef tangere_Scene_t
#define tangere_Scene_t

#include <cstring>

#include <vector>
using std::vector;

#include <Common/FileIO/Image.t>
#include <Common/FileIO/Mesh.h>
#include <Common/FileIO/Scene.h>
#include <Common/rtCore/RGB.t>
#include <Common/Utility/Timer.h>

#include <Math/Math.h>

#include <tangere/BVH.t>
#include <tangere/Camera.t>
#include <tangere/Flags.h>
#include <tangere/Light.t>
#include <tangere/Material.t>
#include <tangere/Node.t>
#include <tangere/Options.h>
#include <tangere/Triangle.t>
#include <tangere/TriangleProcessor.t>

namespace tangere
{

  /////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using FileIO::Image;
  using FileIO::Mesh;

  using Utility::Timer;

  /////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  class BVH;

  class SceneScaler;

  /////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class Scene
  {
  public:
    Scene(const FileIO::Scene* ioScene) :
      ambient(RGB<T>(Constants<T>::ka, Constants<T>::ka, Constants<T>::ka)),
      sky(RGB<T>(ioScene->getSky())),
      ground(RGB<T>(ioScene->getGround()))
    {
      T max = Max(sky[0], sky[1], sky[2]);
      if (max > rtCore::RGB<T>::One[0])
        sky /= max;

      max = Max(ground[0], ground[1], ground[2]);
      if (max > rtCore::RGB<T>::One[0])
        ground /= max;

      /////////////////////////////////////////////////////////////////////////
      // Process geometry

      Output("Processing geometry...");

      vector<Triangle<T> > triangles;
      vector<Material<T> > materials;
      SceneScaler          scaler;
      const FileIO::Scene::ViewParameters& viewParams = 
                                           ioScene->getViewParameters();
      scaler.extend(viewParams.eye);

      // Compute default light position
      const Mesh* mesh = ioScene->getMesh();
      
      vector<FileIO::Mesh::Triangle*> tris  = mesh->getTriangles();
      vector<FileIO::Mesh::Material>  mats  = mesh->getMaterials();
      vector<FileIO::Mesh::Vertex>    verts = mesh->getVertices();

      vector<Point<float> > lightPoints;
#if defined(USE_GEOMETRY_LIGHTS)
      for(uint i = 0; i < tris.size(); ++i)
      {
        if (mats[tris[i]->mID].emissive != rtCore::RGB<float>::Zero)
        {
          const Point<float> v0 = verts[tris[i]->vID[0]].p;
          const Point<float> v1 = verts[tris[i]->vID[1]].p;
          const Point<float> v2 = verts[tris[i]->vID[2]].p;
          const Point<float> c  = (v0 + v1 + v2) / 3;
          
          const Vector<float> normal = Cross(v1 - v0, v2 - v0);
          const Point<float>  point  = c + normal*::Constants<float>::Epsilon;
          lightPoints.push_back(point);
          scaler.extend(point);
        }
      }
#else
      const Vector<float> diagonal = mesh->getBounds().diagonal();
#if defined(OFFSET_EYE_LIGHT)
      const float         scale    = 0.01f*diagonal.length();
#else
      const float         scale    = 0;
#endif
      const Vector<float> in       = (viewParams.lookat - viewParams.eye).normal();
      const Vector<float> up       = viewParams.up.normal();
      const Vector<float> right    = Cross(in, up);
      const Point<float>  lightPos = (viewParams.eye +
                                      scale*right +
                                      scale*up +
                                      scale*in);
      scaler.extend(lightPos);
#endif // defined(USE_GEOMETRY_LIGHTS)

      TriangleProcessor<T> tproc(triangles, materials, scaler);

      vector<FileIO::BVH::Object*> objects = tproc(mesh->getTriangles(),
                                                   mesh->getVertices(),
                                                   mesh->getMaterials());

      Output("  done" << endl);
      Output(endl);

      scaler.printStats();
      tproc.printStats();

      /////////////////////////////////////////////////////////////////////////
      // Build BVH

      Output("Building BVH...");

      FileIO::BVH ioBVH(objects, ioScene->getThreshold());
      bvh = new BVH<T>(ioBVH, triangles, materials, scaler);

      triangles.resize(0);
      materials.resize(0);

      Output("  done" << endl);
      Output(endl);
      bvh->printStats(false);

      /////////////////////////////////////////////////////////////////////////
      // Create scene

#if defined(USE_GEOMETRY_LIGHTS)
      for (uint i = 0; i < lightPoints.size(); ++i)
        addLight(new Light<T>(lightPoints[i], scaler, rtCore::RGB<T>::One));
#else
      addLight(new Light<T>(lightPos, scaler, rtCore::RGB<T>::One));
#endif // defined(USE_GEOMETRY_LIGHTS)

      const FileIO::Scene::ImageParameters& imageParams =
        ioScene->getImageParameters();
      image = new Image<RGB<T> >(imageParams.xRes, imageParams.yRes);

      camera = new Camera<T>(viewParams.eye,
                             viewParams.lookat,
                             viewParams.up,
                             viewParams.fov,
                             image->aspect(),
                             scaler);

      // Output configuration
      Output("Renderer configuration:" << endl);
      Output("  niters = 1" << endl);
      Output("  xres   = " << imageParams.xRes << endl);
      Output("  yres   = " << imageParams.yRes << endl);
      Output(endl);
      Output("  Camera:" << endl);
      Output("    eye    = " << viewParams.eye << endl);
      Output("    lookat = " << viewParams.lookat << endl);
      Output("    up     = " << viewParams.up << endl);
      Output("    hfov   = " << viewParams.fov << endl);
      Output(endl);
      Output("  Lights:" << endl);
      for (uint i = 0; i < lights.size(); ++i)
      {
        Output("    lights[" << i << "] = "
               << lights[i]->getPosition() << ", "
               << lights[i]->getColor() << endl);
      }
      Output(endl);
      Output("  sky    = " << ioScene->getSky() << endl);
      Output("  ground = " << ioScene->getGround() << endl);
    }

    inline Scene()
    {
      // no-op
    }

    inline ~Scene()
    {
      delete bvh;
      delete camera;
      delete image;

      for (uint i = 0; i < lights.size(); ++i)
        delete lights[i];
    }

    inline BVH<T>*         getBVH()        const { return bvh;     }
    inline Camera<T>*      getCamera()     const { return camera;  }
    inline Image<RGB<T> >* getImage()      const { return image;   }
    inline RGB<T>          getAmbient()    const { return ambient; }

    inline RGB<T> getBackground(const Vector<T>& dir) const
    {
      return (Dot(dir, camera->getUp()) > 0 ? sky : ground);
    }

    inline void setBVH(BVH<T>* bvh_)               { bvh     = bvh_;    }
    inline void setCamera(Camera<T>* camera_)      { camera  = camera_; }
    inline void setImage(Image<RGB<T> >* image_)   { image   = image_;  }
    inline void setAmbient(const RGB<T>& ambient_) { ambient = ambient; }
    inline void setSky(const RGB<T>& sky_)         { sky     = sky_;    }
    inline void setGround(const RGB<T>& ground_)   { ground  = ground_; }

    inline const vector<const Light<T>*>& getLights() const { return lights; }
    inline       void addLight(const Light<T>* l) { lights.push_back(l); }

    void write(const Options& /* unused */)
    {
      // no-op
    }

    void read(const Options& /* unused */)
    {
      // no-op
    }

  private:
    BVH<T>*         bvh;
    Camera<T>*      camera;
    Image<RGB<T> >* image;
    RGB<T>          ambient;
    RGB<T>          sky;
    RGB<T>          ground;

    vector<const Light<T>*> lights;
  };

  /////////////////////////////////////////////////////////////////////////////
  // File format
  
  /*
    "int"
    xres
    yres
    camera
    ambient color
    sky color
    ground color
    triangle edge bias
    triangle max edge component
    triangle edge scale
    triangle max edge
    triangle min edge
    bvh threshold
    bvh number of leaves
    bvh leaf min
    bvh leaf max
    bvh max depth
    bvh total cost
    bvh number of nodes
    bvh node data
    bvh number of triangles
    bvh triangle data
    bvh number of materials
    bvh material data
    number of lights
    light data
  */

  template<>
  inline void Scene<int>::write(const Options& opt)
  {
    string fileName = opt.iWrite;
    FILE*  fout     = fopen(fileName.c_str(), "wb");
    if (!fout)
      FatalError("Failed to open \"" << fileName << "\" for writing" << endl);

    Output(endl);
    Output("Writing \"" << fileName << "\"...");

    int   intTemp;
    uint  uintTemp;
    float floatTemp;

    const char* text = "int";
    fwrite((void*)text, sizeof(char), 4, fout);

    uintTemp =     image->w();
    fwrite((void*)&uintTemp, sizeof(uint), 1, fout);
    uintTemp =     image->h();
    fwrite((void*)&uintTemp, sizeof(uint), 1, fout);

    fwrite((void*)camera, sizeof(Camera<int>), 1, fout);

    fwrite((void*)(&ambient), sizeof(RGB<int>), 1, fout);
    fwrite((void*)(&sky),     sizeof(RGB<int>), 1, fout);
    fwrite((void*)(&ground),  sizeof(RGB<int>), 1, fout);

    intTemp   =    Triangle<int>::getEdgeBias();
    fwrite((void*)&intTemp,   sizeof(int),   1, fout);
    floatTemp =    Triangle<int>::getMaxEdgeComponent();
    fwrite((void*)&floatTemp, sizeof(float), 1, fout);
    floatTemp =    Triangle<int>::getEdgeScale();
    fwrite((void*)&floatTemp, sizeof(float), 1, fout);
    floatTemp =    Triangle<int>::getMaxEdge();
    fwrite((void*)&floatTemp, sizeof(float), 1, fout);
    floatTemp =    Triangle<int>::getMinEdge();
    fwrite((void*)&floatTemp, sizeof(float), 1, fout);

    uintTemp  =    bvh->getThreshold();
    fwrite((void*)&uintTemp,  sizeof(uint), 1, fout);
    uintTemp  =    bvh->getNumLeaves();
    fwrite((void*)&uintTemp,  sizeof(uint),1 , fout);
    uintTemp  =    bvh->getLeafMin();
    fwrite((void*)&uintTemp,  sizeof(uint), 1, fout);
    uintTemp  =    bvh->getLeafMax();
    fwrite((void*)&uintTemp,  sizeof(uint), 1, fout);
    uintTemp  =    bvh->getMaxDepth();
    fwrite((void*)&uintTemp,  sizeof(uint), 1, fout);
    floatTemp =    bvh->getTotalCost();
    fwrite((void*)&floatTemp, sizeof(float), 1, fout);

    uintTemp =     bvh->getNumNodes();
    fwrite((void*)&uintTemp, sizeof(uint), 1, fout);
    fwrite((void*) bvh->getNodes(), sizeof(Node<int>), uintTemp, fout);

    uintTemp =     bvh->getNumTriangles();
    fwrite((void*)&uintTemp, sizeof(uint), 1, fout);
    fwrite((void*) bvh->getTriangles(),
                   sizeof(Triangle<int>), uintTemp, fout);

    uintTemp =     bvh->getNumMaterials();
    fwrite((void*)&uintTemp, sizeof(uint), 1, fout);
    fwrite((void*) bvh->getMaterials(),
                   sizeof(Material<int>), uintTemp, fout);

    uintTemp = lights.size();
    fwrite((void*)&uintTemp, sizeof(uint), 1, fout);
    for (uint i = 0; i < uintTemp; ++i)
    {
      fwrite((void*)lights[i], sizeof(Light<int>), 1, fout);
    }

    fclose(fout);

    Output("  done" << endl);
    Output(endl);
  }

  template<>
  inline void Scene<int>::read(const Options& opt)
  {
    string fileName = opt.bname + ".int";
    FILE* fin = fopen(fileName.c_str(), "rb");
    if (fin == NULL)
    {
      FatalError("Failed to open \"" << fileName << "\" for reading" << endl);
    }

    int    intTemp;
    uint   uintTemp;
    float  floatTemp;
    size_t rc;

    char* cText = new char[4];
    rc = fread((void*)cText, sizeof(char), 4, fin);
    if (strcmp(cText, "int") != 0)
    {
      Output(cText << endl);
      FatalError("Unrecognized file format:  expecting \".int\" file" << endl);
    }

    uint xRes;
    uint yRes;
    rc = fread((void*)&xRes, sizeof(uint), 1, fin);
    rc = fread((void*)&yRes, sizeof(uint), 1, fin);
    image = new Image<RGB<int> >(xRes, yRes);

    camera = new Camera<int>();
    rc = fread((void*)camera, sizeof(Camera<int>), 1, fin);

    RGB<int> color;
    rc = fread((void*)&color, sizeof(RGB<int>), 1, fin);
    ambient = color;
    rc = fread((void*)&color, sizeof(RGB<int>), 1, fin);
    sky = color;
    rc = fread((void*)&color, sizeof(RGB<int>), 1, fin);
    ground = color;

    rc = fread((void*)&intTemp,   sizeof(int), 1, fin);
    Triangle<int>::setEdgeBias(intTemp);
    rc = fread((void*)&floatTemp, sizeof(float), 1, fin);
    Triangle<int>::setMaxEdgeComponent(floatTemp);
    rc = fread((void*)&floatTemp, sizeof(float), 1, fin);
    Triangle<int>::setEdgeScale(floatTemp);
    rc = fread((void*)&floatTemp, sizeof(float), 1, fin);
    Triangle<int>::setMaxEdge(floatTemp);
    rc = fread((void*)&floatTemp, sizeof(float), 1, fin);
    Triangle<int>::setMinEdge(floatTemp);

    bvh = new BVH<int>();

    rc = fread((void*)&uintTemp,  sizeof(uint), 1, fin);
    bvh->setThreshold(uintTemp);
    rc = fread((void*)&uintTemp,  sizeof(uint), 1, fin);
    bvh->setNumLeaves(uintTemp);
    rc = fread((void*)&uintTemp,  sizeof(uint), 1, fin);
    bvh->setLeafMin(uintTemp);
    rc = fread((void*)&uintTemp,  sizeof(uint), 1, fin);
    bvh->setLeafMax(uintTemp);
    rc = fread((void*)&uintTemp,  sizeof(uint), 1, fin);
    bvh->setMaxDepth(uintTemp);
    rc = fread((void*)&floatTemp, sizeof(float), 1, fin);
    bvh->setTotalCost(floatTemp);

    rc = fread((void*)&uintTemp, sizeof(uint), 1, fin);
    bvh->setNumNodes(uintTemp);
    Node<int>* nodes = new Node<int>[uintTemp];
    rc = fread((void*)nodes, uintTemp * sizeof(Node<int>), 1, fin);
    bvh->setNodes(nodes);

    rc = fread((void*)&uintTemp, sizeof(uint), 1, fin);
    bvh->setNumTriangles(uintTemp);
    Triangle<int>* t = new Triangle<int>[uintTemp];
    rc = fread((void*)t, uintTemp * sizeof(Triangle<int>), 1, fin);
    bvh->setTriangles(t);

    rc = fread((void*)&uintTemp, sizeof(uint), 1, fin);
    bvh->setNumMaterials(uintTemp);
    Material<int>* mats = new Material<int>[uintTemp];
    rc = fread((void*) mats, uintTemp * sizeof(Material<int>), 1, fin);
    bvh->setMaterials(mats);

    rc = fread((void*)&uintTemp, sizeof(uint), 1, fin);
    for (uint i = 0; i < uintTemp; ++i)
    {
      Light<int>* l = new Light<int>();
      rc = fread((void*)l, sizeof(Light<int>), 1, fin);
      addLight(l);
    }

    fclose(fin);

    Triangle<int>* triangles = bvh->getTriangles();
    for (uint i = 0; i < bvh->getNumTriangles(); ++i)
    {
      const uint& mID = triangles[i].getMID();
      const Material<int>& material = bvh->getMaterials()[mID];
#ifdef USE_NDOTV_SHADER
      triangles[i].setShader(stable.getShader(Shader<int>::NdotV));
#else
      if (material.reflective)
        triangles[i].setShader(stable.getShader(Shader<int>::Reflection));
      else if (material.transparent)
        triangles[i].setShader(stable.getShader(Shader<int>::Dielectric));
      else
        triangles[i].setShader(stable.getShader(Shader<int>::Lambertian));
#endif // USE_NDOTV_SHADER
    }
  }

} // namespace tangere

namespace tangereF
{

  typedef tangere::Scene<float> Scene;

} // namespace tangereF

namespace tangereI
{

  typedef tangere::Scene<int> Scene;

} // namespace tangereI

#endif // tangere_Scene_t
