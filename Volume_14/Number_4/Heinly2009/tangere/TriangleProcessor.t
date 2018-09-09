
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

#ifndef tangere_TriangleProcessor_t
#define tangere_TriangleProcessor_t

#include <vector>
using std::vector;

#include <Common/FileIO/BVH.h>
#include <Common/FileIO/GeometryProcessor.h>
#include <Common/FileIO/Mesh.h>

#include <Common/Utility/MinMax.t>

#include <Math/Vector.t>

#include <tangere/Constants.h>
#include <tangere/Material.t>
#include <tangere/Shader.t>
#include <tangere/ShaderTable.h>
#include <tangere/Triangle.t>

namespace tangere
{

  ///////////////////////////////////////////////////////////////////////////////
  // Type definitions

  typedef vector<FileIO::BVH::Object*>    bvhObjVector;
  typedef vector<FileIO::Mesh::Triangle*> meshTriangleVector;
  typedef vector<FileIO::Mesh::Vertex>    meshVertexVector;
  typedef vector<FileIO::Mesh::Material>  meshMaterialVector;

  ///////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using FileIO::GeometryProcessor;

  using Utility::Max;

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class TriangleProcessor : public GeometryProcessor
  {
  public:
    TriangleProcessor(vector<Triangle<T> >& waldTriangles_,
                      vector<Material<T> >& materials_,
                      SceneScaler&          scaler_) :
      waldTriangles(waldTriangles_),
      materials(materials_),
      scaler(scaler_),
      discardedSpheres(0),
      discardedEdges(0),
      discardedAreas(0),
      initialNumTris(0)
    {
      // no-op
    }

    inline ~TriangleProcessor()
    {
      for (uint i = 0; i < objects.size(); ++i)
        delete objects[i];
    }

    const bvhObjVector& operator()(const meshTriangleVector&,
                                   const meshVertexVector&,
                                   const meshMaterialVector&)
    {
      FatalError("TriangleProcess<T>::operator() should not be called!" << endl);
      
      // To quiet warnings
      bvhObjVector* junk = new bvhObjVector();
      return *junk;
    }

    inline void printStats() const
    {
      Output("Processor stats:" << endl);
      Output("  Initial triangle count     = " << initialNumTris << endl);
      Output("  Discarded spheres          = " << discardedSpheres << endl);
      Output(endl);
    }

  private:
    bvhObjVector          objects;
    vector<Triangle<T> >& waldTriangles;
    vector<Material<T> >& materials;
    SceneScaler&          scaler;
    uint                  initialNumTris;
    uint                  discardedSpheres;
    uint                  discardedEdges;
    uint                  discardedAreas;
  };

  ///////////////////////////////////////////////////////////////////////////////
  // Template specialization - TriangleProcessor<float>

  // XXX(cpg) - why does this template specilaization have to be inline'd
  //            with g++ 4.2.1 on Mac (and maybe elsewhere)?
  template<>
  inline const bvhObjVector& TriangleProcessor<float>::operator()
    (
      const meshTriangleVector& trianglesIn,
      const meshVertexVector&   verticesIn,
      const meshMaterialVector& materialsIn
    )
  {
    using MathF::Point;
    using MathF::Vector;

    initialNumTris = trianglesIn.size();

    // Filter out the spheres
    meshTriangleVector filtered;
    for (uint i = 0; i < trianglesIn.size(); ++i)
    {
      if (trianglesIn[i]->vID[0] != FileIO::Mesh::undefined)
        filtered.push_back(trianglesIn[i]);
      else
        ++discardedSpheres;
    }

    // Compute bounding boxes for all of the triangles
    for (uint i = 0; i < filtered.size(); ++i)
    {
      const uint* vID = filtered[i]->vID;

      FileIO::BBox box;
      box.extend(verticesIn[vID[0]].p);
      box.extend(verticesIn[vID[1]].p);
      box.extend(verticesIn[vID[2]].p);

      FileIO::BVH::Object* obj = new FileIO::BVH::Object();
      obj->box   = box;
      obj->objID = i;

      objects.push_back(obj);
    }

    // Convert materials
    for (uint i = 0; i < materialsIn.size(); ++i)
    {
      const bool isSpecular    = (materialsIn[i].specular == rtCore::RGB<float>::One);
      const bool isTransparent = (materialsIn[i].tr > 0);
      materials.push_back(Material<float>(materialsIn[i].diffuse,
                                          materialsIn[i].specular,
                                          materialsIn[i].emissive,
                                          Constants<float>::ka,
                                          Constants<float>::kd,
                                          materialsIn[i].exp,
                                          materialsIn[i].r0,
                                          isSpecular,
                                          isTransparent));
    }

    // Create Wald triangles
    waldTriangles.reserve(filtered.size());
    for (uint i = 0; i < filtered.size(); ++i)
    {
      const uint*   vID = filtered[i]->vID;
      const Point&  v0  = verticesIn[vID[0]].p;
      const Point&  v1  = verticesIn[vID[1]].p;
      const Point&  v2  = verticesIn[vID[2]].p;
      const Vector& n0  = verticesIn[vID[0]].n;
      const Vector& n1  = verticesIn[vID[1]].n;
      const Vector& n2  = verticesIn[vID[2]].n;
      const uint    mID = filtered[i]->mID;

#ifdef USE_NDOTV_SHADER
      waldTriangles.push_back(Triangle<float>(v0, v1, v2, n0, n1, n2, mID,
                              stable.getShader(Shader<float>::NdotV)));
#else
      if (materials[mID].reflective)
        waldTriangles.push_back(Triangle<float>(v0, v1, v2, n0, n1, n2, mID,
                                stable.getShader(Shader<float>::Reflection)));
      else if (materials[mID].transparent)
        waldTriangles.push_back(Triangle<float>(v0, v1, v2, n0, n1, n2, mID,
                                stable.getShader(Shader<float>::Dielectric)));
      else
        waldTriangles.push_back(Triangle<float>(v0, v1, v2, n0, n1, n2, mID,
                                stable.getShader(Shader<float>::Lambertian)));
#endif // USE_NDOTV_SHADER
    }

    return objects;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Template specialization - TriangleProcessor<int>

  // XXX(cpg) - why does this template specilaization have to be inline'd
  //            with g++ 4.2.1 on Mac (and maybe elsewhere)?
  template<>
  inline const bvhObjVector& TriangleProcessor<int>::operator()
    (
      const meshTriangleVector& trianglesIn,
      const meshVertexVector&   verticesIn,
      const meshMaterialVector& materialsIn
    )
  {
    initialNumTris = trianglesIn.size();

    // Add vertices to the scene scaler and compute scale
    for (uint i = 0; i < verticesIn.size(); ++i)
      scaler.extend(verticesIn[i].p);
    scaler.computeScale();

    // Filter out the spheres
    meshTriangleVector filtered;
    filtered.reserve(trianglesIn.size());
    for (uint i = 0; i < trianglesIn.size(); ++i)
    {
      if (trianglesIn[i]->vID[0] == FileIO::Mesh::undefined)
      {
        ++discardedSpheres;
        continue;
      }

      const uint*         vID = trianglesIn[i]->vID;
      const MathI::Point& v0  = scaler.scale(verticesIn[vID[0]].p);
      const MathI::Point& v1  = scaler.scale(verticesIn[vID[1]].p);
      const MathI::Point& v2  = scaler.scale(verticesIn[vID[2]].p);
      const MathI::Vector e0  = v0 - v1;
      const MathI::Vector e1  = v1 - v2;
      const MathI::Vector e2  = v2 - v0;

      // discard triangles with edges that map to zero
      if (e0 == Math::Vector<int>::Zero ||
          e1 == Math::Vector<int>::Zero || 
          e2 == Math::Vector<int>::Zero)
      {
        ++discardedEdges;
        continue;
      }

      // discard triangles with areas of zero
      if (Cross(e0, e1) == Math::Vector<int>::Zero)
      {
        ++discardedAreas;
        continue;
      }

      filtered.push_back(trianglesIn[i]);
    }

    // Compute bounding boxes for all of the triangles
    for (uint i = 0; i < filtered.size(); ++i)
    {
      const uint*        vID  = filtered[i]->vID;
      const MathF::Point v[3] = 
        {
          verticesIn[vID[0]].p,
          verticesIn[vID[1]].p,
          verticesIn[vID[2]].p
        };

      const MathF::Vector e[3] = 
        {
          v[0] - v[1],
          v[1] - v[2],
          v[2] - v[0]
        };

      FileIO::BBox box;
      box.extend(v[0]);
      box.extend(v[1]);
      box.extend(v[2]);

      const float tmpMax = Max(e[0].length(), e[1].length(), e[2].length());
      const float tmpMin = Min(e[0].length(), e[1].length(), e[2].length());

      Triangle<int>::setMaxEdge(Max(tmpMax, Triangle<int>::getMaxEdge()));
      Triangle<int>::setMinEdge(Min(tmpMin, Triangle<int>::getMinEdge()));

      FileIO::BVH::Object* obj = new FileIO::BVH::Object;
      obj->box   = box;
      obj->objID = i;

      objects.push_back(obj);
    }

    // Convert materials
    for (uint i = 0; i < materialsIn.size(); ++i)
    {
      const bool isSpecular    = (materialsIn[i].specular == rtCore::RGB<float>::One);
      const bool isTransparent = (materialsIn[i].tr > 0);
      materials.push_back(Material<int>(materialsIn[i].diffuse,
                                        materialsIn[i].specular,
                                        materialsIn[i].emissive,
                                        Constants<int>::ka,
                                        Constants<int>::kd,
                                        int(materialsIn[i].exp + 0.5f),
                                        FLOAT_TO_FIXED31(materialsIn[i].r0),
                                        isSpecular,
                                        isTransparent));
    }

    // Create scaled vertex buffer
    vector<MathI::Point>  intVertexBuffer;
    vector<MathI::Vector> intNormalBuffer;
    for (uint i = 0; i < verticesIn.size(); ++i)
    {
      intVertexBuffer.push_back(scaler.scale(verticesIn[i].p));
      intNormalBuffer.push_back(Vector<int>(verticesIn[i].n));
    }

    // Create Wald triangles
    for (uint i = 0; i < filtered.size(); ++i)
    {
      const uint*        vID = filtered[i]->vID;
      const Point<int>&  v0  = intVertexBuffer[vID[0]];
      const Point<int>&  v1  = intVertexBuffer[vID[1]];
      const Point<int>&  v2  = intVertexBuffer[vID[2]];
      const Vector<int>& n0  = intNormalBuffer[vID[0]];
      const Vector<int>& n1  = intNormalBuffer[vID[1]];
      const Vector<int>& n2  = intNormalBuffer[vID[2]];
      const uint         mID = filtered[i]->mID;

#ifdef USE_NDOTV_SHADER
      waldTriangles.push_back(Triangle<int>(v0, v1, v2, n0, n1, n2, mID,
                              stable.getShader(Shader<int>::NdotV)));
#else
      if (materials[mID].reflective)
        waldTriangles.push_back(Triangle<int>(v0, v1, v2, n0, n1, n2, mID,
                                stable.getShader(Shader<int>::Reflection)));
      else if (materials[mID].transparent)
        waldTriangles.push_back(Triangle<int>(v0, v1, v2, n0, n1, n2, mID,
                                stable.getShader(Shader<int>::Dielectric)));
      else
        waldTriangles.push_back(Triangle<int>(v0, v1, v2, n0, n1, n2, mID,
                                stable.getShader(Shader<int>::Lambertian)));
#endif // USE_NDOTV_SHADER
    }

    // Bias triangle edges
    Triangle<int>::computeEdgeBias();
    for (uint i = 0; i < waldTriangles.size(); ++i)
      waldTriangles.at(i).applyEdgeBias();

    return objects;
  }

  // XXX(cpg) - why does this template specilaization have to be inline'd
  //            with g++ 4.2.1 on Mac (and maybe elsewhere)?
  template<>
  inline void TriangleProcessor<int>::printStats() const
  {
    Output("Triangle stats:" << endl);
    Output("  Minimum edge =  " << Triangle<int>::getMinEdge() << endl);
    Output("  Maximum edge =  " << Triangle<int>::getMaxEdge() << endl);
    Output("  Edge bias    =  " << Triangle<int>::getEdgeBias() << endl);
    Output(endl);
    Output("Processor stats:" << endl);
    Output("  Initial triangle count     = " << initialNumTris << endl);
    Output("  Discarded spheres          = " << discardedSpheres << endl);
    Output("  Discarded triangles (edge) = " << discardedEdges << endl);
    Output("  Discarded triangles (area) = " << discardedAreas << endl);
    Output(endl);
  }

}  // namespace tangere

#endif // tangere_TriangleProcessor_t
