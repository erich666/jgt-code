
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

#ifndef tangere_BVH_t
#define tangere_BVH_t

#include <iomanip>
using std::setw;

#include <limits>

#include <stack>
using std::stack;

#include <utility>
using std::pair;
using std::make_pair;

#include <vector>
using std::vector;

#include <Common/FileIO/BVH.h>
#include <Common/Utility/OutputCC.h>
#include <Common/Types.h>

#include <tangere/Context.t>
#include <tangere/HitRecord.t>
#include <tangere/Node.t>
#include <tangere/Ray.t>
#include <tangere/SceneScaler.h>
#include <tangere/Triangle.t>

namespace tangere
{

  ///////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  struct Material;

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class BVH
  {
  public:
    BVH(const FileIO::BVH&       bvhIn,
        const vector<Triangle<T> >& tris,
        const vector<Material<T> >& mats,
        const SceneScaler&          scaler) :
      numNodes(bvhIn.getNumNodes()),
      numTriangles(tris.size()),
      numMaterials(mats.size()),
      threshold(bvhIn.getThreshold()),
      numLeaves(bvhIn.getNumLeaves()),
      leafMin(bvhIn.getLeafMin()),
      leafMax(bvhIn.getLeafMax()),
      maxDepth(bvhIn.getMaxDepth()),
      totalCost(bvhIn.getTotalCost())
    {
      nodes     = new Node<T>[numNodes];
      triangles = new Triangle<T>[numTriangles];
      materials = new Material<T>[numMaterials];

      for(uint i = 0; i < numMaterials; ++i)
        materials[i] = mats[i];

      uint nodeCount = 0;
      uint triCount  = 0;

      stack<pair<uint, const FileIO::BVH::Node*> > stack;

      FileIO::BVH::Node* root = bvhIn.getRoot();

      stack.push(make_pair(0, root));
      nodes[0] = convert(root, scaler);
      ++nodeCount;

      while (!stack.empty())
      {
        const pair<uint, const FileIO::BVH::Node*> topPair = stack.top();
        Node<T>& node = nodes[topPair.first];
        const FileIO::BVH::Node* current = topPair.second;
        stack.pop();

        if (node.numTris == 0)
        {
          node.index         = nodeCount;
          nodes[nodeCount]   = convert(current->left,  scaler);
          nodes[nodeCount+1] = convert(current->right, scaler);
          stack.push(make_pair(nodeCount+1, current->right));
          stack.push(make_pair(nodeCount,   current->left));
          nodeCount += 2;
        }
        else
        {
          node.index = triCount;
          for (uint i = 0; i < node.numTris; ++i)
            triangles[triCount + i] = tris[current->objIDs[i]];
          triCount += node.numTris;
        }
      }
    }

    inline BVH() 
    {
      // no-op
    }

    inline ~BVH()
    {
      delete [] nodes;
      delete [] triangles;
      delete [] materials;
    }

    inline void intersect(      HitRecord<T>&     hit,
                          const RenderContext<T>& rc,
                          const Ray<T>&           ray) const
    {
#if defined(USE_BVH)
      intersect(0, hit, rc, ray);
#else
      for (uint i = 0; i < numTriangles; ++i)
        triangles[i].intersect(hit, rc, ray);
#endif // defined(USE_BVH)
    }

    void printStats(bool all) const
    {
      Output("BVH stats:" << endl);
      Output("  scene bounds     = " << nodes[0].box << endl);
      Output(endl);
      Output("  leaf threshold   = " << threshold << endl);
      Output("  leaf range       = [" << leafMin << ", " << leafMax << ']'
             << endl);
      Output("  maximum depth    = " << maxDepth << endl);
      Output("  sah cost         = " << totalCost << endl);
      Output(endl);
      Output("  # interior nodes = " << numNodes - numLeaves << endl);
      Output("  # leaf nodes     = " << numLeaves << endl);
      Output("  total # nodes    = " << numNodes << endl);
      Output(endl);
      Output("  # triangles      = " << numTriangles << endl);
      Output(endl);

      if (all)
      {
        Output("  root node = " << nodes[0].box << endl);
        Output("    lchild  = " << nodes[0].index << endl);
        Output("    rchild  = " << nodes[0].index+1 << endl);
        Output("    ntris   = " << nodes[0].numTris << endl);

        for (uint i = 1; i < numNodes; ++i)
        {
          const Node<T>& node = nodes[i];
          if (node.numTris == 0)
          {
            Output("  interior node[" << setw(2) << i <<"] = "
                   << node.box << endl);
            Output("    lchild = " << node.index << endl);
            Output("    rchild = " << node.index+1 << endl);
            Output("    ntris  = " << node.numTris << endl);
          }
          else
          {
            Output("  leaf node[" << setw(2) << i <<"] = "
                   << node.box << endl);
            Output("    ntris = " << node.numTris << endl);
            Output("    first = " << node.index << endl);
            Output("    last  = " << node.index+node.numTris << endl);
          }
        }

        Output(endl);
      }
    }
  
    const Node<T>*     getNodes()     const { return nodes;     }
          Triangle<T>* getTriangles() const { return triangles; }
    const Material<T>* getMaterials() const { return materials; }

    void setNodes(Node<T>* n)         { nodes     = n; }
    void setTriangles(Triangle<T>* t) { triangles = t; }
    void setMaterials(Material<T>* m) { materials = m; }

    uint  getNumNodes()     const { return numNodes;     }
    uint  getNumTriangles() const { return numTriangles; }
    uint  getNumMaterials() const { return numMaterials; }
    uint  getThreshold()    const { return threshold;    }
    uint  getNumLeaves()    const { return numLeaves;    }
    uint  getLeafMin()      const { return leafMin;      }
    uint  getLeafMax()      const { return leafMax;      }
    uint  getMaxDepth()     const { return maxDepth;     }
    float getTotalCost()    const { return totalCost;    }

    void setThreshold(uint t)      { threshold    = t;    }
    void setNumLeaves(uint l)      { numLeaves    = l;    }
    void setLeafMin(uint lmin)     { leafMin      = lmin; }
    void setLeafMax(uint lmax)     { leafMax      = lmax; }
    void setMaxDepth(uint md)      { maxDepth     = md;   }
    void setTotalCost(float tc)    { totalCost    = tc;   }
    void setNumNodes(uint num)     { numNodes     = num;  }
    void setNumTriangles(uint num) { numTriangles = num;  }
    void setNumMaterials(uint num) { numMaterials = num;  }

  private:
    Node<T> convert(const FileIO::BVH::Node* node,
                    const SceneScaler&       scaler) const
    {
      Node<T> n;

      n.box.bounds[0] = node->box.getMin();
      n.box.bounds[1] = node->box.getMax();
      n.index         = -1;
      n.numTris       = node->objIDs.size();
      n.axis          = node->axis;

      return n;
    }

    inline void intersect(      uint              nodeID,
                                HitRecord<T>&     hit,
                          const RenderContext<T>& rc,
                          const Ray<T>&           ray) const
    {
      const Node<T>& node = nodes[nodeID];

      ///////////////////////////////////////////////////////////////////////////
      // Intersect leaf node

      if (node.numTris > 0)
      {
        for (uint i = 0; i < node.numTris; ++i) 
          triangles[node.index + i].intersect(hit, rc, ray);

        return;
      }

      ///////////////////////////////////////////////////////////////////////////
      // Intersect interior node

      T tLeft  = numeric_limits<T>::max();
      T tRight = numeric_limits<T>::max();

      // Fetch children
      const uint lidx = node.index;
      const uint ridx = lidx+1;

      // Test left
      if (nodes[lidx].box.intersect(tLeft, ray) &&
          tLeft < hit.getMinT())
      {
        // Hit left, test right
        if (nodes[ridx].box.intersect(tRight, ray) &&
            tRight < hit.getMinT())
        {
          // Hit both, traverse left first
          if (tLeft < tRight)
          {
            intersect(lidx, hit, rc, ray);
        
            // Traverse right (if necessary)
            if (tRight < hit.getMinT())
              intersect(ridx, hit, rc, ray);
          }
          else
          {
            // Hit both, traverse right first
            intersect(ridx, hit, rc, ray);
        
            // Traverse left (if necessary)
            if (tLeft < hit.getMinT())
              intersect(lidx, hit, rc, ray);
          }
        }
        else
        {
          // Missed right, traverse left
          intersect(lidx, hit, rc, ray);
        }
      }
      else
      {
        // Missed left, test right
        if (nodes[ridx].box.intersect(tRight, ray) &&
            tRight < hit.getMinT())
          intersect(ridx, hit, rc, ray);
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    // Data members

    Node<T>*     nodes;
    Triangle<T>* triangles;
    Material<T>* materials;

    uint numNodes;
    uint numTriangles;
    uint numMaterials;

    uint  threshold;
    uint  numLeaves;
    uint  leafMin;
    uint  leafMax;
    uint  maxDepth;
    float totalCost;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Template specialization - BVH<int>

  // XXX(cpg) - why does this template specilaization have to be inline'd
  //            with g++ 4.2.1 on Mac (and maybe elsewhere)?
  template<>
  inline Node<int> BVH<int>::convert(const FileIO::BVH::Node* node,
                                     const SceneScaler&       scaler) const
  {
    Node<int> n;

    n.box.bounds[0] = scaler.scale(node->box.getMin());
    n.box.bounds[1] = scaler.scale(node->box.getMax());
    n.index         = -1;
    n.numTris       = node->objIDs.size();
    n.axis          = node->axis;

    return n;
  }

} // namespace tangere

#endif // tangere_BVH_t
