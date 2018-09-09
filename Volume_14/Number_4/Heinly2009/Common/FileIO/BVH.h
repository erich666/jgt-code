
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

#ifndef Common_FileIO_BVH_h
#define Common_FileIO_BVH_h

#include <iostream>
using std::ostream;

#include <vector>
using std::vector;

#include <Common/FileIO/BBox.h>
#include <Common/Types.h>

namespace FileIO
{

  using MathF::Point;

  class BVH
  {
  public:
    static const float isecCost;
    static const float travCost;
    static const uint  defaultThreshold;

    struct Object
    {
      BBox box;
      uint objID;
    };

    struct Node
    {
      Node();
      ~Node();

      BBox  box;
      uchar axis;
      Node* left;
      Node* right;
      vector<uint> objIDs;
    };

    BVH(const vector<Object*>& objectsIn, uint threshold_ = defaultThreshold);
    ~BVH();

    void build(vector<Object*> objectsIn);

    Node* getRoot()      const;
    uint  getNumNodes()  const;
    uint  getThreshold() const;
    uint  getNumLeaves() const;
    uint  getLeafMin()   const;
    uint  getLeafMax()   const;
    uint  getMaxDepth()  const;
    float getTotalCost() const;

    void printStats(bool = false) const;

  private:
    struct CostEval
    {
      float pos;
      float cost;
      int   numLeft;
      int   numRight;
      int   event;
      int   axis;
    };

    struct Event
    {
      float   pos;
      Object* object;
      float   leftArea;
      float   rightArea;
      int     numLeft;
      int     numRight;
      float   cost;
    };

    struct CompareEvents
    {
      bool operator()(const Event&, const Event&);
    };

    void build(Node* node, vector<Object*>& objectsIn, uint depth);
    uint partitionSAH(int& bestAxis, vector<Object*>& objectsIn);
    bool buildEvents(CostEval& newCost, const vector<Object*>& objectsIn,
                     uint axis);
    void updateBounds(Node* node);
    void computeCost(float& totalCost, Node* node, float globalArea) const;

    void printNode(const Node* node) const;

    /////////////////////////////////////////////////////////////////////////////
    // Data members

    Node* root;
    uint  threshold;
    uint  numNodes;
    uint  numLeaves;
    uint  leafMin;
    uint  leafMax;
    uint  maxDepth;
    float totalCost;

    const vector<Object*>& objects;
  };

  inline BVH::Node* BVH::getRoot() const
  {
    return root;
  }

  inline uint BVH::getNumNodes() const
  {
    return numNodes;
  }

  inline uint BVH::getThreshold() const
  {
    return threshold;
  }

  inline uint BVH::getNumLeaves() const
  {
    return numLeaves;
  }

  inline uint BVH::getLeafMin() const
  {
    return leafMin;
  }

  inline uint BVH::getLeafMax() const
  {
    return leafMax;
  }

  inline uint BVH::getMaxDepth() const
  {
    return maxDepth;
  }

  inline float BVH::getTotalCost() const
  {
    return totalCost;
  }

  inline bool BVH::CompareEvents::operator()(const Event& l, const Event& r)
  {
    return (l.pos < r.pos);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  ostream& operator<<(ostream& out, const BVH::Node& n);

} // namespace FileIO

#endif // Common_FileIO_BVH_h
