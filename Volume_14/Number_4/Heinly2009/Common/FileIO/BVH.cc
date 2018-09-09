
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

#include <cfloat>
#include <climits>

#include <algorithm>
using std::sort;

#include <iostream>
using std::endl;
using std::ostream;

#include <Common/FileIO/BVH.h>
#include <Common/Utility/OutputCC.h>

namespace FileIO
{
  const float BVH::isecCost         = 10.f;
  const float BVH::travCost         = 10.f;
  const uint  BVH::defaultThreshold = 8;

  BVH::BVH(const vector<Object*>& objectsIn, uint threshold_) :
    root(0),
    threshold(threshold_),
    numNodes(0),
    numLeaves(0),
    leafMin(UINT_MAX),
    leafMax(0),
    maxDepth(0),
    totalCost(0),
    objects(objectsIn)
  {
    build(objects);
  }

  BVH::~BVH()
  {
    if (root)
      delete root;
  }

  BVH::Node::Node() :
    left(0), right(0)
  {
    // no-op
  }

  BVH::Node::~Node()
  {
    if (objIDs.size() == 0)
    {
      delete left;
      delete right;
    }
  }

  // NOTE(jsh) - passing by value so that as we split and build, we don't
  //             lose the original data
  void BVH::build(vector<Object*> objectsIn)
  {
    root = new Node();
    build(root, objectsIn, 0);
    updateBounds(root);
    computeCost(totalCost, root, root->box.computeSA());
  }

  void BVH::build(Node* node, vector<Object*>& objectsIn, uint depth)
  {
    const uint size = objectsIn.size();
    int   bestAxis  = -1;
    uint  split     = partitionSAH(bestAxis, objectsIn);

    if (bestAxis == -1)
    {
      // Make a leaf node
      for (uint i = 0; i < size; ++i)
      {
        node->objIDs.push_back(objectsIn[i]->objID);
        node->box.extend(objectsIn[i]->box);
      }

      objectsIn.clear();
      objectsIn.reserve(0);
      node->axis = 0;
      ++numNodes;
      ++numLeaves;

      leafMin = Min(leafMin, uint(node->objIDs.size()));
      leafMax = Max(leafMax, uint(node->objIDs.size()));

      sort(node->objIDs.begin(), node->objIDs.end());
    }
    else
    {
      // Make an interior node
      node->axis = bestAxis;
      ++numNodes;

      node->left  = new Node();
      node->right = new Node();
      vector<Object*> leftObjects;
      vector<Object*> rightObjects;

      leftObjects.reserve(split);
      for (uint i = 0; i < split; ++i)
      {
        leftObjects.push_back(objectsIn[i]);
      }

      rightObjects.reserve(size - split);
      for (uint i = split; i < size; ++i)
      {
        rightObjects.push_back(objectsIn[i]);
      }

      objectsIn.clear();
      objectsIn.reserve(0);

      // Recursively build hierarchy
      ++depth;
      build(node->left,  leftObjects,  depth);
      build(node->right, rightObjects, depth);
    }

    maxDepth = Max(maxDepth, depth);
  }

  uint BVH::partitionSAH(int& bestAxis, vector<Object*>& objectsIn)
  {
    const uint size = objectsIn.size();
    if (size <= threshold)
    {
      bestAxis = -1;
      return -1;
    }

    CostEval bestCost;
    bestCost.cost  = isecCost * size;
    bestCost.axis  = -1;
    bestCost.event = -1;
    for (uint axis = 0; axis < 3; ++axis)
    {
      CostEval newCost = {};
      if (buildEvents(newCost, objectsIn, axis))
      {
        if (newCost.cost < bestCost.cost)
        {
          bestCost = newCost;
        }
      }
    }

    bestAxis = bestCost.axis;

    if (bestAxis != -1)
    {
      vector<Event> events;
      for (uint i = 0; i < size; ++i)
      {
        Event newEvent;
        newEvent.pos    = (objectsIn[i]->box.center())[bestAxis];
        newEvent.object = objectsIn[i];

        events.push_back(newEvent);
      }

      sort(events.begin(), events.end(), CompareEvents());

      for (uint i = 0; i < size; ++i)
      {
        objectsIn[i] = events[i].object;
      }

      uint result = bestCost.event;
      if (result == 0 || result == size)
      {
        return size / 2;
      }

      return result;
    }

    return -1;
  }

  bool BVH::buildEvents(CostEval& newCost,
                        const vector<Object*>& objectsIn,
                        uint axis)
  {
    const uint size = objectsIn.size();
    BBox          global;
    vector<Event> events;
    for (uint i = 0; i < size; ++i)
    {
      Event newEvent;
      BBox  box       = objectsIn[i]->box;
      newEvent.pos    = (box.center())[axis];
      newEvent.object = objectsIn[i];

      events.push_back(newEvent);

      global.extend(box);
    }

    sort(events.begin(), events.end(), CompareEvents());

    BBox left;

    const int numEvents = events.size();
    int numLeft         = 0;
    int numRight        = numEvents;

    for (int i = 0; i < numEvents; ++i)
    {
      events[i].numLeft  = numLeft;
      events[i].numRight = numRight;
      events[i].leftArea = left.computeSA();

      left.extend(events[i].object->box);

      ++numLeft;
      --numRight;
    }

    BBox right;

    newCost.cost  = FLT_MAX;
    newCost.event = -1;

    for (int i = numEvents - 1; i >= 0; --i)
    {
      right.extend(events[i].object->box);

      if (events[i].numLeft > 0 && events[i].numRight > 0)
      {
        events[i].rightArea = right.computeSA();

        float currentCost = (events[i].numLeft * events[i].leftArea +
                             events[i].numRight * events[i].rightArea);
        currentCost /= global.computeSA();
        currentCost *= isecCost;
        currentCost += travCost;

        events[i].cost = currentCost;
        if (currentCost < newCost.cost)
        {
          newCost.cost     = currentCost;
          newCost.pos      = events[i].pos;
          newCost.axis     = axis;
          newCost.event    = i;
          newCost.numLeft  = events[i].numLeft;
          newCost.numRight = events[i].numRight;
        }
      }
    }

    return (newCost.event != -1);
  }

  void BVH::updateBounds(Node* node)
  {
    node->box.reset();

    if (node->objIDs.empty())
    {
      // Interior node, recurse on children
      updateBounds(node->left);
      updateBounds(node->right);

      node->box.extend(node->left->box);
      node->box.extend(node->right->box);
    }
    else
    {
      // Leaf node, bound triangles
      for (uint i = 0; i < node->objIDs.size(); ++i)
      {
        node->box.extend(objects[node->objIDs[i]]->box);
      }
    }
  }

  void BVH::computeCost(float& totalCost, Node* node, float globalArea) const
  {
    // Compute cost of BVH according to SAH:
    //
    //   C = travCost*sum[n in nodes]  SA_n/SA_scene +
    //       isecCost*sum[l in leaves] nprims_l(SA_l/SA_scene)

    const BBox& bbox = node->box;

    // Add traversal cost
    totalCost += bbox.computeSA() / globalArea;
    
    if (node->objIDs.size() == 0)
    {
      // Interior node, recurse on children
      computeCost(totalCost, node->left,  globalArea);
      computeCost(totalCost, node->right, globalArea);
    }
    else
    {
      // Leaf node, add intersection cost
      totalCost += node->objIDs.size() * bbox.computeSA() / globalArea;
    }
  }

  void BVH::printStats(bool all) const
  {
      Output("BVH:" << endl);
      Output("  scene bounds     = " << root->box << endl);
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

      if (all)
      {
        Output("  root node = " << root->box << endl);
        Output("    lchild  = " << root->left << endl);
        Output("    rchild  = " << root->right << endl);
        Output("    ntris   = " << root->objIDs.size() << endl);

        printNode(root->left);
        printNode(root->right);

        Output(endl);
      }
    }

  void BVH::printNode(const Node* node) const
  {
    const uint size = node->objIDs.size();
    if (size == 0)
    {
      Output("  interior node[" << node << "] = "
             << node->box << endl);
      Output("    lchild = " << node->left << endl);
      Output("    rchild = " << node->right << endl);
      Output("    ntris  = " << size << endl);
      printNode(node->left);
      printNode(node->right);
    }
    else
    {
      Output("  leaf node[" << node <<"] = "
             << node->box << endl);
      Output("    ntris = " << size << endl);
      Output("   ");
      for (uint i = 0; i < size; ++i)
      {
        Output(" " << node->objIDs[i]);
      }
    }
  }

  ostream& operator<<(ostream& out, const BVH::Node& n)
  {
    out << n.box << ' ' << n.objIDs.size();
    return out;
  }

} // namespace FileIO
