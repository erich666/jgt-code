Neighand: Neighborhood Handling library
Version 0.2, June 2007.
Nicolas Brodu <nicolas.brodu@free.fr>


Presentation:
-------------

The goal of this library is to find 3D neighbors efficiently. It stores the
position of objects in 3D. The user can then ask what are the neighbor objects
for any given point, within a radius. You may either ask for all the objects
or only the K-nearest neighbors. This problem of finding neighbors is also
known as a locality query, or a neighborhood search.

This library has been designed for performance and ease of use. It consists
of various query methods adapted to dynamic environments where the objects
may move. These methods are explained in the article "Query Sphere Indexing
for Neighborhood Requests".

A short tutorial is presented below, followed by the full API documentation.



Organisation:
-------------

- The "src" subdirectory contains the sources. You may copy the content
wherever you wish, it is self-contained (it uses only the standard C and C++
libraries). Each source file contains a description of what it does, and the
functions are well documented. See below for the file list.

- The "tests" subdirectory contains example programs and a Makefile. These are
not meant to be tutorials, but rather useful tools. However, learning how to
use the library from them is possible too. See below for the file list.

- The file "QuerySphereIndexing.pdf" is a copy of the aforementioned article.
It explains the rationale of the library.



Usage (project setup):
----------------------

- Copy the src/ directory anywhere in your include path

- Include the "neighand.h" file where you need it.

- That's all. The library is header-only, no link is required.



Usage (Programming):
--------------------

- Check that your compiler supports partial template specializations.
  The library was tested with g++ 3.4 and g++ 4.1.2 on Debian/Linux.

- Include the "neighand.h" file. Use the neighand namespace if you wish.

- Define one NeighborhoodHandler object with the parameters you need. The
  possible template and function arguments are explained below.

- Use the "insert" method to insert objects in the region of interest, the
  "update" method to move them, and the "remove" method to remove them if
  needed.

- You may then either:

  - Find the neighbors of a given point, either all of them or only the
    K-nearest ones. The "findNeighbors", "findNearestNeighbors", and
    "findNearestNeighbor" methods are what you need then.

  - Provide a functor or a callback, that will be called on all the neighbors
    found around a given center. The "applyToNeighbors" method is what you
    need in this case.

- Please look at the API below for other useful functions. In particular,
  you may find the "squaredDistance" method handy for cyclic worlds.



Tutorial:
---------

struct Agent {
    int number;                  // put your fields and methods here
    ObjectProxy<Agent>* proxy;   // handy reference, this Agent ID
};
...

// Define a non-cyclic world with a 16x16x16 discretization
typedef NeighborhoodHandler<Agent,4,4,4,false,false,false> NH;

// Make to world cover the region of interest from 0 to 100 in each dimension
NH nh(0,0,0,6.25);

// Insert a few objects
...
agent1.proxy = nh.insert(x1, y1, z1, &agent1, ProxiedObjectRemapper<Agent>());
agent2.proxy = nh.insert(x2, y2, z2, &agent2, ProxiedObjectRemapper<Agent>());
...

// Find all objects within distance d from the point at (x,y,z)
vector<Agent*> neighbors;
nh.findNeighbors(x, y, z, d, neighbors);

// Find the closest object from agent1 (but not itself), within d max. distance
NearestNeighbor<Agent> neighbor;
nh.findNearestNeighbor(agent1.proxy, d, &neighbor);

cout << "The nearest neighbor of agent1 is the agent number " << neighbor.object->number << endl;
cout << "It is at d^2=" << neighbor.squaredDistance << " away from agent1" << endl;



API documentation (see also neighand.h for details):
----------------------------------------------------

- Template arguments:

UserObject:     The user type, like Agent in the tutorial
exp2div(x/y/z): The power-of-two for the number of cells in each dimension.
                Ex: 4 => 16 cells
wrap(X/Y/Z):    Whether the world is cyclic or not in each dimension.
layerZ:         Ignored, reserved for a future extension
_Allocator:     A C++ allocator for getting memory. Defaults to the standard
                allocator.

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ = false, class _Allocator = std::allocator<ObjectProxy<UserObject> > > NeighborhoodHandler


- Constructor:

min(x|y|z):         The corner of the region of interest in world coordinates.
                    It is especially important in case there is no wrapping
                    along at least one direction.
cellSize:           The edge size of the cubic cells (the same for each
                    direction). Thus, for non-cyclic worlds,
                    min[xyz] + cellSize * 2^exp2div[xyz] is the corner with
                    maximum coordinate for the region of interest.
objectInitCapacity: Initial capacity in number of objects. Memory is reserved
                    to hold that max amount of objects, and will be increased
                    if necessary. Use this parameter to avoid run-time memory
                    allocation on further inserts if you know in advance how
                    many objects will be stored at most in the database.
precompFile:        The name of a binary file that can be used to cache the
                    precomputations from one run to the other. Default is empty,
                    so the precomputations are not cached. This should be a
                    valid file name. Valid files are read, invalid files are
                    overwritten by a valid one. The file is ONLY used for
                    initialization, and not accessed anymore after the
                    constructor returns.

NeighborhoodHandler(float minx, float miny, float minz, float cellSize, uint32_t objectInitCapacity = 1024, const char* precompFile = "")


- Inserting an object

x/y/z:           The position of the object
object:          A pointer to the object
remapperFunctor: A functor with the (UserObject*, ObjectProxy<UserObject>*)
                 signature. It is called when a user object is given a new
                 proxy. See the rationale in neighand.h and why this leads to
                 important optimizations.
Return:          A proxy = a key identifier for this object

template<typename RemapperFunctor> ObjectProxy<UserObject>* insert(float x, float y, float z, UserObject* object, RemapperFunctor remapperFunctor)


- Removing an object

proxy:           The proxy of the object to remove
remapperFunctor: See the insert function

template<typename RemapperFunctor> void remove(ObjectProxy<UserObject>* proxy, RemapperFunctor remapperFunctor)


- Updating the position of an object

proxy:  The proxy of the object to move
x/y/z:  The new position of the object

void update(ObjectProxy<UserObject>* proxy, float x, float y, float z)


- Finding all the neighbors of a given point

x/y/z:     The query center
d:         The maximum distance to look for neighbors
p:         An object proxy used as the query center. In that case the object
           itself is excluded from the neighbors. Call the function with
           p->x,p->y,p->z if you wish to include this object.
neighbors: A vector of object pointers filled with the neighbors that are
           found. Note: The neighbors distances are not provided here simply
           because thay are not always computed, and perhaps not necessary
           as well in the user application. Use the squaredDistance function
           (see below) if you need it.

void findNeighbors(float x, float y, float z, float d, std::vector<UserObject*>& neighbors)
void findNeighbors(ObjectProxy<UserObject>* p, float d, std::vector<UserObject*>& neighbors)


- Finding (at most) the K nearest neighbors of a given point

x/y/z:     The query center
d:         The maximum distance to look for neighbors
K:         The maximum number of neighbors to return.
p:         An object proxy used as the query center. See findNeighbors.
neighbor:  A vector or an array of objects that holds the neighbors that were
           found. The distances are provided in addition to the objects
           because they were necessarily already computed.
Return:    The number of objects that were found, in case an array was
           provided. Use the vector size() method otherwise.

void findNearestNeighbors(float x, float y, float z, float d, std::vector<NearestNeighbor<UserObject> >& neighbor, unsigned int K)
void findNearestNeighbors(ObjectProxy<UserObject>* p, float d, std::vector<NearestNeighbor<UserObject> >& neighbor, unsigned int K)
int findNearestNeighbors(float x, float y, float z, float d, NearestNeighbor<UserObject>* neighbor, unsigned int K)
int findNearestNeighbors(ObjectProxy<UserObject>* p, float d, NearestNeighbor<UserObject>* neighbor, unsigned int K)


- Finding the nearest neighbor of a given point

x/y/z:     The query center
d:         The maximum distance to look for the neighbor
p:         An object proxy used as the query center. See findNeighbors.
neighbor:  A pointer to an object for holding the neighbor that is found
           together with its squared distance to the query center.
Return:    1 if a neighbor was found, 0 otherwise.

int findNearestNeighbor(float x, float y, float z, float d, NearestNeighbor<UserObject>* neighbor)
int findNearestNeighbor(ObjectProxy<UserObject>* p, float d, NearestNeighbor<UserObject>* neighbor)


- Applying a functor to all the objects in the viciny of a given point

x/y/z:  The query center
d:      The maximum distance to look for neighbors
f:      The functor or callback to call on each neighbor
p:      An object proxy used as the query center. See findNeighbors.
Return: The functor object (which might have an internal state, useful for ex.
        to count the number of neighbors).

template<typename Functor> Functor applyToNeighbors(ObjectProxy<UserObject>* p, float d, Functor f)
template<typename Functor> Functor applyToNeighbors(float x, float y, float z, FloatType d, Functor f)
typedef void (*Callback)(UserObject* object, void* userData);
void applyToNeighbors(ObjectProxy<UserObject>* p, float d, Callback f, void* userData)
void applyToNeighbors(float x, float y, float z, float d, Callback f, void* userData)


- Applying a functor to all the objects whatever their position

f:      The functor or callback to call on each object
Return: The functor object

template<typename Functor> Functor applyToAll(Functor f)
void applyToAll(Callback f, void* userData)


- Compute the squared distance from point differences

dx/dy/dz: The difference between the points along each dimension: x2-x1, etc.
Return:   Simply dx*dx+dy*dy+dz*dz when there is no wrapping. However when a
          dimension is wrapping it is taken into account, so you'll probably
          find this function very convenient in wrapping worlds.

float squaredDistance(float dx, float dy, float dz)


- Get/Set the query method and method auto-detection parameters

method: One of Auto, Sphere, Cube, NonEmpty, or Brute. See the article for
        what this means. The default is Auto, that will try to select the best
        method for you, but you may force one of them is the Auto detection
        does not give the best results (some user applications are better
        suited to some methods).
weight: Additional weight factors to help the auto-detection routine. For
        example your architecture may be advantageous to the Brute-force
        method and in that case give it a weight <1.0
Return: The current method and weights

void setQueryMethod(QueryMethod method); QueryMethod getQueryMethod()
void setWeightSphere(float weight); float getWeightSphere()
void setWeightCube(float weight); float getWeightCube()
void setWeightNonEmpty(float weight); float getWeightNonEmpty()
void setWeightBrute(float weight); float getWeightBrute()


- Get the automatic selection routine estimated cost factors

x/y/z:   The query center
d:       The maximum distance to look for neighbors
factorX: Will hold the estimated cost of the method X on return
Return:  The method that would be chosen by the automatic selection routine in
         this situation. If you don't like this choice then you may weight the
         methods using the setWeight functions.

QueryMethod getAutoFactors(float x, float y, float z, float d, float& factorSphere, float& factorCube, float& factorNonEmpty, float& factorBrute)
QueryMethod getAutoFactorsClosest(float x, float y, float z, float d, int N, float& factorSphere, float& factorCube, float& factorNonEmpty, float& factorBrute)


- Get statistics (only available if NEIGHAND_SELECT_METHOD_STAT is defined)

These counters are incremented each time the corresponding method is used.

uint32_t statSphere, statCube, statNonEmpty, statBrute;
void resetStat();



Source files:
-------------

neighand.h: The main header file and the only one you need to include
neighand_structures.hpp: Declare the types like ObjectProxy, QueryMethod, etc.
neighand_apply.hpp: Main query routine, applies a functor to neighbors
neighand_apply_all.hpp: Implements the corresponding function
neighand_apply_checkdist.hpp: Subroutine for neighand_apply.hpp
neighand_apply_processlines_cond.hpp: Subroutine, conditional object inclusion
neighand_apply_processlines_uncond.hpp: Unconditional object inclusion
neighand_closest.hpp: Similar to neighand_apply but for K nearest neighbors
neighand_closest_processlines.hpp: Subroutine for neighand_closest.hpp
neighand.hpp: Implements the other declarations of the main header
neighand_helpers.hpp: Common utilities to all wrapping cases
neighand_wraphelper_ffff.hpp: Specialized routines for the no wrapping case
neighand_wraphelper_tttf.hpp: Specialized routines for the all-wrapping case
neighand_wraphelper_ttff.hpp: Specialized routines for wrapping along X and Y



Test files:
-----------

consistencyTest.cpp: Checks that the library gives correct results. It's
                     probably a good idea to call this test at least once for
                     each set of compilation options you're using.

perfTest.cpp: Used to generate the histograms in the article. Might be useful
              too to get an idea how the automatic selection routine performs
              on your system, though you should instead really call the
              getAutoFactors for a representative case of your application

distknn.cpp: Used to generate the plot in the article

lq.c/lq.h: Locality query routine from the OpenSteer project. Similar to the
           Cube query method in the non-wrapping case.

lq.c.diff: Change from the OpenSteer version so as to make lq.c compile
           independently. Also contains a bug correction not yet in the
           OpenSteer CVS, as of 07 June 07.

libkdtree++: Slighly patched 0.2.1 version of the library providing
             a kd-tree C++ implementation with an interface similar to
             the STL.

libkdtree++-0.2.1.diff: The exact changes operated on the library

kdtree++: Symbolic link, so the kdtree++ library is "installed" here.

KDLQConsistencyTest.cpp: Checks that the kd-tree and the lq bin-lattice files
                         work as intended.

Makefile: Instructions for compiling the test programs, and in particular the
          compiler optimization options (default is -O3). Can also run the
          consistency and performance tests automatically for all wrapping
          cases, just type "make consistencyTest" for example.



History:
--------

v0.2:
- Added support for directly calling the different query methods (sphere
  indexing, bin-lattice cube, non-empty cell list, brute-force).
- Improved the algorithm for automatically selecting the best method.
- API improvements and modifications.
- Made the library compatible with user-defined memory allocators.
- Made the library compatible with the aliasing rule.
- Removed support for separate CPP specialization. That feature was a
  maintenance nightmare and it was (and still is) simpler to just specialize
  the templates in the user project CPP files anyway.

v0.1:
- Initial public release


Happy hacking!

Nicolas Brodu, 2007
Code released according to the GNU LGPL, v2 or above.
