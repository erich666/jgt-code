/*
    Neighand: Neighborhood Handling library

    The goal of this class is to find 3D neighbors efficiently.
    It stores the position of objects in 3D. The user can then ask what
    are the neighbor objects for any given point, within a radius
    (aka locality query, or neighborhood search).

    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/
#ifndef NEIGHAND_H
#define NEIGHAND_H

// posix_memalign may require the next define to be declared in stdlib on some systems
//#define _XOPEN_SOURCE 600
#include <stdlib.h>
#include <stdint.h>
#include <fenv.h>

#include <memory>
#include <vector>
#include <fstream>
#include <limits>

namespace neighand {

// This may be useful to specialize on a special float type, like a software implementation or a type with protected FPU flags. See for example the streflop library.
// This MUST be a 4-bytes type, IEEE754-compliant.
// Changing this, even to streflop types, is untested yet.
#ifdef NEIGHAND_FLOAT_TYPE
typedef NEIGHAND_FLOAT_TYPE FloatType;
#else
typedef float FloatType;
#endif

/*
Force inlining of some functions.
The compiler may think they are too big, and that's usually perfectly justified.
A too big inlined function is actually less performant than a call, because the call
re-uses the code trace L1 cache whereas the duplication due to inlining does not.
However, in our case, some "big" functions are called within critical neighborhood
loops, and only there. Some other functions are just placed in a separate class so
as to make use of template specialization, and should be inlined too.
*/
#if !defined(__GNUC__)
#define NEIGHAND_INLINE inline
#define NEIGHAND_ALWAYS_INLINE
#else
#ifdef NEIGHAND_DEBUG
#define NEIGHAND_INLINE
#define NEIGHAND_ALWAYS_INLINE __attribute__ ((__noinline__))
#else
#define NEIGHAND_INLINE inline
#define NEIGHAND_ALWAYS_INLINE __attribute__ ((__always_inline__))
#endif
#endif

// Internal data structures
#include "neighand_structures.hpp"

// Partially specialized template helpers for different combinations of wrapping
#include "neighand_helpers.hpp"


/*
    Template arguments:
    - UserObject: The user type for the objects in the database.
    - exp2div(x|y|z): 2-exponents for the subdivisions in (x|y|z). Ex: 5 = 32 cells.
                      Note: limitation for exp2divx + exp2divy + exp2divz <= 16
                      Note2: Of course, each one should also be >=1
                      Note3: These conditions are compile-time statically checked.
    - wrap(x|y|z): whether to consider a cyclic world in that dimension or not, that
                   is, wrap one edge with the other. Non-cyclic programs should be
                   designed so as to minimize the number of objects "outside" the
                   main region of interest for best performance.
    - layerZ: Reserved for a future extension. Ignored for now.
    - _Allocator: A C++ allocator for getting memory. Defaults to the standard allocator.

    Thread safety: All access to this object should be made within the same thread.
*/
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ = false, class _Allocator = std::allocator<ObjectProxy<UserObject> > >
class NeighborhoodHandler : private _Allocator {
public:
COMPILE_TIME_ASSERT(exp2divx>=1)
COMPILE_TIME_ASSERT(exp2divy>=1)
COMPILE_TIME_ASSERT(exp2divz>=1)
COMPILE_TIME_ASSERT(exp2divx+exp2divy+exp2divz<=16)

typedef _Allocator Allocator;


////////////////////////////    PUBLIC API    ////////////////////////////

//// PART 1: Initialization & destruction

    // Constructor arguments:
    // min(x|y|z) is the corner of the region of interest in world coordinates.
    //            It is especially important in case there is no wrapping
    //            along at least one direction.
    // cellSize is the edge size of the cubic cells (the same for each direction).
    //          Thus, for non-cyclic worlds, min[xyz] + cellSize * 2^exp2div[xyz]
    //          is the corner with maximum coordinate for the region of interest.
    // objectInitCapacity: Initial capacity in number of objects. Memory is reserved
    //                     to hold that max amount of objects, and will be increased
    //                     if necessary. Use this parameter to avoid run-time memory
    //                     allocation on further inserts if you know in advance how
    //                     many objects will be stored at most in the database.
    // precompFile is the name of a binary file that can be used to cache the
    //             precomputations from one run to the other. Default is empty,
    //             so the precomputations are not cached. This should be a valid file
    //             name. Valid files are read, invalid files are overwritten by a
    //             valid one. The file is ONLY used for initialization, and not
    //             accessed anymore after the constructor returns.
    NeighborhoodHandler(FloatType minx, FloatType miny, FloatType minz, FloatType cellSize, uint32_t objectInitCapacity = 1024, const char* precompFile = "");

    ~NeighborhoodHandler();

//// PART 2: Managing user objects in the region of interest

    // Insert a user object at the given location (world coordinates)
    // The object can then be considered in the neighborhood queries.
    // Return a proxy = a key identifier for this object
    // Note: It may be that some user object / ObjectProxy mappings are changed in the process.
    //       The rationale for the change is an optimization of memory (see the "remove" function notes).
    //       If the change matters (and it probably does to your application) then provide a mapper functor
    //       That mapper functor should have the (UserObject*, ObjectProxy<UserObject>*) signature
    //       and should change the given user-object-to-ObjectProxy mapping
    // Example: Your objects have an internal "proxy" field. Then the mapper should change
    //          that field to the new updated proxy for that object: object->proxy = updated_proxy
    //          See also the example ProxiedObjectRemapperFunctor in neighand_structures.h which does exactly that.
    template<typename RemapperFunctor>
    NEIGHAND_INLINE ObjectProxy<UserObject>* insert(FloatType x, FloatType y, FloatType z, UserObject* object, RemapperFunctor remapperFunctor) NEIGHAND_ALWAYS_INLINE;

    // Remove an object proxy from the neighborhood considerations.
    // The proxy must have been returned by "insert" and must be removed only once.
    // Rationale: Object proxies are stored internally as a contiguous memory array for maximum performances.
    //            When removing an object the proxy used by that object is replaced by the one at the end
    //            of the array, so the end-of-array element may then be deallocated.
    //            The mapper is called during this operation to maintain the user object/proxy correct associations.
    //            The alternative from v0.1 is a simpler API that really removes, but cause memory fragmentation.
    //            The apparent save in complexity from the simpler API is thus just shifted to a defrag
    //            or equivalent utility, which the user then has to handle. Or possibly garbage collection,
    //            auto-defrag, etc, which then causes perf problems at one point or another in addition to
    //            having to deal with non-contiguous memory arrays which induce processing costs too.
    //            On the other hand updating an external ref is as easy as an affectation, real fast constant-time,
    //            avoids further mem frag problems, and allows for very efficient traversal routine for the
    //            linear memory array that would not be possible with fragged memory.
    // See also the Note in the insert function.
    template<typename RemapperFunctor>
    NEIGHAND_INLINE void remove(ObjectProxy<UserObject>* proxy, RemapperFunctor remapperFunctor) NEIGHAND_ALWAYS_INLINE;


    // Update a managed object position
    // proxy:  The proxy of the object to move
    // x/y/z:  The new position of the object
    NEIGHAND_INLINE void update(ObjectProxy<UserObject>* proxy, FloatType x, FloatType y, FloatType z) NEIGHAND_ALWAYS_INLINE;


//// PART 3: Neighborhood query functions


    /* Set the method to use for the queries.
        One method may be better than the others according to the query distance, number of objects, size of world, wrapping or not...
        General guidelines:
        - Small distances are better handled by the Cube method (aka bin-lattice norm-1 query). "Small" may mean a tenth of the cell size for high loads, or a few cells for very low loads. You might also consider changing the discretization size in these cases.
        - Large distances are better handled by either the non-empty list or the brute force. Use the non-empty list if you expect that your objects are concentrated in a few cells which may be rejected in one go with a single distance check. Use the brute force method otherwise as it has the lowest setup and per-processed-object costs (but it processes all objects).
        - Sphere generally works best for all intermediate query distances, especially in wrapping worlds and for high load ratios. Sphere also has a premature stopping cabability which make it well adapted to K-nearest neighbors.
        - Auto selects at run-time which method to use. The estimated volumes that will be processed by each method are multiplied by weights to reflect their different processing costs, then the method with estimated lowest cost is chosen. The interest is that according to the position of the query center in a non-wrapping world the Sphere and Cube methods may have varying volumes inside the region of interest, hence the Auto method allows to dynamically choose which is the best method.
        Default is Auto but with equal weights. Tuning these parameters may get you the best of the library. Try playing with the other methods too, if one of them is best suited to your application then you may avoid the (slight) cost of the weighting scheme.
    */
    // The current query method. Default: Auto
    NEIGHAND_INLINE QueryMethod getQueryMethod() {return queryMethod;}
    NEIGHAND_INLINE void setQueryMethod(QueryMethod method);

    // Additional parameters for Auto. Defaults: 1.0 for all weights
    NEIGHAND_INLINE FloatType getWeightSphere() {return weightSphere;}
    NEIGHAND_INLINE FloatType getWeightCube() {return weightCube;}
    NEIGHAND_INLINE FloatType getWeightNonEmpty() {return weightNonEmpty;}
    NEIGHAND_INLINE FloatType getWeightBrute() {return weightBrute;}
    // Setter accessors do additional job
    NEIGHAND_INLINE void setWeightSphere(FloatType weight);
    NEIGHAND_INLINE void setWeightCube(FloatType weight);
    NEIGHAND_INLINE void setWeightNonEmpty(FloatType weight);
    NEIGHAND_INLINE void setWeightBrute(FloatType weight);


    // Main query routines: apply a functor to all neighbors within a given range.
    // The function that takes a proxy as argument excludes that proxy object from the search.
    // The function that takes a x/y/z center position applies to functor to all objects around that center
    // The rationale is that you can use the x/y/z function with p->x,p->y,p->z to include p->object if need be.
    // Arguments:
    // p: An object proxy that is the center of the query
    // x/y/z: The center coordinates
    // d: The maximum distance to look for neighbors (consider using applyToAll for best performances when covering the whole region of interest).
    // f: A functor taking a UserObject* object as argument. This functor is called exactly once for each neighbor.
    template<typename Functor> NEIGHAND_INLINE Functor applyToNeighbors(ObjectProxy<UserObject>* p, FloatType d, Functor f) NEIGHAND_ALWAYS_INLINE;
    template<typename Functor> NEIGHAND_INLINE Functor applyToNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, Functor f) NEIGHAND_ALWAYS_INLINE;

    // Apply a functor to all managed objects, whatever their position
    template<typename Functor> NEIGHAND_INLINE Functor applyToAll(Functor f) NEIGHAND_ALWAYS_INLINE;

    // C API equivalent functions, using a callback instead of a functor.
    typedef void (*Callback)(UserObject* object, void* userData);
    NEIGHAND_INLINE void applyToNeighbors(ObjectProxy<UserObject>* p, FloatType d, Callback f, void* userData) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE void applyToNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, Callback f, void* userData) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE void applyToAll(Callback f, void* userData) NEIGHAND_ALWAYS_INLINE;

    // Return a list of all neighbors within a given radius.
    // The function that takes a proxy as argument excludes that proxy object from the returned list.
    // The function that takes a x/y/z position returns all objects in neighborhood
    // This is just a wrapper around the previous functions with a functor that builds the neighbor vector.
    // Note: The vector this function returns is NOT sorted by distance.
    //       The distance for each object in that vector is not even returned simply because
    //       it needs not be computed in many cases (and thus it would be a waste of time).
    //       => You may still compute it easily if you need it with the provided squaredDistance function
    // ex: vector<UserObject*> neighbors; int nfound = findNeighbors(my_proxy, dist, neighbors);
    // Tip: reserve() your vector memory in advance may avoid run-time dynamic allocation
    NEIGHAND_INLINE void findNeighbors(ObjectProxy<UserObject>* p, FloatType d, std::vector<UserObject*>& neighbors) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE void findNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, std::vector<UserObject*>& neighbors) NEIGHAND_ALWAYS_INLINE;

    // Find at most the N nearest neighbors within a given radius.
    // The function that takes a proxy as argument excludes that proxy object from the returned list.
    // The neighbors are returned in the vector provided by the user, sorted by distance.
    // ex: vector<NearestNeighbor<UserObject> > neighbors; findNearestNeighbors(my_position, dist, neighbors, 5);
    NEIGHAND_INLINE void findNearestNeighbors(ObjectProxy<UserObject>* p, FloatType d, std::vector<NearestNeighbor<UserObject> >& neighbor, unsigned int N) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE void findNearestNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, std::vector<NearestNeighbor<UserObject> >& neighbor, unsigned int N) NEIGHAND_ALWAYS_INLINE;

    // Find at most the N nearest neighbors within a given radius (C API version)
    // The function that takes a proxy as argument excludes that proxy object from the returned list.
    // The neighbors are returned in the array provided by the user, sorted by distance.
    // Return the number of neighbors found (potentially 0).
    // ex: NearestNeighbor<UserObject> array[5]; int nfound = findNearestNeighbors(my_position, dist, array, 5);
    NEIGHAND_INLINE int findNearestNeighbors(ObjectProxy<UserObject>* p, FloatType d, NearestNeighbor<UserObject>* neighbor, unsigned int N) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE int findNearestNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, NearestNeighbor<UserObject>* neighbor, unsigned int N) NEIGHAND_ALWAYS_INLINE;

    // Find only the closest neighbor within a certain radius (or one of the ex-aequo if any)
    // The function that takes a proxy as argument excludes that proxy object for a neighbor candidate.
    // Returns 1 if a neighbor was found, 0 otherwise.
    // Note: This function is faster than calling the previous ones with N=1
    // ex: NearestNeighbor<UserObject> neighbor; int found = findNearestNeighbor(my_position, dist, &neighbor);
    NEIGHAND_INLINE int findNearestNeighbor(ObjectProxy<UserObject>* p, FloatType d, NearestNeighbor<UserObject>* neighbor) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE int findNearestNeighbor(FloatType x, FloatType y, FloatType z, FloatType d, NearestNeighbor<UserObject>* neighbor) NEIGHAND_ALWAYS_INLINE;



//// PART 4: Helpers and other utilities


    // Return a squared distance for a position difference, taking into account world wrapping.
    // So, this just returns "dx*dx + dy*dy + dz*dz" when there is no wrapping, but
    // performs modulo arithmetic first in the wrapping dimensions.
    // Ex: FloatType dsq = squaredDistance(x1-x2, y1-y2, z1-z2);
    // Note: Exposed to user API from wrap helper. You may find other helper functions useful too.
    // Note2: See neighand_helpers.h for a fast modulo function you may reuse.
    NEIGHAND_INLINE FloatType squaredDistance(FloatType dx, FloatType dy, FloatType dz) NEIGHAND_ALWAYS_INLINE {
        return helper.squaredDistance(dx,dy,dz);
    }

    // Main helper object. This allows to keep a generic template version here,
    // and only the helper will be specialized for the different wrap modes
    WrapHelper<UserObject,exp2divx,exp2divy,exp2divz,wrapX,wrapY,wrapZ,layerZ,_Allocator> helper;

    // This is the maximum squared distance in number of cells that may separate
    // 2 points in the region of interest
    // Propagated from helper to this name space for convenience
    enum {
        MaxDQ = WrapHelper<UserObject,exp2divx,exp2divy,exp2divz,wrapX,wrapY,wrapZ,layerZ,_Allocator>::MaxDQ
    };

    // Simulate a query at the given location and get what factors for each method would be computed by the Auto mechanism
    // Return the method that would be selected
    // This function might help you tune the weights for a given situation
    NEIGHAND_INLINE QueryMethod getAutoFactors(FloatType x, FloatType y, FloatType z, FloatType d, FloatType& factorSphere, FloatType& factorCube, FloatType& factorNonEmpty, FloatType& factorBrute);

    // Simulate a closest query at the given location and get what factors for each method would be computed by the Auto mechanism
    // Return the method that would be selected
    // This function might help you tune the weights for a given situation
    NEIGHAND_INLINE QueryMethod getAutoFactorsClosest(FloatType x, FloatType y, FloatType z, FloatType d, int N, FloatType& factorSphere, FloatType& factorCube, FloatType& factorNonEmpty, FloatType& factorBrute);

#ifdef NEIGHAND_SELECT_METHOD_STAT
    uint32_t statSphere, statCube, statNonEmpty, statBrute;
    NEIGHAND_INLINE void resetStat() NEIGHAND_ALWAYS_INLINE {
        statSphere = statCube = statNonEmpty = statBrute = 0;
    }
#endif

////////////////////////////    INTERNALS    ////////////////////////////
protected:

    NEIGHAND_INLINE void removeNoDealloc(ObjectProxy<UserObject>* proxy) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE void insertNoAllocate(ObjectProxy<UserObject>* proxy, FloatType x, FloatType y, FloatType z, uint32_t idx) NEIGHAND_ALWAYS_INLINE;
    NEIGHAND_INLINE void updateWeightBaseTables() NEIGHAND_ALWAYS_INLINE;

    // Distance -> sphere cell offsets table lookup
    // indexed by dq, max entries into sphereOffsets array
    uint32_t* baseDistanceArray;
    // all sphere offsets as for no shaving, ordered starting from the center
    uint32_t* sphereOffsets;
    // complements that are not shaved off. Each shavingComplement[x] is an array indexed by dq,
    // of indices into shavingOffsets[x]
    uint32_t* shavingComplement[64];
    // arrays of all cell offset complements, stacked up: complement for dq=0, then for 1, then for 2...
    uint32_t* shavingOffsets[64];

    // Average "sphere" volume for each 1/32th of cell distance
    // allows quick estimation for which method is fastest
    uint32_t maxWorldDist32;
    FloatType* weightSphereTable, *weightSphereTableBase, *volumeSphereTable, *weightSphereTableClosest, *weightSphereTableClosestBase;
    FloatType* weightNonEmptyTable, *weightNonEmptyTableBase;
    FloatType* weightBruteTable, *weightBruteTableBase;
    FloatType weightCubeWithLoad;

    // Array of all the cells, a.k.a "blocks", or "bricks"
    CellEntry<UserObject>* cells;
    // Faster to lookup when not using non-empty list
    ObjectProxy<UserObject>** cachedLists;

    // All proxies in a large linear memory array
    ObjectProxy<UserObject>* allProxies;
    uint32_t numProxies;
    uint32_t proxyCapacity;

    // The inverse of the cell size
    FloatType cellSizeSqInv;

    // Total number of non-empty cells in the main region
    uint32_t totalNonEmpty;
    // Linked list of non-empty cells in the main region
    CellEntry<UserObject>* firstNonEmptyCell;

    QueryMethod queryMethod;
    FloatType weightSphere, weightCube, weightNonEmpty, weightBrute;
    bool updateWeightBaseTablesNeeded;

};

} // end namespace

// Declare all the routines in the header => no link.
#include "neighand.hpp"

#endif


