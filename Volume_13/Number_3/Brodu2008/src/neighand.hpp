/*
    Neighand: Neighborhood Handling library

    The goal of this class is to find 3D neighbors efficiently.
    It stores the position of objects in 3D. The user can then ask what
    are the neighbor objects for any given point, within a radius
    (aka locality query, or neighborhood search).

    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines the body of the member functions for the main class.
    It is included from the neighand.h file, don't include it directly

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

#include <string.h>
#include <math.h>

// And this is an included header file for the generic version
#define NEIGHAND_TEMPLATE_ARGUMENTS UserObject,exp2divx,exp2divy,exp2divz,wrapX,wrapY,wrapZ,layerZ,_Allocator

namespace neighand {

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::NeighborhoodHandler(FloatType minx, FloatType miny, FloatType minz, FloatType cellSize, uint32_t objectInitCapacity, const char* precompFile)
: helper(minx, miny, minz, cellSize)
{

#ifdef NEIGHAND_SELECT_METHOD_STAT
    resetStat();
#endif

    cellSizeSqInv = FloatType(1.0f) / (cellSize*cellSize);

    maxWorldDist32 = uint32_t(sqrtf(MaxDQ)*32.f+55.425626f);

    // Average "sphere" volume for each 1/32th of cell distance
    // allows quick estimation for which method is fastest
    volumeSphereTable = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    for (uint32_t i=0; i<=maxWorldDist32; ++i) volumeSphereTable[i] = FloatType(0.f);

    // The sphere offsets pre-computation depends on wrapping
    WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::buildDistanceArray(baseDistanceArray, sphereOffsets, shavingComplement, shavingOffsets, maxWorldDist32, volumeSphereTable, precompFile, typename WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::UINT32Allocator(*this));


    // allocate the other tables as well
    weightSphereTable = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    weightSphereTableBase = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    weightSphereTableClosest = typename Allocator::template rebind<FloatType>::other(*this).allocate(256);
    weightSphereTableClosestBase = typename Allocator::template rebind<FloatType>::other(*this).allocate(256);
    weightNonEmptyTable = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    weightNonEmptyTableBase = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    weightBruteTable = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    weightBruteTableBase = typename Allocator::template rebind<FloatType>::other(*this).allocate(maxWorldDist32+1);
    // Initialize so Brute-Force is selected by Auto when there is no object!
    // the tables are recomputed when objects are inserted
    for (uint32_t i=0; i<=maxWorldDist32; ++i) weightSphereTable[i] = FloatType(0.f);
    for (uint32_t i=0; i<=maxWorldDist32; ++i) weightSphereTableBase[i] = FloatType(0.f);
    for (uint32_t i=0; i<=maxWorldDist32; ++i) weightNonEmptyTable[i] = FloatType(0.f);
    for (uint32_t i=0; i<=maxWorldDist32; ++i) weightNonEmptyTableBase[i] = FloatType(0.f);
    for (uint32_t i=0; i<=maxWorldDist32; ++i) weightBruteTable[i] = FloatType(-1.f);
    for (uint32_t i=0; i<=maxWorldDist32; ++i) weightBruteTableBase[i] = FloatType(-1.f);
    for (uint32_t i=0; i<256; ++i) weightSphereTableClosest[i] = FloatType(0.f);
    for (uint32_t i=0; i<256; ++i) weightSphereTableClosestBase[i] = FloatType(0.f);

    queryMethod = Auto;
    weightSphere = weightCube = weightNonEmpty = weightBrute = 1.0f;

    weightCubeWithLoad = 0.0f;
    updateWeightBaseTablesNeeded = false;

    // reserve memory to hold the proxies
    proxyCapacity = objectInitCapacity;
    numProxies = 0;
    allProxies = Allocator::allocate(proxyCapacity);

    // Now, create the object structure
    cells = typename Allocator::template rebind<CellEntry<UserObject> >::other(*this).allocate(WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::ArraySize);
    // and the cells object list cache
    cachedLists = typename Allocator::template rebind<ObjectProxy<UserObject>*>::other(*this).allocate(WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::ArraySize);

    // prepare array
    for (uint_fast32_t i=0; i<WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::ArraySize; ++i) {
        // start empty
        cells[i].objects = 0;
        cells[i].nextNonEmpty = 0;
        // cache is empty too
        cachedLists[i] = 0;
    }
    // Done, array is empty and ready to serve

    // Cache the real outside cell reference, no need to get from array index each time
    // The index at 2^(exp2divx+exp2divy+exp2divz) is the dummy cell, it is not a valid main region index
    // The index just after is the real outside cell
    // => the dummy index is used to avoid duplicates when the query sphere intersects the outside region
    // => the dummy index is the "virtual" cell that would be at that position of the sphere outside, but all are mapped to the empty one.
    // This function has no effect when there is no outside cell, so in that case it doesn't matter that the indice is wrong
    helper.setOutsideCell(&cells[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::ArraySize-1]);

    firstNonEmptyCell = 0; totalNonEmpty = 0;
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::~NeighborhoodHandler()
{
    Allocator::deallocate(allProxies, proxyCapacity);
    typename Allocator::template rebind<CellEntry<UserObject> >::other(*this).deallocate(cells, WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::ArraySize);
    typename Allocator::template rebind<ObjectProxy<UserObject> *>::other(*this).deallocate(cachedLists, WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::ArraySize);

    for (int shaving = 0; shaving < 64; ++shaving) {
        typename Allocator::template rebind<uint32_t>::other(*this).deallocate(shavingComplement[shaving]-1, MaxDQ+2);
        typename Allocator::template rebind<uint32_t>::other(*this).deallocate(shavingOffsets[shaving]-1, *(shavingOffsets[shaving]-1));
    }
    typename Allocator::template rebind<uint32_t>::other(*this).deallocate(baseDistanceArray-1, MaxDQ+2);
    typename Allocator::template rebind<uint32_t>::other(*this).deallocate(sphereOffsets-1, *(sphereOffsets-1));

    // deallocate all tables
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(volumeSphereTable, maxWorldDist32+1);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightSphereTable, maxWorldDist32+1);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightSphereTableBase, maxWorldDist32+1);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightSphereTableClosest, 256);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightSphereTableClosestBase, 256);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightNonEmptyTable, maxWorldDist32+1);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightNonEmptyTableBase, maxWorldDist32+1);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightBruteTable, maxWorldDist32+1);
    typename Allocator::template rebind<FloatType>::other(*this).deallocate(weightBruteTableBase, maxWorldDist32+1);
}


template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::updateWeightBaseTables() {

    // maintain weighting tables: see neighand_apply.hpp for formula
    // Cube
    FloatType cellLoad = FloatType(numProxies) / FloatType(WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume);
    weightCubeWithLoad = weightCube * (1.0f + 1.52f * cellLoad);
    // Sphere
    for (uint32_t d32=0; d32<=maxWorldDist32; ++d32) {
        weightSphereTableBase[d32] = volumeSphereTable[d32] * (2.0f + helper.getSphereWeightingLoadFactor(d32)*cellLoad) + 10.0f;
        weightSphereTable[d32] = weightSphere * weightSphereTableBase[d32];
    }
    // Model premature stopping ability of Sphere by estimating
    // how many cells are processed on average for K objects.
    // This is an underestimate, but a fast one.
    // ignore weightLoadFactor: would have to compute equivalent d32, plus this kind of compensate the previous underestimate
    // One may always change weightSphere if that's not enough.
    for (uint32_t K=1; K<256; ++K) {
        FloatType eqV = K * FloatType(WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume) / FloatType(numProxies);
        weightSphereTableClosestBase[K] = eqV * (2.0f + cellLoad) + 10.0f;
        weightSphereTableClosest[K] = weightSphere * weightSphereTableClosestBase[K];
    }
    // NonEmpty
    for (uint32_t d32=0; d32<=maxWorldDist32; ++d32) {
        FloatType minV = d32*0.03125f; // d
        minV *= minV*minV * 4.1888f; // 4/3 pi d^3
        if (minV > WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume) minV = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume;
        //weightNonEmptyTableBase[d32] = 1.0f + 2.0f * cellLoad * minV / WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume;
        weightNonEmptyTableBase[d32] = 2.0f * numProxies * minV / WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume;
        weightNonEmptyTable[d32] = weightNonEmpty * weightNonEmptyTableBase[d32];
    }
    // Brute
    for (uint32_t d32=0; d32<=maxWorldDist32; ++d32) {
        FloatType minV = d32*0.03125f; // d
        minV *= minV*minV * 4.1888f; // 4/3 pi d^3
        if (minV > WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume) minV = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume;
        //weightBruteTableBase[d32] = cellLoad * minV / WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume;
        weightBruteTableBase[d32] = numProxies * (1.0f + minV / WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume);
        // Force selection of Brute when there is no object
        if (numProxies==0) weightBruteTableBase[d32] = -1.0f;
        weightBruteTable[d32] = weightBrute * weightBruteTableBase[d32];
    }

    updateWeightBaseTablesNeeded = false;
}





template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::insertNoAllocate(ObjectProxy<UserObject>* proxy, FloatType x, FloatType y, FloatType z, uint32_t idx)
{
    CellEntry<UserObject>* cell = &cells[idx];

    proxy->x = x;
    proxy->y = y;
    proxy->z = z;

    proxy->next = cell->objects;
    proxy->prev = 0;
    cell->objects = proxy;
    if (proxy->next) proxy->next->prev = proxy;

    // Was this an empty cell ? => Maintain non-empty list if needed
    if ((!cachedLists[idx]) && WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isInside(idx)) {
        if ((firstNonEmptyCell==0) || (cell < firstNonEmptyCell)) {
            cell->nextNonEmpty = firstNonEmptyCell;
            firstNonEmptyCell = cell;
        } else {
            // prev will find first object in worse case, hence cannot go <0
            CellEntry<UserObject>* prev = cell-1; while (prev->objects==0) --prev;
            cell->nextNonEmpty = prev->nextNonEmpty;
            prev->nextNonEmpty = cell;
        }
        ++totalNonEmpty;
    }

    // update cache
    cachedLists[idx] = cell->objects;

    // update proxy info, increase counts
    proxy->cellIndex = idx;
    proxy->cell = cell;
}


template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
template<typename RemapperFunctor>
NEIGHAND_INLINE ObjectProxy<UserObject>* NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::insert(FloatType x, FloatType y, FloatType z, UserObject* object, RemapperFunctor remapperFunctor)
{
    // as in std::vector, when reaches capacity must reallocate new space
    // as in std::vector, use twice mem so global behavior is logarithmic in num alloc
    if (numProxies == proxyCapacity) {
        proxyCapacity += proxyCapacity;
        ObjectProxy<UserObject>* newAllProxies = Allocator::allocate(proxyCapacity);
        memcpy(newAllProxies, allProxies, numProxies*sizeof(ObjectProxy<UserObject>));
        Allocator::deallocate(allProxies, numProxies);
        // remap user objects to new proxies location
        for (unsigned int i=0; i<numProxies; ++i) {
            remapperFunctor(newAllProxies[i].object, &newAllProxies[i]);
            // translate pointer addresses
            if (newAllProxies[i].next) newAllProxies[i].next += newAllProxies - allProxies;
            if (newAllProxies[i].prev) newAllProxies[i].prev += newAllProxies - allProxies;
            // also translate cell start of list if needed!
            else {
                newAllProxies[i].cell->objects = &newAllProxies[i];
                cachedLists[newAllProxies[i].cellIndex] = &newAllProxies[i];
            }
        }
        allProxies = newAllProxies;
    }

    ObjectProxy<UserObject>* proxy = &allProxies[numProxies++];
    proxy->object = object;

    // call internal routine
    // Find cell for that x/y/z position & call internal insert
    insertNoAllocate(proxy, x, y, z, helper.getCellIndexForWorldPosition(x, y, z) );

    // allow for multiple fast insert/remove, and rebuild the table only once later on if needed by Auto
    updateWeightBaseTablesNeeded = true;

    return proxy;
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::removeNoDealloc(ObjectProxy<UserObject>* proxy)
{
    CellEntry<UserObject>* cell = proxy->cell;

    if (proxy->prev) proxy->prev->next = proxy->next;
    if (proxy->next) proxy->next->prev = proxy->prev;

    if (cell->objects == proxy) {
        cell->objects = proxy->next;
        cachedLists[proxy->cellIndex] = proxy->next;
        if (proxy->next) proxy->next->prev = 0;

        // Is this now an empty cell ? => Maintain non-empty list if needed
        if ((!proxy->next) && WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isInside(proxy->cellIndex)) {
            if (totalNonEmpty>1) {
                if (proxy->cell <= firstNonEmptyCell) {
                    firstNonEmptyCell = cell->nextNonEmpty;
                } else {
                    // prev will find first object in worse case, hence cannot go <0
                    CellEntry<UserObject>* prev = cell-1; while (prev->objects==0) --prev;
                    prev->nextNonEmpty = cell->nextNonEmpty;
                }
            } else firstNonEmptyCell = 0;
            --totalNonEmpty;
        }
    }

}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
template<typename RemapperFunctor>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::remove(ObjectProxy<UserObject>* proxy, RemapperFunctor remapperFunctor)
{
    // internal remove, common with update
    removeNoDealloc(proxy);
    // and now release the proxy, by moving the end of array one to this location and decreasing array size
    // Note: assumption is that proxy is valid, hence array size >0
    *proxy = allProxies[--numProxies];
    // job done when there is no more object/proxies in the database
    if (numProxies==0) return;

    // the new object at that position is mapped to its new proxy
    remapperFunctor(proxy->object, proxy);

    // maintain prev/next pointing TO the new location. Thanks double-linked lists.
    if (proxy->next) proxy->next->prev = proxy;
    if (proxy->prev) proxy->prev->next = proxy;
    // maintain start of list if needed
    else {
        proxy->cell->objects = proxy;
        cachedLists[proxy->cellIndex] = proxy;
    }

    // allow for multiple fast insert/remove, and rebuild the table only once later on if needed by Auto
    updateWeightBaseTablesNeeded = true;
}


template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::update(ObjectProxy<UserObject>* proxy, FloatType x, FloatType y, FloatType z)
{

    uint32_t idx = helper.getCellIndexForWorldPosition(x, y, z);

    // if this is still the same cell index, update x/y/z and done!
    if (idx==proxy->cellIndex) {
        proxy->x = x;
        proxy->y = y;
        proxy->z = z;
        return;
    }

    // Otherwise, need to change cell.

    // first remove from current cell
    removeNoDealloc(proxy);

    // Now insert in new cell
    insertNoAllocate(proxy, x, y, z, idx);
}

// Main query routine: apply functor to all neighbors
// Use separate logical file for maintenance
#include "neighand_apply.hpp"

// Again, using x/y/z API
#define NEIGHAND_APPLY_XYZ_API 1
#include "neighand_apply.hpp"
#undef NEIGHAND_APPLY_XYZ_API

// Again, using C callback API
#define NEIGHAND_C_CALLBACK_API 1
#include "neighand_apply.hpp"

#define NEIGHAND_APPLY_XYZ_API 1
#include "neighand_apply.hpp"
#undef NEIGHAND_APPLY_XYZ_API

#undef NEIGHAND_C_CALLBACK_API


#include "neighand_apply_all.hpp"

#define NEIGHAND_C_CALLBACK_API 1
#include "neighand_apply_all.hpp"
#undef NEIGHAND_C_CALLBACK_API

namespace detail {

// Functor to build neighbor lists, without an object to avoid
template <typename UserObject>
struct ListBuilderNoAvoid {
    std::vector<UserObject*>& neighbors;
    NEIGHAND_INLINE ListBuilderNoAvoid(std::vector<UserObject*>& _neighbors) : neighbors(_neighbors) {}
    NEIGHAND_INLINE void operator()(UserObject* object) NEIGHAND_ALWAYS_INLINE {
        neighbors.push_back(object);
    }
};

// Functor to build neighbor lists, with an object to avoid
template <typename UserObject>
struct ListBuilderAvoid : public ListBuilderNoAvoid<UserObject> {
    UserObject* avoidObject;
    NEIGHAND_INLINE ListBuilderAvoid(std::vector<UserObject*>& _neighbors, UserObject* o) : ListBuilderNoAvoid<UserObject>(_neighbors), avoidObject(o) {}
    NEIGHAND_INLINE void operator()(UserObject* object) NEIGHAND_ALWAYS_INLINE {
        if (object==avoidObject) return;
        ListBuilderNoAvoid<UserObject>::operator()(object);
    }
};

}


template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNeighbors(ObjectProxy<UserObject>* p, FloatType d, std::vector<UserObject*>& neighbors)
{
    applyToNeighbors(p,d, detail::ListBuilderAvoid<UserObject>(neighbors, p->object));
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, std::vector<UserObject*>& neighbors)
{
    applyToNeighbors(x,y,z,d, detail::ListBuilderNoAvoid<UserObject>(neighbors));
}

#include "neighand_closest.hpp"

// Find only THE closest neighbor
// This function is close enough to "find the N nearest" so as to justify
//   a common source code for maintenance & no copy/paste error risk
// This function is different enough to justify a separate routine for performance reasons
// See the included file
#define NEIGHAND_CLOSEST_N_EQ_1 1
#include "neighand_closest.hpp"
#undef NEIGHAND_CLOSEST_N_EQ_1


// Again using x/y/z API
#define NEIGHAND_APPLY_XYZ_API 1
#include "neighand_closest.hpp"
#define NEIGHAND_CLOSEST_N_EQ_1 1
#include "neighand_closest.hpp"
#undef NEIGHAND_CLOSEST_N_EQ_1
#undef NEIGHAND_APPLY_XYZ_API

// Again, using C callback API
#define NEIGHAND_C_CALLBACK_API 1
#include "neighand_closest.hpp"
#define NEIGHAND_APPLY_XYZ_API 1
#include "neighand_closest.hpp"
#undef NEIGHAND_APPLY_XYZ_API
#undef NEIGHAND_C_CALLBACK_API


// See neighand_apply for the usage of the tables

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::setWeightSphere(FloatType weight) {
    weightSphere = weight;
    if (updateWeightBaseTablesNeeded) updateWeightBaseTables();
    else {
        for (unsigned int d32 = 0; d32<=maxWorldDist32; ++d32)
            weightSphereTable[d32] = weight * weightSphereTableBase[d32];
        for (uint32_t K=1; K<256; ++K)
            weightSphereTableClosest[K] = weightSphere * weightSphereTableClosestBase[K];
    }
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::setWeightCube(FloatType weight) {
    weightCube = weight;
    weightCubeWithLoad = weight * (1.0f + 1.52f * FloatType(numProxies) / FloatType(WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::MaxVolume));
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::setWeightNonEmpty(FloatType weight) {
    weightNonEmpty = weight;
    if (updateWeightBaseTablesNeeded) updateWeightBaseTables();
    else for (unsigned int d32 = 0; d32<=maxWorldDist32; ++d32)
        weightNonEmptyTable[d32] = weight * weightNonEmptyTableBase[d32];
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::setWeightBrute(FloatType weight) {
    weightBrute = weight;
    if (updateWeightBaseTablesNeeded) updateWeightBaseTables();
    else for (unsigned int d32 = 0; d32<=maxWorldDist32; ++d32)
        weightBruteTable[d32] = weight * weightBruteTableBase[d32];
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::setQueryMethod(QueryMethod method) {
    queryMethod = method;
    if (updateWeightBaseTablesNeeded) updateWeightBaseTables();
}



template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE QueryMethod NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::getAutoFactors(FloatType x, FloatType y, FloatType z, FloatType d, FloatType& factorSphere, FloatType& factorCube, FloatType& factorNonEmpty, FloatType& factorBrute) {

    // See neighand_apply.hpp, this is copy/pasted & adapted code

    if (updateWeightBaseTablesNeeded) updateWeightBaseTables();
    FloatType minWeight = weightCubeWithLoad * helper.getInternalParallelepipedVolume(x,y,z,d);
    factorCube = minWeight;
    QueryMethod method = Cube;

    FloatType d_cellSpace = d * helper.cellSizeInv;
    uint32_t volTableEntry = uint32_t(d_cellSpace*32.f);
    if (volTableEntry > maxWorldDist32) volTableEntry = maxWorldDist32;

    if (helper.isInside(x,y,z)) {
        factorSphere = weightSphereTable[volTableEntry];
        if (factorSphere < minWeight) {
            minWeight = factorSphere;
            method = Sphere;
        }
    } else {
        // generate +Inf
        if (numeric_limits<float>::has_infinity)
            factorSphere = numeric_limits<float>::infinity();
        else factorSphere = numeric_limits<float>::max();
    }

    factorNonEmpty = totalNonEmpty + weightNonEmptyTable[volTableEntry];
    if (factorNonEmpty < minWeight) {
        minWeight = factorNonEmpty;
        method = NonEmpty;
    }

    factorBrute = weightBruteTable[volTableEntry];
    if (factorBrute < minWeight) {
        method = Brute;
    }

    return method;
}

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE QueryMethod NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::getAutoFactorsClosest(FloatType x, FloatType y, FloatType z, FloatType d, int N, FloatType& factorSphere, FloatType& factorCube, FloatType& factorNonEmpty, FloatType& factorBrute) {

    if (updateWeightBaseTablesNeeded) updateWeightBaseTables();
    FloatType minWeight = weightCubeWithLoad * helper.getInternalParallelepipedVolume(x,y,z,d);
    factorCube = minWeight;
    QueryMethod method = Cube;

    FloatType d_cellSpace = d * helper.cellSizeInv;
    uint32_t volTableEntry = uint32_t(d_cellSpace*32.f);
    if (volTableEntry > maxWorldDist32) volTableEntry = maxWorldDist32;

    if (helper.isInside(x,y,z)) {
        factorSphere = weightSphereTableClosest[N<256?N:255];
        if (factorSphere < minWeight) {
            minWeight = factorSphere;
            method = Sphere;
        }
    } else {
        // generate +Inf
        if (numeric_limits<float>::has_infinity)
            factorSphere = numeric_limits<float>::infinity();
        else factorSphere = numeric_limits<float>::max();
    }

    factorNonEmpty = totalNonEmpty + weightNonEmptyTable[volTableEntry];
    if (factorNonEmpty < minWeight) {
        minWeight = factorNonEmpty;
        method = NonEmpty;
    }

    factorBrute = weightBruteTable[volTableEntry];
    if (factorBrute < minWeight) {
        method = Brute;
    }

    return method;
}








} // end namespace

// cleanup preprocessor helper
#undef NEIGHAND_TEMPLATE_ARGUMENTS
