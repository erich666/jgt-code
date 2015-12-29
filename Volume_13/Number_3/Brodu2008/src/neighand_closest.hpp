/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines the functions for finding only the N
    nearest neighbors, within a given distance.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

#ifdef NEIGHAND_APPLY_XYZ_API

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE int NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNearestNeighbor(FloatType x, FloatType y, FloatType z, FloatType d, NearestNeighbor<UserObject>* neighbor)
#else
#if defined(NEIGHAND_C_CALLBACK_API)
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE int NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNearestNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, NearestNeighbor<UserObject>* neighbor, unsigned int N)
#else
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNearestNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, std::vector<NearestNeighbor<UserObject> >& neighbor, unsigned int N)
#endif
#endif

// else for X/Y/Z API
#else

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE int NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNearestNeighbor(ObjectProxy<UserObject>* p, FloatType d, NearestNeighbor<UserObject>* neighbor)
#else
#if defined(NEIGHAND_C_CALLBACK_API)
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE int NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNearestNeighbors(ObjectProxy<UserObject>* p, FloatType d, NearestNeighbor<UserObject>* neighbor, unsigned int N)
#else
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::findNearestNeighbors(ObjectProxy<UserObject>* p, FloatType d, std::vector<NearestNeighbor<UserObject> >& neighbor, unsigned int N)
#endif
#endif

#endif

{

    // The x,y,z,dsq and avoidObject are common to all cases hence computed before
#ifndef NEIGHAND_APPLY_XYZ_API
    FloatType x = p->x;
    FloatType y = p->y;
    FloatType z = p->z;
    UserObject* avoidObject = p->object;
#endif
    FloatType dsq = d*d;

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
    // Initialize dist with max query distance
    FloatType neighborSquaredDistance = dsq;
    // but keep this flag to remind this is not a real object, just a limit
    UserObject* neighborObject = 0;
#else

#ifndef NEIGHAND_C_CALLBACK_API
    neighbor.resize(N);
#endif

    // number found
    uint_fast32_t nfound = 0;

    // Initialize dist array with max possible non-infinity distance
    for (unsigned int i=0; i<N; ++i) {
        neighbor[i].squaredDistance = numeric_limits<float>::max();
        // neighborsHolder[i].object = 0; // not necessary
    }
#endif



    // Switch method right now to minimize cost when not using Auto
    // Auto then jumps to the correct label rather than recursing
    switch(queryMethod) {

        case Auto: {

            // See neighand_apply for rationale

            FloatType minWeight = weightCubeWithLoad * helper.getInternalParallelepipedVolume(x,y,z,d);
            QueryMethod method = Cube;

            // Sphere is a valid choice only when the query center is inside
#ifdef NEIGHAND_APPLY_XYZ_API
            if (helper.isInside(x,y,z)) {
#else
            if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isInside(p->cellIndex)) {
#endif

                if (numProxies==0) {
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                    return 0;
#else
#ifndef NEIGHAND_C_CALLBACK_API
                    neighbor.resize(0);
                    return;
#endif
                    return 0;
#endif
                }

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                FloatType weight = weightSphereTableClosest[1];
#else
                FloatType weight = weightSphereTableClosest[N<256?N:255];
#endif
                if (weight < minWeight) {
                    minWeight = weight;
                    method = Sphere;
                }
            }

            FloatType d_cellSpace = d * helper.cellSizeInv;
            uint32_t volTableEntry = uint32_t(d_cellSpace*32.f);
            if (volTableEntry > maxWorldDist32) volTableEntry = maxWorldDist32;

            FloatType weight = totalNonEmpty + weightNonEmptyTable[volTableEntry];
            if (weight < minWeight) {
                minWeight = weight;
                method = NonEmpty;
            }

            weight = weightBruteTable[volTableEntry];
            if (weight < minWeight) {
                goto Brute_NH_Closest_Label;
            }
            if (method == NonEmpty) goto NonEmpty_NH_Closest_Label;
            if (method == Cube) goto Cube_NH_Closest_Label;
            //goto Sphere_NH_Closest_Label;
            // switch fall through
        }

        case Sphere: {
            int32_t cellx, celly, cellz;
            FloatType xcenter, ycenter, zcenter;
#ifdef NEIGHAND_APPLY_XYZ_API
            uint_fast32_t centerCellIndexPacked = helper.getCellIndexForWorldPosition(x,y,z, cellx, celly, cellz, xcenter, ycenter, zcenter);
            if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isInside(centerCellIndexPacked)) {
                uint_fast32_t centerCellIndex = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::unpack(centerCellIndexPacked);
#else
            WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packedIndexToXYZ(p->cellIndex, cellx, celly, cellz);
            if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isInside(p->cellIndex)) {
                xcenter = (x - helper.minx) * helper.cellSizeInv;
                ycenter = (y - helper.miny) * helper.cellSizeInv;
                zcenter = (z - helper.minz) * helper.cellSizeInv;
                // Unpacked index for sphere search: this is the center, necessarily inside
                uint_fast32_t centerCellIndex = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::unpack(p->cellIndex);
#endif

#ifdef NEIGHAND_SELECT_METHOD_STAT
                ++statSphere;
#endif

                FloatType d_cellSpace = d * helper.cellSizeInv;

                // flag that is set to true as soon as N neighbors are found
                // At that point, possible closer candidates are necessarily inside the
                // cells at the sphere edge for that distance
                // Cells further away than that cannot contain closer neighbors
                bool NWereFound = false;

                uint32_t dmax = uint32_t(d_cellSpace);
                FloatType dsq_cellSpace = d_cellSpace*d_cellSpace;
                uint32_t lastdq = uint32_t(dsq_cellSpace);
                uint32_t dmax_p1 = dmax + 1;

                int32_t mincellx_unbounded = fastFloorInt(xcenter - d_cellSpace);
                int32_t maxcellx_unbounded = fastFloorInt(xcenter + d_cellSpace);
                int32_t mincelly_unbounded = fastFloorInt(ycenter - d_cellSpace);
                int32_t maxcelly_unbounded = fastFloorInt(ycenter + d_cellSpace);
                int32_t mincellz_unbounded = fastFloorInt(zcenter - d_cellSpace);
                int32_t maxcellz_unbounded = fastFloorInt(zcenter + d_cellSpace);

                uint32_t shaving = ((cellx - mincellx_unbounded - dmax_p1) >> 31)
                | (((maxcellx_unbounded - cellx - dmax_p1) & 0x80000000) >> 30)
                | (((celly - mincelly_unbounded - dmax_p1) & 0x80000000) >> 29)
                | (((maxcelly_unbounded - celly - dmax_p1) & 0x80000000) >> 28)
                | (((cellz - mincellz_unbounded - dmax_p1) & 0x80000000) >> 27)
                | (((maxcellz_unbounded - cellz - dmax_p1) & 0x80000000) >> 26);

                int32_t dmaxSq_m1 = dmax*dmax - 1;

                FloatType dm1 = d_cellSpace - WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::distCompHeuristicConstant();
                int32_t dm1sq = uint32_t(dm1*dm1);
                if (dm1<FloatType(0.f)) dm1sq = -1;

                if (lastdq>MaxDQ) {
                    lastdq = MaxDQ;
                    dmaxSq_m1 = MaxDQ;
                    shaving = 0;
                }

                xcenter -= cellx;
                ycenter -= celly;
                zcenter -= cellz;

                uint32_t dbldmax = dmax_p1 + dmax_p1 + 1;
                uint32_t dbldmax3 = dbldmax+dbldmax+dbldmax;
                // Initialized constructor: all 0 to begin with
                FloatConverter all_cache[dbldmax3];
                // alias to mid-array to allow indexing by negative offsets
                FloatConverter* cachedDeltaX = all_cache + dmax_p1;
                FloatConverter* cachedDeltaY = cachedDeltaX + dbldmax;
                FloatConverter* cachedDeltaZ = cachedDeltaY + dbldmax;

                uint32_t* baseMaxUp = &sphereOffsets[baseDistanceArray[dmaxSq_m1]];
                uint32_t* cmpCellLimit = &sphereOffsets[baseDistanceArray[dm1sq]];

                uint32_t* interOffsetList = &shavingOffsets[shaving][shavingComplement[shaving][dmaxSq_m1]];
                uint32_t* interMaxUp = &shavingOffsets[shaving][shavingComplement[shaving][lastdq]];

                uint32_t* offsetlist = &sphereOffsets[0];
                while (offsetlist < baseMaxUp) {
                    uint32_t offset = *offsetlist++;

                    uint32_t unpackedCell = centerCellIndex + offset;
                    if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isUnpackedOutside(unpackedCell)) continue;
                    if (offsetlist<=cmpCellLimit) {
                        #include "neighand_apply_checkdist.hpp"
                    }

                    register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packwrap(unpackedCell)];

                    #define NEIGHAND_CLOSEST_CHECK_FIRST
                    #include "neighand_closest_processlines.hpp"
                    #undef NEIGHAND_CLOSEST_CHECK_FIRST

                    // N neighbors were found ?
                    // Potentially closer candidates can only be on the edge of the sphere at that distance
                    // Test only valid the first time, or the edge could be pushed up to max!
                    if (NWereFound) continue;
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                    if (neighborObject==0) continue;
#else
                    if (nfound != N) continue;
#endif
                    NWereFound = true;

                    int_fast32_t aboveDQ = uint_fast32_t(
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                        neighborSquaredDistance
#else
                        dsq // was updated
#endif
                    * cellSizeSqInv);

                    // Further neighbors can only be in edge now
                    if (aboveDQ <= dmaxSq_m1) {
                        baseMaxUp = &sphereOffsets[baseDistanceArray[aboveDQ]];
                        interOffsetList = interMaxUp = 0;
                    } else if (uint32_t(aboveDQ) < lastdq) {
                        interMaxUp = &shavingOffsets[shaving][shavingComplement[shaving][aboveDQ]];
                    }
                } // end sphere list run up

                offsetlist = interOffsetList;
                while (offsetlist < interMaxUp) {
                    uint32_t offset = *offsetlist++;

                    uint32_t unpackedCell = centerCellIndex + offset;
                    if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isUnpackedOutside(unpackedCell)) continue;
                    if (offsetlist<=cmpCellLimit) {
                        #include "neighand_apply_checkdist.hpp"
                    }

                    register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packwrap(unpackedCell)];

                    #define NEIGHAND_CLOSEST_CHECK_FIRST
                    #include "neighand_closest_processlines.hpp"
                    #undef NEIGHAND_CLOSEST_CHECK_FIRST

                    if (NWereFound) continue;
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                    if (neighborObject==0) continue;
#else
                    if (nfound != N) continue;
#endif
                    NWereFound = true;
                    uint_fast32_t aboveDQ = uint_fast32_t(
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                        neighborSquaredDistance
#else
                        dsq // was updated
#endif
                    * cellSizeSqInv);

                    // Further neighbors can only be in edge now
                    if (aboveDQ < lastdq) {
                        interMaxUp = &shavingOffsets[shaving][shavingComplement[shaving][aboveDQ]];
                    }
                }

                // Now run through the external region list only if necessary
                // Using furthest found neighbor distance compared to outside distance
                if (helper.dsqImpliesOutside(x,y,z,
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                neighborSquaredDistance
#else
                dsq
#endif
                )) {
                    // Region is not empty at this point
                    register ObjectProxy<UserObject>* plist = helper.getExternalRegionObjectList();
                    #include "neighand_closest_processlines.hpp"
                }

                // may return 0 is there was no neighbor within the required distance
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
                neighbor->object = neighborObject;
                neighbor->squaredDistance = neighborSquaredDistance;
                return neighborObject != 0;
#else
#ifndef NEIGHAND_C_CALLBACK_API
                neighbor.resize(nfound);
                return;
#endif
                return nfound;
#endif
            }
            // fall through to Cube when query center is outside
        }

        Cube_NH_Closest_Label:
        case Cube: {
#ifdef NEIGHAND_SELECT_METHOD_STAT
            ++statCube;
#endif
            int32_t mincellx, mincelly, mincellz, maxcellx, maxcelly, maxcellz;
            helper.getInternalParallelepiped(mincellx, mincelly, mincellz, maxcellx, maxcelly, maxcellz, x, y, z, d);

            for (int_fast32_t cellz = mincellz; cellz <= maxcellz; ++cellz) {
                for (int_fast32_t celly = mincelly; celly <= maxcelly; ++celly) {
                    for (int_fast32_t cellx = mincellx; cellx <= maxcellx; ++cellx) {
                        register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::xyzToPackedIndex(cellx,celly,cellz)];
                        #define NEIGHAND_CLOSEST_CHECK_FIRST
                        #include "neighand_closest_processlines.hpp"
                        #undef NEIGHAND_CLOSEST_CHECK_FIRST
                    }
                }
            }

            helper.clearOutsideFlag();
            helper.flagOutside(x,y,z,d);
            if (helper.outsideIsFlagged()) {
                ObjectProxy<UserObject>* plist = helper.getExternalRegionObjectList();
                #include "neighand_closest_processlines.hpp"
            }

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
            neighbor->object = neighborObject;
            neighbor->squaredDistance = neighborSquaredDistance;
            return neighborObject != 0;
#else
#ifndef NEIGHAND_C_CALLBACK_API
            neighbor.resize(nfound);
            return;
#endif
            return nfound;
#endif
        }

        NonEmpty_NH_Closest_Label:
        case NonEmpty: {
#ifdef NEIGHAND_SELECT_METHOD_STAT
            ++statNonEmpty;
#endif
            // distance squared, cell space
            FloatType dscs = dsq * cellSizeSqInv;
            // x/y/z cell space
            FloatType xcs = (x - helper.minx) * helper.cellSizeInv;
            FloatType ycs = (y - helper.miny) * helper.cellSizeInv;
            FloatType zcs = (z - helper.minz) * helper.cellSizeInv;

            for (CellEntry<UserObject>* entry = firstNonEmptyCell; entry!=0; entry = entry->nextNonEmpty) {
                uint32_t cellIndexPacked = entry - cells;
                // alway same packed index whatever wrapping scheme
                uint32_t cx = cellIndexPacked & ((1 << exp2divx) - 1);
                uint32_t cy = (cellIndexPacked >> exp2divx) & ((1 << exp2divy) - 1);
                uint32_t cz = (cellIndexPacked >> (exp2divx+exp2divy)) & ((1 << exp2divz) - 1);
                FloatType dx = cx - xcs;
                FloatType dxA = helper.squaredDXCellSpace(dx);
                FloatType dxB = helper.squaredDXCellSpace(dx+1.0f);
                if (dxA>dxB) dxA = dxB; if (dxA < 1.0f) dxA=0.0f;
                FloatType dy = cy - ycs;
                FloatType dyA = helper.squaredDYCellSpace(dy);
                FloatType dyB = helper.squaredDYCellSpace(dy+1.0f);
                if (dyA>dyB) dyA = dyB; if (dyA < 1.0f) dyA=0.0f;
                FloatType dz = cz - zcs;
                FloatType dzA = helper.squaredDYCellSpace(dz);
                FloatType dzB = helper.squaredDYCellSpace(dz+1.0f);
                if (dzA>dzB) dzA = dzB; if (dzA < 1.0f) dzA=0.0f;
                // reject cell if it is too far
                if (dxA+dyA+dzA > dscs) continue;

                register ObjectProxy<UserObject>* plist = entry->objects;
                #include "neighand_closest_processlines.hpp"
            }

            // Run through the external region list if it is non-empty
            if (helper.outsideIsNonEmpty()) {
                register ObjectProxy<UserObject>* plist = helper.getExternalRegionObjectList();
                #include "neighand_closest_processlines.hpp"
            }

            // Done! May return 0 if there was no neighbor within the required distance
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
            neighbor->object = neighborObject;
            neighbor->squaredDistance = neighborSquaredDistance;
            return neighborObject != 0;
#else
#ifndef NEIGHAND_C_CALLBACK_API
            neighbor.resize(nfound);
            return;
#endif
            return nfound;
#endif
        }

        Brute_NH_Closest_Label:
        case Brute: {
#ifdef NEIGHAND_SELECT_METHOD_STAT
            ++statBrute;
#endif

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
            ObjectProxy<UserObject>* endProxy = allProxies + numProxies;
            for (ObjectProxy<UserObject>* proxy = allProxies; proxy<endProxy; ++proxy) {
                FloatType dist = helper.squaredDistance(x-proxy->x, y-proxy->y, z-proxy->z);
                if ((dist<=neighborSquaredDistance)
#ifndef NEIGHAND_APPLY_XYZ_API
                && (proxy->object!=avoidObject)
#endif
                ) {
                    neighborSquaredDistance = dist;
                    neighborObject = proxy->object;
                }
            }
#else
            NearestNeighbor<UserObject> currentObject;
            ObjectProxy<UserObject>* endProxy = allProxies + numProxies;
            for (ObjectProxy<UserObject>* proxy = allProxies; proxy<endProxy; ++proxy) {
                currentObject.squaredDistance = helper.squaredDistance(x-proxy->x, y-proxy->y, z-proxy->z);
                if ((currentObject.squaredDistance<=dsq)
#ifndef NEIGHAND_APPLY_XYZ_API
                // object is the caller => rejected
                && (proxy->object!=avoidObject)
#endif
                ){
                    currentObject.object = proxy->object;
                    if (++nfound>N) nfound=N;
                    // bubble up. Complex sorting algo are costly for small N anyway
                    for (uint_fast32_t j=0; j<nfound; ++j) if (currentObject.squaredDistance < neighbor[j].squaredDistance) {
                        NearestNeighbor<UserObject> tmp = neighbor[j];
                        neighbor[j] = currentObject;
                        currentObject = tmp;
                    }
                    // if array is full, use current max distance to limit the search
                    if (nfound==N) dsq = neighbor[N-1].squaredDistance;
                }
            }
#endif
        }

    }

#if defined(NEIGHAND_CLOSEST_N_EQ_1)
    neighbor->object = neighborObject;
    neighbor->squaredDistance = neighborSquaredDistance;
    return neighborObject != 0;
#else
#ifndef NEIGHAND_C_CALLBACK_API
    neighbor.resize(nfound);
    return;
#endif
    return nfound;
#endif

}

