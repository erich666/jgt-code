/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines the functions for applying a user-defined
    functor or callback to all objects within a given distance.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

#if defined(NEIGHAND_C_CALLBACK_API)

#ifdef NEIGHAND_APPLY_XYZ_API
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::applyToNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, Callback f, void* userData)
#else
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::applyToNeighbors(ObjectProxy<UserObject>* p, FloatType d, Callback f, void* userData)
#endif

#else

#ifdef NEIGHAND_APPLY_XYZ_API
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
template<typename Functor>
NEIGHAND_INLINE Functor NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::applyToNeighbors(FloatType x, FloatType y, FloatType z, FloatType d, Functor f)
#else
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
template<typename Functor>
NEIGHAND_INLINE Functor NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::applyToNeighbors(ObjectProxy<UserObject>* p, FloatType d, Functor f)
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

    // Switch method right now to minimize cost when not using Auto
    // Auto then jumps to the correct label rather than recursing
    switch(queryMethod) {

        case Auto: {
            // Principle: roughly count the number of memory access and consider the other costs
            // are negligible
            // - 1 access for each cell & array
            // - 1 access per object compared
            // - 1 access per user functor called
            // Then compensate by allowing the user to weight the result to reflect
            // the different processing costs of each method.

            /* Formula:
               Cube: num cells = volume = exact internal parallelepiped info.
                     num objects = V * load ratio on average
                     num functors = about 52% of volume, assuming perfect continuous cube case
                     final: V*(1+1.52*L)
               Sphere: num cells = volume = estimated volume from helper function
                       array indirect = volume again, + about 10 constant mem access
                       objects compared only on edge, unconditional in center
                       functor unconditional in center, not always on edge
                           => use (V * cell load) for total object + functor
                       cells "outside" are loaded but involve no object
                           => subtract (vol outside * cell load)
                       final: V+V+10+V*L-Out*L = V*(2+(1-Out)*L)+10
                Non-Empty: Vne = num Non-Empty cells available : Vne loads
                           cells too far are rejected
                              => 4/3*pi*d^3 volume Vs with objects & functors
                              but no more than total volume Vt
                           density of cells in that sphere = Vne / Vt
                              => min(Vs,Vt) * Vne / Vt * Load for objects and for functors
                              but load = (N/Vne) for these cells.
                           final: Vne + 2* min(Vs,Vt) * N / Vt
                Brute: no cell load
                       objects indiv tested: N objects
                       4/3*pi*d^3 volume range Vs, but not more than Vt
                       object necessarily within non-empty cell => Vne / Vt density in Vs
                       and (N/Vne) load
                       N + min(Vs,Vt) / Vt * N
                       final: N * (1 + min(Vs,Vt) / Vt)

               updateWeightBaseTables() implements these formula.
               Tip: use setQueryMethod(Auto) to check & rebuild tables if needed.
            */
            if (updateWeightBaseTablesNeeded) updateWeightBaseTables();

            FloatType minWeight = weightCubeWithLoad * helper.getInternalParallelepipedVolume(x,y,z,d);

//if (statSphere+statCube+statNonEmpty+statBrute<5) cout << "Cube weight(volume="<<minWeight/weightCubeWithLoad<<"): " << minWeight << endl;
            QueryMethod method = Cube;

            FloatType d_cellSpace = d * helper.cellSizeInv;
            uint32_t volTableEntry = uint32_t(d_cellSpace*32.f);
            if (volTableEntry > maxWorldDist32) volTableEntry = maxWorldDist32;

            // Sphere is a valid choice only when the query center is inside
#ifdef NEIGHAND_APPLY_XYZ_API
            if (helper.isInside(x,y,z)) {
#else
            if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isInside(p->cellIndex)) {
#endif
                FloatType weight = weightSphereTable[volTableEntry];

//if (statSphere+statCube+statNonEmpty+statBrute<5) cout << "Sphere weight: " << weight << endl;
                if (weight < minWeight) {
                    minWeight = weight;
                    method = Sphere;
                }
            }

            // rationale for non-empty: only totalNonEmpty needs to be maintained on move ops, not table
            FloatType weight = totalNonEmpty + weightNonEmptyTable[volTableEntry];
//if (statSphere+statCube+statNonEmpty+statBrute<5) cout << "NE weight(totalNonEmpty="<<totalNonEmpty<<"): " << weight << endl;

            if (weight < minWeight) {
                minWeight = weight;
                method = NonEmpty;
            }

            weight = weightBruteTable[volTableEntry];
//if (statSphere+statCube+statNonEmpty+statBrute<5) cout << "Brute weight: " << weight << endl;
            if (weight < minWeight) {
                goto Brute_NH_Apply_Label;
            }
            if (method == NonEmpty) goto NonEmpty_NH_Apply_Label;
            if (method == Cube) goto Cube_NH_Apply_Label;
            //goto Sphere_NH_Apply_Label;
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

                // See documentation article: select one of the 64 specialized array
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
                    dm1sq = MaxDQ;
                    shaving = 0;
                }

                uint32_t* baseMaxUp = &sphereOffsets[baseDistanceArray[dmaxSq_m1]];

                uint32_t* interOffsetList = &shavingOffsets[shaving][shavingComplement[shaving][dmaxSq_m1]];
                uint32_t* interMaxUp = &shavingOffsets[shaving][shavingComplement[shaving][lastdq]];

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

                uint32_t* baseOffsetList = &sphereOffsets[0];
                if (d_cellSpace > FloatType(1.73205081)) {
                    d_cellSpace -= FloatType(1.73205081);
                    baseOffsetList = &sphereOffsets[uint32_t(d_cellSpace * d_cellSpace)];
                    register uint32_t* offsetlist = &sphereOffsets[0];
                    while (offsetlist < baseOffsetList) {
                        uint32_t offset = *offsetlist++;
                        uint32_t unpackedCell = centerCellIndex + offset;
                        if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isUnpackedOutside(unpackedCell)) continue;
                        register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packwrap(unpackedCell)];
                        #define NEIGHAND_APPLY_CHECK_FIRST
                        #include "neighand_apply_processlines_uncond.hpp"
                        #undef NEIGHAND_APPLY_CHECK_FIRST
                    }
                }

                register uint32_t* offsetlist = baseOffsetList;
                baseOffsetList = &sphereOffsets[baseDistanceArray[dm1sq]];
                while (offsetlist < baseOffsetList) {
                    uint32_t offset = *offsetlist++;
                    uint32_t unpackedCell = centerCellIndex + offset;
                    if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isUnpackedOutside(unpackedCell)) continue;
                    register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packwrap(unpackedCell)];
                    #define NEIGHAND_APPLY_CHECK_FIRST
                    #include "neighand_apply_processlines_cond.hpp"
                    #undef NEIGHAND_APPLY_CHECK_FIRST
                }

                while (offsetlist < baseMaxUp) {
                    uint32_t offset = *offsetlist++;
                    uint32_t unpackedCell = centerCellIndex + offset;
                    if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isUnpackedOutside(unpackedCell)) continue;
                    #include "neighand_apply_checkdist.hpp"
                    register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packwrap(unpackedCell)];
                    #define NEIGHAND_APPLY_CHECK_FIRST
                    #include "neighand_apply_processlines_cond.hpp"
                    #undef NEIGHAND_APPLY_CHECK_FIRST
                }

                offsetlist = interOffsetList;
                while (offsetlist < interMaxUp) {
                    uint32_t offset = *offsetlist++;
                    uint32_t unpackedCell = centerCellIndex + offset;
                    if (WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::isUnpackedOutside(unpackedCell)) continue;
                    #include "neighand_apply_checkdist.hpp"
                    register ObjectProxy<UserObject>* plist = cachedLists[WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::packwrap(unpackedCell)];
                    #define NEIGHAND_APPLY_CHECK_FIRST
                    #include "neighand_apply_processlines_cond.hpp"
                    #undef NEIGHAND_APPLY_CHECK_FIRST
                }

                // Run through the external region list only if necessary
                helper.clearOutsideFlag();
                helper.flagOutside(x,y,z,d);
                if (helper.outsideIsFlagged()) {
                    register ObjectProxy<UserObject>* plist = helper.getExternalRegionObjectList();
                    #include "neighand_apply_processlines_cond.hpp"
                }

#ifdef NEIGHAND_C_CALLBACK_API
                return;
#else
                return f;
#endif
            }
            // fall through to Cube when query center is outside
        }

        Cube_NH_Apply_Label:
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
                        #define NEIGHAND_APPLY_CHECK_FIRST
                        #include "neighand_apply_processlines_cond.hpp"
                        #undef NEIGHAND_APPLY_CHECK_FIRST
                    }
                }
            }

            // Run through the external region list only if necessary
            helper.clearOutsideFlag();
            helper.flagOutside(x,y,z,d);
            if (helper.outsideIsFlagged()) {
                register ObjectProxy<UserObject>* plist = helper.getExternalRegionObjectList();
                #include "neighand_apply_processlines_cond.hpp"
            }

#ifdef NEIGHAND_C_CALLBACK_API
            return;
#else
            return f;
#endif
        }

        NonEmpty_NH_Apply_Label:
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
                #include "neighand_apply_processlines_cond.hpp"
            }

            // Run through the external region list if it is non-empty
            if (helper.outsideIsNonEmpty()) {
                register ObjectProxy<UserObject>* plist = helper.getExternalRegionObjectList();
                #include "neighand_apply_processlines_cond.hpp"
            }

#ifdef NEIGHAND_C_CALLBACK_API
            return;
#else
            return f;
#endif
        }

        Brute_NH_Apply_Label:
        case Brute: {
#ifdef NEIGHAND_SELECT_METHOD_STAT
            ++statBrute;
#endif
            ObjectProxy<UserObject>* endProxy = allProxies + numProxies;
            for (ObjectProxy<UserObject>* proxy = allProxies; proxy<endProxy; ++proxy) {
                if ((helper.squaredDistance(x-proxy->x, y-proxy->y, z-proxy->z) <= dsq)
#ifndef NEIGHAND_APPLY_XYZ_API
                && (proxy->object!=avoidObject)
#endif
                )
#ifdef NEIGHAND_C_CALLBACK_API
                    f(proxy->object, userData);
#else
                    f(proxy->object);
#endif
            }
        }
        // Brute falls through end of switch to end of method
    }

#ifdef NEIGHAND_C_CALLBACK_API
    return;
#else
    return f;
#endif

}
