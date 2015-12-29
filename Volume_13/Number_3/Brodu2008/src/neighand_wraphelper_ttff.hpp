/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines a helper object that is used when the region
    of interest is cyclic in X and Y, but not in Z.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

// Specialization: wrap in X/Y, but do not wrap in Z
// This is especially handy for making cyclic terrains, etc.
//
// Packed index structure
// |0..0|E|idx|
// With:
// - E: external bit
// - idx: . either the internal |z|y|x| pattern for inside cells
//        . or the external region number as 1|0|0|0|
// => This gives 2^(exp2divx+exp2divy+exp2divz)+1 entries in the cells array
// Actually, reserve one for a "dummy" external that is always empty for quick rejection => +2 is the true array size
//
// Unpacked index structure:
// |0..0|y|nz-1 times 0|bz|z|ny times 0|x|
// With:
// - bz: bit for z outside flag.
//   The trick is to use 2^(exp2+1) complement => this bit is set both for <0 and >=2^exp, exactly what we want
//   Since the offsets are added from a position inside the main region, there cannot be overflow

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, class _Allocator>
struct WrapHelper<UserObject, exp2divx, exp2divy, exp2divz, true, true, false, false, _Allocator> {


// See general code comments in ffff


COMPILE_TIME_ASSERT(exp2divx*2+exp2divy*2+exp2divz<=29)

    enum {
        MaxDQ = ((1 << (exp2divx-1))-1) * ((1 << (exp2divx-1))-1)
              + ((1 << (exp2divy-1))-1) * ((1 << (exp2divy-1))-1)
              + ( (exp2divz>1) ? (((1 << exp2divz) - 2) * ((1 << exp2divz) - 2)) : 0 ),
        MaxVolume = (1<<exp2divx)*(1<<exp2divy)*(1<<exp2divz),
        ArraySize = (1 << (exp2divx+exp2divy+exp2divz)) + 2,
        OutsideIndex = 1 << (exp2divx + exp2divy + exp2divz),
        OutsideIndexReal = (1 << (exp2divx + exp2divy + exp2divz))+1,
        WMaskX = ((1 << exp2divx) - 1),
        WMaskY = ((1 << exp2divy) - 1) << exp2divx,
        ShiftY = exp2divx*2 + exp2divy + exp2divz,
        ShiftZ = exp2divx + exp2divy,
        CorrConst = (1 << (exp2divx + exp2divy + exp2divz)) | (1 << exp2divx),
        MaxEdges = ((1 << exp2divx) - 1) | (((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz)) | (((1 << exp2divz) - 1) << (exp2divx+exp2divy)),
        UnpackedXMask = (1 << exp2divx) - 1,
        UnpackedYMask = ((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz),
        UnpackedZMask = ((1 << exp2divz) - 1) << (exp2divx+exp2divy),
        XMask = (1 << exp2divx) - 1,
        YMask = ((1 << exp2divy) - 1) << exp2divx,
        ZMask = ((1 << exp2divz) - 1) << (exp2divx+exp2divy),
        UnpackMask = ((1 << exp2divy) - 1) << exp2divx,
        UnpackMaskInv = ~(((1 << exp2divy) - 1) << exp2divx),
        ShiftInsert = exp2divx + exp2divy + exp2divz,
        WrapMask = ((1 << exp2divx) - 1) | (((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz)) | (((1 << exp2divz) - 1) << (exp2divx+exp2divy)),
        OutPackMask = ((1 << exp2divz) << (exp2divx+exp2divy)),
        ZTo1OutShift = exp2divx+exp2divy+exp2divz,
        PackMask = (1 << (exp2divx + exp2divy + exp2divz)) - 1,
        EXMask = (1 << exp2divx) - 1,
        EYMask = (1 << exp2divy) - 1,
        EZMask = (1 << (exp2divz+1)) - 1
    };

    uint32_t outsideFlag;
    CellEntry<UserObject>* outsideCell;

    FloatType minx, miny, minz, maxx, maxy, maxz, cellSize, cellSizeInv, cellSizeInvDown, cellSizeSquared;

    WrapHelper(FloatType _minx, FloatType _miny, FloatType _minz, FloatType _cellSize) :
        outsideFlag(0),
        minx(_minx), miny(_miny), minz(_minz),
        maxx(_minx + _cellSize * (1<<exp2divx)),
        maxy(_miny + _cellSize * (1<<exp2divy)),
        maxz(_minz + _cellSize * (1<<exp2divz)),
        cellSize(_cellSize), cellSizeSquared(_cellSize*_cellSize)
    {
        cellSizeInv = FloatType(1.0f) / _cellSize;
        int roundmode = fegetround();
        fesetround(FE_TOWARDZERO);
        cellSizeInvDown = FloatType(1.0f) / _cellSize;
        fesetround(roundmode);
    }

    NEIGHAND_INLINE uint_fast32_t getCellIndexForWorldPosition(FloatType dx, FloatType dy, FloatConverter dz) NEIGHAND_ALWAYS_INLINE {

        dx = (dx - minx) * cellSizeInv;
        dy = (dy - miny) * cellSizeInv;

        FloatConverter maxTestz(maxz - dz.f);
        dz.f -= minz;

        // See comment in ffff version
        if ((dz.i | (maxTestz.i-1)) > 0x7FFFFFFF)
            return OutsideIndexReal;

        uint32_t idx = fastFloorInt(dx) & WMaskX;
        idx |= (fastFloorInt(dy) << exp2divx) & WMaskY;
        // z necessary inside now
        return idx | (uint32_t(dz.f * cellSizeInvDown) << ShiftZ);
    }

    NEIGHAND_INLINE uint_fast32_t getCellIndexForWorldPosition(FloatType x, FloatType y, FloatConverter z, int32_t& cellx, int32_t& celly, int32_t& cellz, FloatType &xcenter, FloatType &ycenter, FloatType &zcenter) NEIGHAND_ALWAYS_INLINE {
        xcenter = (x - minx) * cellSizeInv;
        ycenter = (y - miny) * cellSizeInv;
        cellx = fastFloorInt(xcenter);
        celly = fastFloorInt(ycenter);

        FloatConverter maxTestz(maxz - z.f);
        z.f -= minz;

        zcenter = z.f * cellSizeInvDown;
        cellz = int32_t(zcenter);

        if ((z.i | (maxTestz.i-1)) > 0x7FFFFFFF)
            return OutsideIndexReal;

        return (cellx & WMaskX) | ((celly << exp2divx) & WMaskY) | (cellz << ShiftZ);
    }


    // wrap in X/Y, not Z
    NEIGHAND_INLINE FloatType squaredDistance(FloatType dx, FloatType dy, FloatType dz) NEIGHAND_ALWAYS_INLINE {
        dx = fastExp2Rem<exp2divx>(dx * cellSizeInv);
        dy = fastExp2Rem<exp2divy>(dy * cellSizeInv);
        return (dx*dx+dy*dy) * cellSizeSquared + dz*dz;
    }

    NEIGHAND_INLINE FloatType squaredDXCellSpace(FloatType dx) NEIGHAND_ALWAYS_INLINE {
        dx = fastExp2Rem<exp2divx>(dx);
        return dx*dx;
    }

    NEIGHAND_INLINE FloatType squaredDYCellSpace(FloatType dy) NEIGHAND_ALWAYS_INLINE {
        dy = fastExp2Rem<exp2divy>(dy);
        return dy*dy;
    }

    NEIGHAND_INLINE FloatType squaredDZCellSpace(FloatType dz) NEIGHAND_ALWAYS_INLINE {
        return dz*dz;
    }


    // See neighand_apply.hpp, this computes 1-out
    NEIGHAND_INLINE FloatType getSphereWeightingLoadFactor(uint32_t d32) NEIGHAND_ALWAYS_INLINE {
        // See fff file for comments
        // Here alpha = 1/Z
        enum {Z = (1<<exp2divz)};
        FloatType alpha_d = d32 * 0.03125f / FloatType(Z);
        return 1.0f - alpha_d*alpha_d * 0.25f * (3.0f-alpha_d);
    }


    NEIGHAND_INLINE void setOutsideCell(CellEntry<UserObject>* _outsideCell) NEIGHAND_ALWAYS_INLINE {
        outsideCell = _outsideCell;
    }

    NEIGHAND_INLINE void clearOutsideFlag() NEIGHAND_ALWAYS_INLINE {
        outsideFlag = 0;
    }

    NEIGHAND_INLINE uint32_t isInside(FloatType x, FloatType y, FloatConverter z) NEIGHAND_ALWAYS_INLINE {
        FloatConverter maxTestz(maxz - z.f);
        z.f -= minz;

        // use float sign bit.
        return 1 - (uint32_t(z.i | (maxTestz.i-1)) >> 31);
    }

    // xyz are in world coordinates, inside, not in cell units
    NEIGHAND_INLINE void flagOutside(FloatType x, FloatType y, FloatConverter z, FloatType d) NEIGHAND_ALWAYS_INLINE {

        // center+radius used to check if sphere intersects one of the 2 Z planes
        FloatConverter maxTestz(maxz - (z.f+d));
        z.f -= minz + d;

        outsideFlag = (outsideCell->objects!=0);
        outsideFlag &= uint32_t(z.i | (maxTestz.i-1)) >> 31;
    }

    NEIGHAND_INLINE bool dsqImpliesOutside(FloatType x, FloatType y, FloatType z, FloatType dsq) NEIGHAND_ALWAYS_INLINE {

        if (!outsideCell->objects) return false;

        FloatConverter d(maxz - z); d.f *= d.f; d.f -= dsq;
        FloatConverter d2(z - minz); d2.f *= d2.f; d2.f -= dsq;

        return (d.i | d2.i) >> 31;
    }

    NEIGHAND_INLINE void flagOutside() NEIGHAND_ALWAYS_INLINE {
        outsideFlag = (outsideCell->objects!=0);
    }

    NEIGHAND_INLINE bool outsideIsFlagged() NEIGHAND_ALWAYS_INLINE {
        return outsideFlag!=0;
    }

    NEIGHAND_INLINE bool outsideIsNonEmpty() NEIGHAND_ALWAYS_INLINE {
        return outsideCell->objects != 0;
    }

    NEIGHAND_INLINE ObjectProxy<UserObject>* getExternalRegionObjectList() NEIGHAND_ALWAYS_INLINE {
        return outsideCell->objects;
    }

    // center position in world coordinates, return the inside parallelepiped cells
    // Unlike ffff case, there could be several "blocks" or sub-regions for the parallelepiped
    // on the different sides for x/y, if the parallelepiped is over the edge.
    // => choose to use negative (or >max) index so there is only one region, and then xyzToPackedIndex will handle the situation
    NEIGHAND_INLINE void getInternalParallelepiped(int32_t &mincellx, int32_t &mincelly, int32_t &mincellz, int32_t &maxcellx, int32_t &maxcelly, int32_t &maxcellz, FloatType cx, FloatType cy, FloatType cz, FloatType d) NEIGHAND_ALWAYS_INLINE {

        // a,b,c cube englobing the query sphere
        FloatConverter minc(cz - d);
        FloatConverter maxc(cz + d);

        // compare with main region bound
        FloatConverter c1(maxz - minc.f);
        maxc.f -= minz;

        // Test if any one of the above main region bound is negative => sphere is completely outside
        // It's better to add one test here than have a null condition on y or z in the main for loops
        // => report the condition on x, so the outermost loop is not entered
        if ((maxc.i | (c1.i-1)) > 0x7FFFFFFF) {
            // ensure mincellz > maxcellz for the loop to fail
            mincellx = 1; maxcellx = 0;
            mincelly = 1; maxcelly = 0;
            mincellz = 1; maxcellz = 0;
            return;
        }

        FloatConverter mina((cx - d - minx) * cellSizeInv);
        FloatConverter maxa((cx + d - minx) * cellSizeInv);
        FloatConverter minb((cy - d - miny) * cellSizeInv);
        FloatConverter maxb((cy + d - miny) * cellSizeInv);

        // x/y are directly mapped and MUST be negative if need be for looping around parallelepiped cells
        // xyzToPackedIndex will handle the situation later on
        mincellx = int32_t(mina.f) - uint32_t(mina.i >> 31);
        mincelly = int32_t(minb.f) - uint32_t(minb.i >> 31);
        maxcellx = int32_t(maxa.f) - uint32_t(maxa.i >> 31);
        maxcelly = int32_t(maxb.f) - uint32_t(maxb.i >> 31);
        // whole world detection on each dimension
        if (maxcellx - mincellx >= (1<<exp2divx)) { mincellx = 0; maxcellx = (1<<exp2divx)-1;}
        if (maxcelly - mincelly >= (1<<exp2divy)) { mincelly = 0; maxcelly = (1<<exp2divy)-1;}

        // finish the convertion from world coordinates to cell space
        minc.f -= minz;
        // Force signed cast from float, then assigned to unsigned int
        mincellz = static_cast<int32_t>(minc.f * cellSizeInv);
        maxcellz = static_cast<int32_t>(maxc.f * cellSizeInv);

        // Intersect by saturating z min/max to the main region bound. See ffff code
        mincellz &= (uint32_t(mincellz) >> 31) - 1;
        maxcellz -= (1 << exp2divz) - 1;
        maxcellz &= -(uint32_t(maxcellz) >> 31);
        maxcellz += (1 << exp2divz) - 1;
    }

    NEIGHAND_INLINE uint_fast32_t getInternalParallelepipedVolume(FloatType cx, FloatType cy, FloatType cz, FloatType d) NEIGHAND_ALWAYS_INLINE {

        // a,b,c cube englobing the query sphere
        FloatConverter minc(cz - d);
        FloatConverter maxc(cz + d);

        // compare with main region bound
        FloatConverter c1(maxz - minc.f);
        maxc.f -= minz;

        // Test if any one of the above main region bound is negative => sphere is completely outside
        // It's better to add one test here than have a null condition on y or z in the main for loops
        // => report the condition on x, so the outermost loop is not entered
        if ((maxc.i | (c1.i-1)) > 0x7FFFFFFF) {
            return 0;
        }

        FloatConverter mina((cx - d - minx) * cellSizeInv);
        FloatConverter maxa((cx + d - minx) * cellSizeInv);
        FloatConverter minb((cy - d - miny) * cellSizeInv);
        FloatConverter maxb((cy + d - miny) * cellSizeInv);

        int32_t mincellx = int32_t(mina.f) - uint32_t(mina.i >> 31);
        int32_t mincelly = int32_t(minb.f) - uint32_t(minb.i >> 31);
        int32_t maxcellx = int32_t(maxa.f) - uint32_t(maxa.i >> 31);
        int32_t maxcelly = int32_t(maxb.f) - uint32_t(maxb.i >> 31);
        // whole world detection on each dimension
        if (maxcellx - mincellx >= (1<<exp2divx)) { mincellx = 0; maxcellx = (1<<exp2divx)-1;}
        if (maxcelly - mincelly >= (1<<exp2divy)) { mincelly = 0; maxcelly = (1<<exp2divy)-1;}

        // finish the convertion from world coordinates to cell space
        minc.f -= minz;
        // Force signed cast from float, then assigned to unsigned int
        int32_t mincellz = static_cast<int32_t>(minc.f * cellSizeInv);
        int32_t maxcellz = static_cast<int32_t>(maxc.f * cellSizeInv);

        // Intersect by saturating z min/max to the main region bound. See ffff code
        mincellz &= (uint32_t(mincellz) >> 31) - 1;
        maxcellz -= ((1 << exp2divz) - 1);
        maxcellz &= -(uint32_t(maxcellz) >> 31);
        maxcellz += (1 << exp2divz);

        // maxcell - mincell + 1 in each dim
        return (maxcellx - mincellx + 1) * (maxcelly - mincelly + 1) * (maxcellz - mincellz);
    }

// Static functions below

    static NEIGHAND_INLINE bool isInside(uint_fast32_t packedIndex) NEIGHAND_ALWAYS_INLINE {
        return packedIndex < OutsideIndex;
    }

    static NEIGHAND_INLINE int isUnpackedOutside(uint_fast32_t unpackedIndex) NEIGHAND_ALWAYS_INLINE {
        return unpackedIndex & OutPackMask;
    }

    static NEIGHAND_INLINE uint_fast32_t unpack(uint_fast32_t idx) NEIGHAND_ALWAYS_INLINE {
        return ((idx & UnpackMask) << ShiftInsert) | (idx & UnpackMaskInv);
    }

    // This function may not return the real outside cell, only the dummy index one (always empty)
    static NEIGHAND_INLINE uint_fast32_t packwrap(uint_fast32_t idx) NEIGHAND_ALWAYS_INLINE {
        idx &= WrapMask;
        return ((idx >> ShiftInsert) | idx) & PackMask;
    }

    // This function does NOT always receive valid arguments in the ttff case, masking is necessary for x/y, z is OK
    static NEIGHAND_INLINE uint_fast32_t xyzToPackedIndex(int_fast32_t cellx, int_fast32_t celly, int_fast32_t cellz) NEIGHAND_ALWAYS_INLINE {
        return (cellx & EXMask) | ((celly & EYMask) << exp2divx) | (cellz << ShiftZ);
    }

    static NEIGHAND_INLINE void packedIndexToXYZ(uint32_t packedCellIndex, int32_t& cellx, int32_t& celly, int32_t& cellz) NEIGHAND_ALWAYS_INLINE {
        cellx = packedCellIndex & XMask;
        celly = (packedCellIndex & YMask) >> exp2divx;
        cellz = (packedCellIndex & ZMask) >> ShiftZ;
    }

    static NEIGHAND_INLINE int32_t extractX(uint32_t offset) NEIGHAND_ALWAYS_INLINE {
        int32_t ret = offset & EXMask;
        ret |= -(ret & (1<< (exp2divx-1))); // extend bit sign
        return ret;
    }
    static NEIGHAND_INLINE int32_t extractY(uint32_t offset) NEIGHAND_ALWAYS_INLINE {
        int32_t ret = (offset >> ShiftY) & EYMask;
        ret |= -(ret & (1<< (exp2divy-1))); // extend bit sign
        return ret;
    }
    static NEIGHAND_INLINE int32_t extractZ(uint32_t offset) NEIGHAND_ALWAYS_INLINE {
        int32_t ret = (offset >> ShiftZ) & EZMask;
        ret |= -(ret & (1<< exp2divz)); // extend bit sign
        return ret;
    }

    static NEIGHAND_INLINE FloatType wrapCellDeltaX(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return fastExp2Rem<exp2divx>(delta);
    }
    static NEIGHAND_INLINE FloatType wrapCellDeltaY(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return fastExp2Rem<exp2divy>(delta);
    }
    static NEIGHAND_INLINE FloatType wrapCellDeltaZ(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return delta;
    }

    static NEIGHAND_INLINE FloatType distCompHeuristicConstant() NEIGHAND_ALWAYS_INLINE {
        return FloatType(1.00001f);
    }



    // take two cells at unpacked index idx1 and idx2, both inside
    // return the minimum squared distance any point in these cells can have, in cell units
    static NEIGHAND_INLINE uint_fast32_t getWrappedDeltaUnpackedIndexDQ(uint_fast32_t idx1, uint_fast32_t idx2) NEIGHAND_ALWAYS_INLINE {
        // 1. Extract X/Y/Z
        uint32_t dx = (idx1 & UnpackedXMask) - (idx2 & UnpackedXMask);
        uint32_t dy = ((idx1 & UnpackedYMask) >> ShiftY) - ((idx2 & UnpackedYMask) >> ShiftY);
        uint32_t dz = ((idx1 & UnpackedZMask) >> ShiftZ) - ((idx2 & UnpackedZMask) >> ShiftZ);
        // 2. bit signs, on exp2 bits for X/Y
        uint32_t bsx = (dx >> (exp2divx-1)) & 1;
        uint32_t bsy = (dy >> (exp2divy-1)) & 1;
        uint32_t bsz = dz >> 31;
        // 3. compute abs(xyz) - 1, expand on 32 bits for X/Y
        dx = (((dx ^ (-bsx)) + bsx) & ((1 << exp2divx)-1) ) - 1;
        dy = (((dy ^ (-bsy)) + bsy) & ((1 << exp2divy)-1) ) - 1;
        dz = (dz ^ (-bsz)) + bsz - 1;
        // compute (|xyz|-1)^2 if |xyz|>1, or 0
        // Note: 0xFFFFFFFF squared is 1 even in unsigned arithmetic
        //       (2^32-1)*(2^32-1) = 2^64 + 1 - 2*2^32 => only 1 remains on a 32 bits variable
        dx = dx * dx - (dx >> 31);
        dy = dy * dy - (dy >> 31);
        dz = dz * dz - (dz >> 31);
        // return min dist squared from all 64 combinations of cube summits
        return dx+dy+dz;
    }

    enum {
        ShaveXN = 1,
        ShaveXP = 2,
        ShaveYN = 4,
        ShaveYP = 8,
        ShaveZN = 16,
        ShaveZP = 32
    };

    typedef typename _Allocator::template rebind<uint32_t >::other UINT32Allocator;

    // The sphere offsets pre-computation depends on wrapping
    static NEIGHAND_INLINE void buildDistanceArray(uint32_t* &baseDistanceArray, uint32_t* &sphereOffsets, uint32_t** shavingComplement, uint32_t** shavingOffsets, uint32_t maxWorldDist32, FloatType* distanceVolumeTable, const char* initFileName, UINT32Allocator allocator) NEIGHAND_ALWAYS_INLINE {

        // N - version/exp2divx/y/z on 4 bits each = 16 bits - ttff in binary = 1100
        uint32_t version = 1; // binary file format version, not software release version
        uint32_t magic = 'N' | ( ((version<<12)|((exp2divx<<8)|(exp2divy<<4)|exp2divz))<< 8) | ( 0xC << 24);

        // try loading the init file first
        std::ifstream initFile;
        initFile.open(initFileName, std::ios::binary);
        if (initFile.good()) {

            uint32_t counts;
            // read a magic marker: don't allocate mem otherwise
            initFile.read(reinterpret_cast<char*>(&counts), sizeof(uint32_t));
            if (counts == magic) {

                initFile.read(reinterpret_cast<char*>(&maxWorldDist32), sizeof(uint32_t));
                distanceVolumeTable = new FloatType[maxWorldDist32+1];
                initFile.read(reinterpret_cast<char*>(distanceVolumeTable), (maxWorldDist32+1)*sizeof(FloatType));

                initFile.read(reinterpret_cast<char*>(&counts), sizeof(uint32_t));
                sphereOffsets = allocator.allocate(counts+1);
                *sphereOffsets++ = counts+1;
                initFile.read(reinterpret_cast<char*>(sphereOffsets), counts*sizeof(uint32_t));

                baseDistanceArray = allocator.allocate(MaxDQ+2);
                initFile.read(reinterpret_cast<char*>(baseDistanceArray), (MaxDQ+2)*sizeof(uint32_t));
                ++baseDistanceArray;

                for (int shaving=0; shaving<64; ++shaving) {
                    initFile.read(reinterpret_cast<char*>(&counts), sizeof(uint32_t));
                    shavingOffsets[shaving] = allocator.allocate(counts+1);
                    *shavingOffsets[shaving]++ = counts+1;
                    initFile.read(reinterpret_cast<char*>(shavingOffsets[shaving]), counts*sizeof(uint32_t));

                    shavingComplement[shaving] = allocator.allocate(MaxDQ+2);
                    initFile.read(reinterpret_cast<char*>(shavingComplement[shaving]), (MaxDQ+2)*sizeof(uint32_t));
                    ++shavingComplement[shaving];
                }

                bool success = initFile.good();
                // read a magic marker again for end of file
                initFile.read(reinterpret_cast<char*>(&counts), sizeof(uint32_t));
                success &= (counts == magic);
                initFile.close();
                if (success) return;

                // Error! deallocate memory and recompute the tables
                for (int shaving = 0; shaving < 64; ++shaving) {
                    allocator.deallocate(shavingComplement[shaving]-1, MaxDQ+2);
                    allocator.deallocate(shavingOffsets[shaving]-1, *(shavingOffsets[shaving]-1));
                }
                allocator.deallocate(baseDistanceArray-1, MaxDQ+2);
                allocator.deallocate(sphereOffsets-1, *(sphereOffsets-1));
            }
        }

        std::vector<uint32_t>* baseOffsets = new std::vector<uint32_t>[MaxDQ+2];
        ++baseOffsets; // so index -1 is valid
        uint32_t counts = 0;
        std::vector<uint32_t>* sOffsets[64];
        uint32_t sCounts[64];
        for (int i=0; i<64; ++i) {
            sOffsets[i] = new std::vector<uint32_t>[MaxDQ+1]; sCounts[i]=0;
        }

        // no wrap case in all 3 dims, offset from (0,0,0) is considered in either one of 8 directions, see below
        for (int z = 0; z<(1 << exp2divz); ++z) {
            for (int y = 0; y<(1 << exp2divy); ++y) {
                for (int x = 0; x<(1 << exp2divx); ++x) {

                    // Offset only for the positive direction, others will be obtained by symmetry
                    uint32_t unpackedIndex = uint32_t(x) | (uint32_t(y) << ShiftY) | (uint32_t(z) << ShiftZ);

                    // Match the 8 summits of the first cube with the 8 of the second
                    // retain min dist
                    uint32_t dq = getWrappedDeltaUnpackedIndexDQ(0,unpackedIndex);

                    int32_t dbase = uint32_t(sqrtf(dq));

                    // extend bit signs
                    int32_t ex = x | -(x & (1<<(exp2divx-1)));
                    int32_t ey = y | -(y & (1<<(exp2divy-1)));
                    FloatType dx = (ex<0) ? (ex+1) : ex;
                    FloatType dy = (ey<0) ? (ey+1) : ey;

                    // handle the 2 z directions, the distances above are preserved by symmetry.
                    int zval[] = {z, -z};
                    // should not duplicate the cells at 0 offset
                    for (int iz = 0; iz <= (z!=0); ++iz) {
                        // Extended x/y/z, possibly negative
                        int ez = zval[iz];
                        FloatType dz = (ez<0) ? (ez+1) : ez;

                        // for each cell, subdivide center in 1000 cubes
                        for (int i=0; i<10; ++i) for (int j=0; j<10; ++j) for (int k=0; k<10; ++k) {
                            FloatType cx = i * 0.1f + 0.05f;
                            FloatType cy = j * 0.1f + 0.05f;
                            FloatType cz = k * 0.1f + 0.05f;

                            // distance to target cell.
                            FloatType d = 0.f;
                            if (ex!=0) {
                                FloatType tmp = dx - cx;
                                d += tmp * tmp;
                            }
                            if (ey!=0) {
                                FloatType tmp = dy - cy;
                                d += tmp * tmp;
                            }
                            if (ez!=0) {
                                FloatType tmp = dz - cz;
                                d += tmp * tmp;
                            }
                            // increment count
                            uint32_t dbin = uint32_t(sqrtf(d) * 32.f);
                            if (dbin>maxWorldDist32) dbin = maxWorldDist32;
                            distanceVolumeTable[dbin] += FloatType(1.0f);
                        }

                        if ((dq==0) && (x==0) && (y==0) && (z==0)) {
                            baseOffsets[-1].push_back(0);
                            ++counts;
                            continue;
                        }

                        uint32_t currentCellOffset = (uint32_t(ex) & EXMask)
                            | ((uint32_t(ey) & EYMask) << ShiftY)
                            | ((uint32_t(ez) & EZMask) << ShiftZ);

                        if ((ex!=0) && (ey!=0)) currentCellOffset |= 0x60000000;
                        if ((ex!=0) && (ez!=0)) currentCellOffset |= 0xA0000000;
                        if ((ey!=0) && (ez!=0)) currentCellOffset |= 0xC0000000;
                        if (x == (1<<(exp2divx-1))) currentCellOffset &= ~0x20000000;
                        if (y == (1<<(exp2divy-1))) currentCellOffset &= ~0x40000000;

                        baseOffsets[dq].push_back(currentCellOffset);
                        ++counts;

                        for (uint_fast32_t shaving = 0; shaving < 64; ++shaving) {

                            // shave off some offsets
                            if ((dq==0) && ((ex!=0) || (ey!=0) || (ez!=0))) {
                                if (
                                ((((ex<0) && (shaving&ShaveXN)) || ((ex>0) && (shaving&ShaveXP))) && (x != (1<<(exp2divx-1))))
                                || ((((ey<0) && (shaving&ShaveYN)) || ((ey>0) && (shaving&ShaveYP))) && (y != (1<<(exp2divy-1))))
                                || ((ez<0) && (shaving&ShaveZN)) || ((ez>0) && (shaving&ShaveZP))
                                ) {
                                    continue;
                                }
                            } else if (dq>0) {
                                if ((((ex<0) && (shaving&ShaveXN)) || ((ex>0) && (shaving&ShaveXP)) || (ex==0) || (ex==-1) || (ex==1))
                                && (((ey<0) && (shaving&ShaveYN)) || ((ey>0) && (shaving&ShaveYP)) || (ey==0) || (ey==-1) || (ey==1))
                                && (((ez<0) && (shaving&ShaveZN)) || ((ez>0) && (shaving&ShaveZP)) || (ez==0) || (ez==-1) || (ez==1))
                                && (x != (1<<(exp2divx-1)))
                                && (y != (1<<(exp2divy-1)))
                                ) {
                                    continue;
                                }
                            }

                            if ((dq>1) && (
                            ((((ex<=-dbase) && (shaving&ShaveXN)) || ((ex>=dbase) && (shaving&ShaveXP))) && (x != (1<<(exp2divx-1))))
                            || ((((ey<=-dbase) && (shaving&ShaveYN)) || ((ey>=dbase) && (shaving&ShaveYP))) && (y != (1<<(exp2divy-1))))
                            || ((ez<=-dbase) && (shaving&ShaveZN)) || ((ez>=dbase) && (shaving&ShaveZP))
                            )) {
                                continue;
                            }

                            sOffsets[shaving][dq].push_back(currentCellOffset);
                            ++sCounts[shaving];
                        }

                    } // end z switcher

                }
            }
        }

        // finish volume distance table computation
        // cumulate to get volume < dist
        for (uint32_t i=1; i<=maxWorldDist32; ++i)
            distanceVolumeTable[i] += distanceVolumeTable[i-1];
        // finally convert sum of elementary cells to volume
        for (uint32_t i=0; i<=maxWorldDist32; ++i)
            distanceVolumeTable[i] = distanceVolumeTable[i] / FloatType(1000.f);

        // Build the base array
        sphereOffsets = allocator.allocate(counts+1);
        *sphereOffsets++ = counts+1;
        baseDistanceArray = allocator.allocate(MaxDQ+2) + 1;
        baseDistanceArray[-1] = -1;
        uint32_t ntotal = 0;

        // linearize the array of vectors, keep the entry positions
        for (int dq = -1; dq <= MaxDQ; ++dq) {
            for (unsigned int i = 0; i<baseOffsets[dq].size(); ++i) {
                sphereOffsets[ntotal++] = baseOffsets[dq][i];
            }
            baseDistanceArray[dq] = ntotal;
        }
        delete [] (baseOffsets - 1);

        // now build the shaving complement arrays
        for (int shaving=0; shaving<64; ++shaving) {
            shavingOffsets[shaving] = allocator.allocate(sCounts[shaving]+1);
            *shavingOffsets[shaving]++ = sCounts[shaving]+1;

            shavingComplement[shaving] = allocator.allocate(MaxDQ+2);
            shavingComplement[shaving][0] = 0; // -1: used as init pos, no dup
            ++shavingComplement[shaving];

            uint32_t sTotal = 0;
            for (int dq = 0; dq <= MaxDQ; ++dq) {
                for (unsigned int i = 0; i<sOffsets[shaving][dq].size(); ++i) {
                    shavingOffsets[shaving][sTotal++] = sOffsets[shaving][dq][i];
                }
                shavingComplement[shaving][dq] = sTotal;
            }
            delete [] sOffsets[shaving];
        }

        // Done. Try saving the file
        std::ofstream outInitFile;
        outInitFile.open(initFileName, std::ios::binary);
        if (outInitFile.good()) {
            // write a magic marker
            outInitFile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
            outInitFile.write(reinterpret_cast<char*>(&maxWorldDist32), sizeof(uint32_t));
            outInitFile.write(reinterpret_cast<char*>(distanceVolumeTable), (maxWorldDist32+1)*sizeof(FloatType));
            outInitFile.write(reinterpret_cast<char*>(&counts), sizeof(uint32_t));
            outInitFile.write(reinterpret_cast<char*>(sphereOffsets), counts*sizeof(uint32_t));
            outInitFile.write(reinterpret_cast<char*>(baseDistanceArray-1), (MaxDQ+2)*sizeof(uint32_t));
            for (int shaving=0; shaving<64; ++shaving) {
                outInitFile.write(reinterpret_cast<char*>(&sCounts[shaving]), sizeof(uint32_t));
                outInitFile.write(reinterpret_cast<char*>(shavingOffsets[shaving]), sCounts[shaving]*sizeof(uint32_t));
                outInitFile.write(reinterpret_cast<char*>(shavingComplement[shaving]-1), (MaxDQ+2)*sizeof(uint32_t));
            }
            // write a magic marker again for end of file
            outInitFile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
            outInitFile.close();
        }
    }


};
