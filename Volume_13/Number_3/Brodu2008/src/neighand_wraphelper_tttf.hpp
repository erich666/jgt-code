/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines a helper object that is used when the region
    of interest is cyclic all X/Y/Z dimensions. There is no exterior
    in that case, the region of interest is the whole world.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

// wrap in all dim
template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, class _Allocator>
struct WrapHelper<UserObject, exp2divx, exp2divy, exp2divz, true, true, true, false, _Allocator> {

// need 3 bits to store the run-time cell checking flags (see apply.hpp)
COMPILE_TIME_ASSERT(exp2divx*2+exp2divy*2+exp2divz<=29)

    enum {
        MaxDQ = ((1 << (exp2divx-1))-1) * ((1 << (exp2divx-1))-1)
              + ((1 << (exp2divy-1))-1) * ((1 << (exp2divy-1))-1)
              + ((1 << (exp2divz-1))-1) * ((1 << (exp2divz-1))-1),
        MaxVolume = (1<<exp2divx)*(1<<exp2divy)*(1<<exp2divz),
        ArraySize = 1 << (exp2divx+exp2divy+exp2divz),
        WrapMask = ((1 << exp2divx) - 1) | (((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz)) | (((1 << exp2divz) - 1) << (exp2divx+exp2divy)),
        PackMask = (1 << (exp2divx + exp2divy + exp2divz)) - 1,
        ShiftInsert = exp2divx + exp2divy + exp2divz,
        ShiftY = exp2divx*2 + exp2divy + exp2divz,
        ShiftZ = exp2divx + exp2divy,
        CorrConst = (1 << (exp2divx + exp2divy+ exp2divz)) | (1 << exp2divx),
        UnpackMask = ((1 << exp2divy) - 1) << exp2divx,
        UnpackMaskInv = ~(((1 << exp2divy) - 1) << exp2divx),
        WMaskX = ((1 << exp2divx) - 1),
        WMaskY = ((1 << exp2divy) - 1) << exp2divx,
        WMaskZ = ((1 << exp2divz) - 1) << (exp2divx + exp2divy),
        UnpackMaskY = ((1 << exp2divy) - 1) << (exp2divx*2 + exp2divy + exp2divz),
        EXMask = (1 << exp2divx) - 1,
        EYMask = (1 << exp2divy) - 1,
        EZMask = (1 << exp2divz) - 1
    };

    FloatType minx, miny, minz, cellSize, cellSizeInv, cellSizeSquared;

    NEIGHAND_INLINE WrapHelper(FloatType _minx, FloatType _miny, FloatType _minz, FloatType _cellSize) : minx(_minx), miny(_miny), minz(_minz), cellSize(_cellSize), cellSizeInv(FloatType(1.0f)/_cellSize), cellSizeSquared(_cellSize*_cellSize) {
    }

    // Wrap version for all X/Y/Z
    NEIGHAND_INLINE uint_fast32_t getCellIndexForWorldPosition(FloatType dx, FloatType dy, FloatType dz) NEIGHAND_ALWAYS_INLINE {
        dx = (dx - minx) * cellSizeInv;
        dy = (dy - miny) * cellSizeInv;
        dz = (dz - minz) * cellSizeInv;

        // floor down the value, then possibly wrap
        // assembly could avoid all this using round down to -infinity mode

        uint32_t idx = fastFloorInt(dx) & WMaskX;

        // process y similarly
        idx |= (fastFloorInt(dy) << exp2divx) & WMaskY;

        // process z similarly
        return idx | ((fastFloorInt(dz) << ShiftZ) & WMaskZ);
    }

    NEIGHAND_INLINE uint_fast32_t getCellIndexForWorldPosition(FloatType dx, FloatType dy, FloatType dz, int32_t& cellx, int32_t& celly, int32_t& cellz, FloatType &xcenter, FloatType &ycenter, FloatType &zcenter) NEIGHAND_ALWAYS_INLINE {
        xcenter = (dx - minx) * cellSizeInv;
        ycenter = (dy - miny) * cellSizeInv;
        zcenter = (dz - minz) * cellSizeInv;
        cellx = fastFloorInt(xcenter);
        celly = fastFloorInt(ycenter);
        cellz = fastFloorInt(zcenter);
        return (cellx & WMaskX) | ((celly << exp2divx) & WMaskY) | ((cellz << ShiftZ) & WMaskZ);
    }

    // dx * dx + dy * dy + dz * dz, but wrapping distances first
    // Note: first convert from world coordinates to cells space, scaling by cellSizeInv
    //       => this allows to use fast Exp2Rem to wrap the delta
    //       Then, scale back by cellSize^2 to restore world coordinates
    NEIGHAND_INLINE FloatType squaredDistance(FloatType dx, FloatType dy, FloatType dz) NEIGHAND_ALWAYS_INLINE {
        dx = fastExp2Rem<exp2divx>(dx * cellSizeInv);
        dy = fastExp2Rem<exp2divy>(dy * cellSizeInv);
        dz = fastExp2Rem<exp2divz>(dz * cellSizeInv);
        return (dx*dx+dy*dy+dz*dz) * cellSizeSquared;
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
        dz = fastExp2Rem<exp2divz>(dz);
        return dz*dz;
    }


    // See neighand_apply.hpp, this computes 1-out. Here no out, so dout=0, so return 1.
    NEIGHAND_INLINE FloatType getSphereWeightingLoadFactor(uint32_t) NEIGHAND_ALWAYS_INLINE {
        return 1.0f;
    }

    // Region support is very simple in all-wrap case
    // inline empty or so functions are hopefully not even generated by the compiler
    NEIGHAND_INLINE void setOutsideCell(CellEntry<UserObject>*) NEIGHAND_ALWAYS_INLINE {
    }
    NEIGHAND_INLINE void clearOutsideFlag() NEIGHAND_ALWAYS_INLINE {
    }
    NEIGHAND_INLINE uint32_t isInside(FloatType x, FloatType y, FloatType z) NEIGHAND_ALWAYS_INLINE {
        return 1;
    }
    NEIGHAND_INLINE void flagOutside(FloatType,FloatType,FloatType,FloatType) NEIGHAND_ALWAYS_INLINE {
    }
    NEIGHAND_INLINE bool dsqImpliesOutside(FloatType x, FloatType y, FloatType z, FloatType dsq) NEIGHAND_ALWAYS_INLINE {
        return false;
    }
    NEIGHAND_INLINE void flagOutside() NEIGHAND_ALWAYS_INLINE {
    }
    NEIGHAND_INLINE bool outsideIsFlagged() NEIGHAND_ALWAYS_INLINE {
        return false;
    }
    NEIGHAND_INLINE bool outsideIsNonEmpty() NEIGHAND_ALWAYS_INLINE {
        return false;
    }
    NEIGHAND_INLINE ObjectProxy<UserObject>* getExternalRegionObjectList() NEIGHAND_ALWAYS_INLINE {
        return 0;
    }


    NEIGHAND_INLINE void getInternalParallelepiped(int32_t &mincellx, int32_t &mincelly, int32_t &mincellz, int32_t &maxcellx, int32_t &maxcelly, int32_t &maxcellz, FloatType cx, FloatType cy, FloatType cz, FloatType d) NEIGHAND_ALWAYS_INLINE {
        FloatConverter mina((cx - d - minx) * cellSizeInv);
        FloatConverter maxa((cx + d - minx) * cellSizeInv);
        FloatConverter minb((cy - d - miny) * cellSizeInv);
        FloatConverter maxb((cy + d - miny) * cellSizeInv);
        FloatConverter minc((cz - d - minz) * cellSizeInv);
        FloatConverter maxc((cz + d - minz) * cellSizeInv);

        // x/y are directly mapped and MUST be negative if need be for looping around parallelepiped cells
        // xyzToPackedIndex will handle the situation later on
        mincellx = int32_t(mina.f) - uint32_t(mina.i >> 31);
        mincelly = int32_t(minb.f) - uint32_t(minb.i >> 31);
        mincellz = int32_t(minc.f) - uint32_t(minc.i >> 31);
        maxcellx = int32_t(maxa.f) - uint32_t(maxa.i >> 31);
        maxcelly = int32_t(maxb.f) - uint32_t(maxb.i >> 31);
        maxcellz = int32_t(maxc.f) - uint32_t(maxc.i >> 31);

        // whole world detection on each dimension
        if (maxcellx - mincellx >= (1<<exp2divx)) { mincellx = 0; maxcellx = (1<<exp2divx)-1;}
        if (maxcelly - mincelly >= (1<<exp2divy)) { mincelly = 0; maxcelly = (1<<exp2divy)-1;}
        if (maxcellz - mincellz >= (1<<exp2divz)) { mincellz = 0; maxcellz = (1<<exp2divz)-1;}
    }

    NEIGHAND_INLINE uint_fast32_t getInternalParallelepipedVolume(FloatType cx, FloatType cy, FloatType cz, FloatType d) NEIGHAND_ALWAYS_INLINE {
        FloatConverter mina((cx - d - minx) * cellSizeInv);
        FloatConverter maxa((cx + d - minx) * cellSizeInv);
        FloatConverter minb((cy - d - miny) * cellSizeInv);
        FloatConverter maxb((cy + d - miny) * cellSizeInv);
        FloatConverter minc((cz - d - minz) * cellSizeInv);
        FloatConverter maxc((cz + d - minz) * cellSizeInv);

        // x/y are directly mapped and MUST be negative if need be for looping around parallelepiped cells
        // xyzToPackedIndex will handle the situation later on
        int32_t mincellx = int32_t(mina.f) - uint32_t(mina.i >> 31);
        int32_t mincelly = int32_t(minb.f) - uint32_t(minb.i >> 31);
        int32_t mincellz = int32_t(minc.f) - uint32_t(minc.i >> 31);
        int32_t maxcellx = int32_t(maxa.f) - uint32_t(maxa.i >> 31);
        int32_t maxcelly = int32_t(maxb.f) - uint32_t(maxb.i >> 31);
        int32_t maxcellz = int32_t(maxc.f) - uint32_t(maxc.i >> 31);

        // whole world detection on each dimension
        if (maxcellx - mincellx >= (1<<exp2divx)) { mincellx = 0; maxcellx = (1<<exp2divx)-1;}
        if (maxcelly - mincelly >= (1<<exp2divy)) { mincelly = 0; maxcelly = (1<<exp2divy)-1;}
        if (maxcellz - mincellz >= (1<<exp2divz)) { mincellz = 0; maxcellz = (1<<exp2divz)-1;}

        return (maxcellx - mincellx + 1) * (maxcelly - mincelly + 1) * (maxcellz - mincellz + 1);
    }


// Static functions below

    static NEIGHAND_INLINE bool isInside(uint_fast32_t) NEIGHAND_ALWAYS_INLINE {
        return true;
    }

    static NEIGHAND_INLINE int isUnpackedOutside(uint_fast32_t unpackedIndex) NEIGHAND_ALWAYS_INLINE {
        return 0;
    }

    static NEIGHAND_INLINE uint_fast32_t unpack(uint_fast32_t idx) NEIGHAND_ALWAYS_INLINE {
        return ((idx & UnpackMask) << ShiftInsert) | (idx & UnpackMaskInv);
    }

    static NEIGHAND_INLINE uint_fast32_t packwrap(uint_fast32_t idx) NEIGHAND_ALWAYS_INLINE {
        // wrap all x/y/z coordinates at the same time
        idx &= WrapMask;
        // pack the index back
        return ((idx >> ShiftInsert) | idx) & PackMask;
    }

    static NEIGHAND_INLINE uint_fast32_t xyzToPackedIndex(int_fast32_t cellx, int_fast32_t celly, int_fast32_t cellz) NEIGHAND_ALWAYS_INLINE {
        return (cellx & EXMask) | ((celly & EYMask) << exp2divx) | ((cellz & EZMask) << ShiftZ);
    }

    static NEIGHAND_INLINE void packedIndexToXYZ(uint32_t packedCellIndex, int32_t& cellx, int32_t& celly, int32_t& cellz) NEIGHAND_ALWAYS_INLINE {
        cellx = packedCellIndex & WMaskX;
        celly = (packedCellIndex & WMaskY) >> exp2divx;
        cellz = (packedCellIndex & WMaskZ) >> ShiftZ;
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
        ret |= -(ret & (1<< (exp2divz-1))); // extend bit sign
        return ret;
    }

    static NEIGHAND_INLINE FloatType wrapCellDeltaX(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return fastExp2Rem<exp2divx>(delta);
    }
    static NEIGHAND_INLINE FloatType wrapCellDeltaY(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return fastExp2Rem<exp2divy>(delta);
    }
    static NEIGHAND_INLINE FloatType wrapCellDeltaZ(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return fastExp2Rem<exp2divz>(delta);
    }

    static NEIGHAND_INLINE FloatType distCompHeuristicConstant() NEIGHAND_ALWAYS_INLINE {
        return FloatType(1.00001f);
    }

    // take two cells at unpacked index idx1 and idx2
    // return the minimum squared distance any point in these cells can have, in cell units
    static NEIGHAND_INLINE uint_fast32_t getWrappedDeltaUnpackedIndexDQ(uint_fast32_t idx1, uint_fast32_t idx2) NEIGHAND_ALWAYS_INLINE {
        // diff
        idx1 = (idx1 - idx2 + CorrConst) & WrapMask;
        // could implement a parallel/SWAR version? not worth it, only used at init.
        uint32_t x = idx1 & WMaskX;
        uint32_t y = (idx1 & UnpackMaskY) >> ShiftY;
        uint32_t z = (idx1 & WMaskZ) >> ShiftZ;
        // bit signs, that's where SWAR would need more template meta-prog for cases exp2divx == exp2divy, etc (5 cases)
        uint32_t bsx = x >> (exp2divx-1);
        uint32_t bsy = y >> (exp2divy-1);
        uint32_t bsz = z >> (exp2divz-1);
        // compute abs(xyz) - 1, abs on exp2 bits, result extended on 32 bits
        x = (((x ^ (-bsx)) + bsx) & ((1 << exp2divx)-1) ) - 1;
        y = (((y ^ (-bsy)) + bsy) & ((1 << exp2divy)-1) ) - 1;
        z = (((z ^ (-bsz)) + bsz) & ((1 << exp2divz)-1) ) - 1;
        // compute (|xyz|-1)^2 if |xyz|>1, or 0
        // Note: 0xFFFFFFFF squared is 1 even in unsigned arithmetic
        //       (2^32-1)*(2^32-1) = 2^64 + 1 - 2*2^32 => only 1 remains on a 32 bits variable
        x = x * x - (x >> 31);
        y = y * y - (y >> 31);
        z = z * z - (z >> 31);
        // return min dist squared from all 64 combinations of cube summits
        return x+y+z;
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

        // N - version/exp2divx/y/z on 4 bits each = 16 bits - tttf in binary = 1110
        uint32_t version = 1; // binary file format version, not software release version
        uint32_t magic = 'N' | ( ((version<<12)|((exp2divx<<8)|(exp2divy<<4)|exp2divz))<< 8) | ( 0xE << 24);

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
                    int32_t ez = z | -(z & (1<<(exp2divz-1)));

                    FloatType dx = (ex<0) ? (ex+1) : ex;
                    FloatType dy = (ey<0) ? (ey+1) : ey;
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

                    uint32_t currentCellOffset = unpackedIndex;

                    if ((ex!=0) && (ey!=0)) currentCellOffset |= 0x60000000;
                    if ((ex!=0) && (ez!=0)) currentCellOffset |= 0xA0000000;
                    if ((ey!=0) && (ez!=0)) currentCellOffset |= 0xC0000000;
                    if (x == (1<<(exp2divx-1))) currentCellOffset &= ~0x20000000;
                    if (y == (1<<(exp2divy-1))) currentCellOffset &= ~0x40000000;
                    if (z == (1<<(exp2divz-1))) currentCellOffset &= ~0x80000000;

                    baseOffsets[dq].push_back(currentCellOffset);
                    ++counts;

                    for (uint_fast32_t shaving = 0; shaving < 64; ++shaving) {

                        // shave off some offsets
                        if ((dq==0) && ((ex!=0) || (ey!=0) || (ez!=0))) {
                            if (
                               ((((ex<0) && (shaving&ShaveXN)) || ((ex>0) && (shaving&ShaveXP))) && (x != (1<<(exp2divx-1))))
                            || ((((ey<0) && (shaving&ShaveYN)) || ((ey>0) && (shaving&ShaveYP))) && (y != (1<<(exp2divy-1))))
                            || ((((ez<0) && (shaving&ShaveZN)) || ((ez>0) && (shaving&ShaveZP))) && (z != (1<<(exp2divz-1))))
                            ) {
                                continue;
                            }
                        } else if (dq>0) {
                            if (
                            (((ex<0) && (shaving&ShaveXN)) || ((ex>0) && (shaving&ShaveXP)) || (ex==0) || (ex==-1) || (ex==1))
                            && (((ey<0) && (shaving&ShaveYN)) || ((ey>0) && (shaving&ShaveYP)) || (ey==0) || (ey==-1) || (ey==1))
                            && (((ez<0) && (shaving&ShaveZN)) || ((ez>0) && (shaving&ShaveZP)) || (ez==0) || (ez==-1) || (ez==1))
                            && (x != (1<<(exp2divx-1)))
                            && (y != (1<<(exp2divy-1)))
                            && (z != (1<<(exp2divz-1)))
                            ) {
                                continue;
                            }
                        }

                        if ((dq>1) && (
                           ((((ex<=-dbase) && (shaving&ShaveXN)) || ((ex>=dbase) && (shaving&ShaveXP))) && (x != (1<<(exp2divx-1))))
                        || ((((ey<=-dbase) && (shaving&ShaveYN)) || ((ey>=dbase) && (shaving&ShaveYP))) && (y != (1<<(exp2divy-1))))
                        || ((((ez<=-dbase) && (shaving&ShaveZN)) || ((ez>=dbase) && (shaving&ShaveZP))) && (z != (1<<(exp2divz-1))))
                        )) {
                            continue;
                        }

                        sOffsets[shaving][dq].push_back(currentCellOffset);
                        ++sCounts[shaving];
                    }

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
