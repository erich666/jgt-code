/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines a helper object that is used when the region
    of interest is not cyclic, in any dimension.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

// Specialization: do not wrap at all
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
// |0..0|by|y|nz-1 times 0|bz|z|ny-1 times 0|bx|x|
// With:
// - bA: bit for that variable A outside flag.
//   The trick is to use 2^(exp2+1) complement => this bit is set both for <0 and >=2^exp, exactly what we want
//   Since the offsets are added from a position inside the main region, there cannot be overflow

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, class _Allocator>
struct WrapHelper<UserObject, exp2divx, exp2divy, exp2divz, false, false, false, false, _Allocator> {

// Quick and dirty checks, see above comment on 2^(exp2+1) for why overflow may be a problem.
// A better solution would be to partially specialize the template for exp2 = 1
COMPILE_TIME_ASSERT(exp2divx>=2)
COMPILE_TIME_ASSERT(exp2divy>=2)

// need 3 bits to store the run-time cell checking flags (see apply.hpp)
COMPILE_TIME_ASSERT(exp2divx*2+exp2divy*2+exp2divz+1<=29)

    enum {
        MaxDQ = ((1 << exp2divx) - 2) * ((1 << exp2divx) - 2)
              + ((1 << exp2divy) - 2) * ((1 << exp2divy) - 2)
              + ( (exp2divz>1) ? (((1 << exp2divz) - 2) * ((1 << exp2divz) - 2)) : 0 ),
        ArraySize = (1 << (exp2divx+exp2divy+exp2divz)) + 2,   // add 2 for dummy, real
        MaxVolume = (1<<exp2divx)*(1<<exp2divy)*(1<<exp2divz),
        UnpackMask = ((1 << exp2divy) - 1) << exp2divx,
        UnpackMaskInv = ~(((1 << exp2divy) - 1) << exp2divx),
        OutsideIndex = 1 << (exp2divx + exp2divy + exp2divz),         // for dummy cell
        OutsideIndexReal = (1 << (exp2divx + exp2divy + exp2divz))+1, // for the real one
        UnpackedXMask = (1 << exp2divx) - 1,
        UnpackedYMask = ((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz),
        UnpackedZMask = ((1 << exp2divz) - 1) << (exp2divx+exp2divy),
        XMask = (1 << exp2divx) - 1,
        YMask = ((1 << exp2divy) - 1) << exp2divx,
        ZMask = ((1 << exp2divz) - 1) << (exp2divx+exp2divy),
        ShiftY = exp2divx*2 + exp2divy + exp2divz,
        ShiftZ = exp2divx + exp2divy,
        ShiftInsert = exp2divx + exp2divy + exp2divz,
        MaxEdges = ((1 << exp2divx) - 1) | (((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz)) | (((1 << exp2divz) - 1) << (exp2divx+exp2divy)),
        CorrConst = (1 << (exp2divx + exp2divy + exp2divz)) | (1 << exp2divx),
        PrePackMask = ((1 << (exp2divx+1)) - 1) | (((1 << (exp2divy+1)) - 1) << (exp2divx*2+exp2divy+exp2divz)) | (((1 << (exp2divz+1)) - 1) << (exp2divx+exp2divy)),
        OutPackMask = (1 << exp2divx) | ((1 << exp2divy) << (exp2divx*2+exp2divy+exp2divz)) | ((1 << exp2divz) << (exp2divx+exp2divy)),
        WrapMask = ((1 << exp2divx) - 1) | (((1 << exp2divy) - 1) << (exp2divx*2+exp2divy+exp2divz)) | (((1 << exp2divz) - 1) << (exp2divx+exp2divy)),
        PackMask = (1 << (exp2divx + exp2divy + exp2divz)) - 1,
        XOutMask = 1 << exp2divx,
        YOutMask = 1 << (exp2divx*2 + exp2divy*2 + exp2divz),
        XToZOutShift = exp2divy + exp2divz,
        YToZOutShift = exp2divy + exp2divx,
        ZTo1OutShift = exp2divx+exp2divy+exp2divz,
        EXMask = (1 << (exp2divx+1)) - 1,
        EYMask = (1 << (exp2divy+1)) - 1,
        EZMask = (1 << (exp2divz+1)) - 1
    };

    // Whether the outside region is marked for further processing or not
    uint32_t outsideFlag;
    // Cached reference to outside cell, no need to fetch from array each time
    CellEntry<UserObject>* outsideCell;

    // Dimensions for external check & distance wrapping
    FloatType minx, miny, minz, maxx, maxy, maxz, cellSize, cellSizeInv, cellSizeInvDown, cellSizeSquared;

    NEIGHAND_INLINE WrapHelper(FloatType _minx, FloatType _miny, FloatType _minz, FloatType _cellSize) :
        outsideFlag(0),
        minx(_minx), miny(_miny), minz(_minz),
        maxx(_minx + _cellSize * (1<<exp2divx)),
        maxy(_miny + _cellSize * (1<<exp2divy)),
        maxz(_minz + _cellSize * (1<<exp2divz)),
        cellSize(_cellSize), cellSizeSquared(_cellSize*_cellSize)
    {
        // Switch to round-down FPU mode for cellSizeInvDown, see usage in next function below
        cellSizeInv = FloatType(1.0f) / _cellSize;
        int roundmode = fegetround();
        fesetround(FE_TOWARDZERO);
        cellSizeInvDown = FloatType(1.0f) / _cellSize;
        fesetround(roundmode);
    }

    // This function may return the real outside cell, for adding/removing objects in it
    NEIGHAND_INLINE uint_fast32_t getCellIndexForWorldPosition(FloatConverter dx, FloatConverter dy, FloatConverter dz) NEIGHAND_ALWAYS_INLINE {

        FloatConverter maxTestx(maxx - dx.f);
        FloatConverter maxTesty(maxy - dy.f);
        FloatConverter maxTestz(maxz - dz.f);
        dx.f -= minx;
        dy.f -= miny;
        dz.f -= minz;

        // Don't use FPU compare, but ALU compare to test for bit sign
        // Additionally, it's enough that either x,y,z is <0 or >=max to make the cell outside
        // => no need to make 6 FPU tests: Do only 1 ALU test, and OR ops
        // => OR all bits, including sign bit, and test if it is positionned in the result
        // Note: this would consider -0.0 as outside, +0.0 as inside.
        //       however, -0.0 cannot happen since we get dx/dy/dz by subtracting the values above
        //       => there is no +- "side" distinction for (de)normal numbers, +0.0 is returned in that case
        //       A vicious case would be '-0.0'+'-0.0', but '-0.0'-'-0.0' should give '+0.0' (supposedly)
        // Max conditions are more tricky, equality means outside.
        // -0.0 don't happen either, so 0 is really a zero bit pattern => subtract one to get bit sign flag
        // positive denormal 1 ULP means inside, subtracting one makes 0-bit pattern without bit sign: OK
        // negative denormal 1 ULP means outside, subtracting 1 makes it 0x80000000 still has bit sign = outside, OK
        // => in all cases subtracting one before test works
        // Remaining problems are +Infinity and Positive NaN, "negative" NaN and -Infinity have sign bits.
        // But this whole neighborhood management code assumes they do not happen anyway, at various places.
        if ((dx.i | dy.i | dz.i | (maxTestx.i - 1) | (maxTesty.i - 1) | (maxTestz.i - 1)) > 0x7FFFFFFF)
            return OutsideIndexReal;

        // floor down the value to get cell position, necessary inside now
        // floor(x) is int(x) if x>0, and <0 was checked already
        // Note: see constructor for securing cellSizeInv
        return uint32_t(dx.f * cellSizeInvDown) | (uint32_t(dy.f * cellSizeInvDown) << exp2divx) | (uint32_t(dz.f * cellSizeInvDown) << ShiftZ);
    }

    // This function may return the real outside cell, for adding/removing objects in it
    NEIGHAND_INLINE uint_fast32_t getCellIndexForWorldPosition(FloatConverter x, FloatConverter y, FloatConverter z, int32_t& cellx, int32_t& celly, int32_t& cellz, FloatType &xcenter, FloatType &ycenter, FloatType &zcenter) NEIGHAND_ALWAYS_INLINE {

        FloatConverter maxTestx(maxx - x.f);
        FloatConverter maxTesty(maxy - y.f);
        FloatConverter maxTestz(maxz - z.f);
        x.f -= minx;
        y.f -= miny;
        z.f -= minz;

        xcenter = x.f * cellSizeInvDown;
        cellx = int32_t(xcenter);
        ycenter = y.f * cellSizeInvDown;
        celly = int32_t(ycenter);
        zcenter = z.f * cellSizeInvDown;
        cellz = int32_t(zcenter);

        if ((x.i | y.i | z.i | (maxTestx.i - 1) | (maxTesty.i - 1) | (maxTestz.i - 1)) > 0x7FFFFFFF)
            return OutsideIndexReal;

        return cellx | (celly << exp2divx) | (cellz << ShiftZ);
    }

    // dx * dx + dy * dy + dz * dz, world coordinates. No wrapping, nothing to do.
    NEIGHAND_INLINE FloatType squaredDistance(FloatType dx, FloatType dy, FloatType dz) NEIGHAND_ALWAYS_INLINE {
        return dx * dx + dy * dy + dz * dz;
    }


    NEIGHAND_INLINE FloatType squaredDXCellSpace(FloatType dx) NEIGHAND_ALWAYS_INLINE {
        return dx*dx;
    }

    NEIGHAND_INLINE FloatType squaredDYCellSpace(FloatType dy) NEIGHAND_ALWAYS_INLINE {
        return dy*dy;
    }

    NEIGHAND_INLINE FloatType squaredDZCellSpace(FloatType dz) NEIGHAND_ALWAYS_INLINE {
        return dz*dz;
    }

    // See neighand_apply.hpp, this computes 1-out
    NEIGHAND_INLINE FloatType getSphereWeightingLoadFactor(uint32_t d32) NEIGHAND_ALWAYS_INLINE {
        // quick and dirty: compute average part "out" for a given dist
        // that's integral[0..doutMax]( dout *p(dout)) with p(dout) proba = (dist out)/total size, along each dim X Y Z
        // => doutMax^2 / W, with W = X,Y,Z. Then average over all W = alpha * d^2 with alpha = (XY+YZ+ZX)/(XYZ)
        // Then compute volume of portion of sphere:
        // V = pi/3 * h^2 * (3*d - h) with h = alpha * d^2 here
        // V = pi/3 * alpha^2 * d^4 * ( 3*d + alpha * d^2)
        // V = pi/3 * alpha^2 * d^5 * ( 3 + alpha * d)
        // V = 4 * pi/3 * d^3 * alpha^2 * d^2 / 4 * ( 3 + alpha * d)
        // V = Vs * alpha^2 * d^2 * 0.25 * ( 3 + alpha * d)
        // Then Vs is factored out in the main formula (see neighand_apply.hpp)
        // => out = alpha^2 * d^2 * 0.25 * ( 3 + alpha * d)
        // 1 - out = 1 - (alpha * d)^2 * 0.25 * ( 3 + alpha * d)
        enum {X = (1<<exp2divx), Y = (1<<exp2divx), Z = (1<<exp2divz)};
        enum {XYZ = X*Y*Z, S2T=(X*Y+Y*Z+Z*X)};
        FloatType alpha_d = d32 * 0.03125f * FloatType(S2T) / FloatType(XYZ);
        return 1.0f - alpha_d*alpha_d * 0.25f * (3.0f-alpha_d);
    }


////// Region support: no wrap case, outside possible in every direction /////

    NEIGHAND_INLINE void setOutsideCell(CellEntry<UserObject>* _outsideCell) NEIGHAND_ALWAYS_INLINE {
        outsideCell = _outsideCell;
    }

    NEIGHAND_INLINE void clearOutsideFlag() NEIGHAND_ALWAYS_INLINE {
        outsideFlag = 0;
    }

    NEIGHAND_INLINE uint32_t isInside(FloatConverter x, FloatConverter y, FloatConverter z) NEIGHAND_ALWAYS_INLINE {
        FloatConverter maxTestx(maxx - x.f);
        FloatConverter maxTesty(maxy - y.f);
        FloatConverter maxTestz(maxz - z.f);
        x.f -= minx;
        y.f -= miny;
        z.f -= minz;

        // use float sign bit.
        return 1 - (uint32_t(x.i| y.i | z.i | (maxTestx.i-1) | (maxTesty.i-1) | (maxTestz.i-1)) >> 31);
    }

    // xyz are in world coordinates, inside, not in cell units
    NEIGHAND_INLINE void flagOutside(FloatConverter x, FloatConverter y, FloatConverter z, FloatType d) NEIGHAND_ALWAYS_INLINE {

        // center+radius used to check if sphere intersects one of the 6 planes
        FloatConverter maxTestx(maxx - (x.f+d));
        FloatConverter maxTesty(maxy - (y.f+d));
        FloatConverter maxTestz(maxz - (z.f+d));
        x.f -= minx + d;
        y.f -= miny + d;
        z.f -= minz + d;

        outsideFlag = (outsideCell->objects!=0);
        // use float sign bit.
        outsideFlag &= uint32_t(x.i| y.i | z.i | (maxTestx.i-1) | (maxTesty.i-1) | (maxTestz.i-1)) >> 31;
    }

    NEIGHAND_INLINE bool dsqImpliesOutside(FloatType x, FloatType y, FloatType z, FloatType dsq) NEIGHAND_ALWAYS_INLINE {

        if (!outsideCell->objects) return false;

        // avoid sqrt, but up to 6 muls
        FloatType d = maxx - x; d *= d; d -= dsq;
        if (d<0) return true;
        d = maxy - y; d *= d; d -= dsq;
        if (d<0) return true;
        d = maxz - z; d *= d; d -= dsq;
        if (d<0) return true;
        d = x - minx; d *= d; d -= dsq;
        if (d<0) return true;
        d = y - miny; d *= d; d -= dsq;
        if (d<0) return true;
        d = z - minz; d *= d; d -= dsq;
        if (d<0) return true;

        return false;
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
    NEIGHAND_INLINE void getInternalParallelepiped(int32_t &mincellx, int32_t &mincelly, int32_t &mincellz, int32_t &maxcellx, int32_t &maxcelly, int32_t &maxcellz, FloatType cx, FloatType cy, FloatType cz, FloatType d) NEIGHAND_ALWAYS_INLINE {

        // a,b,c cube englobing the query sphere
        FloatConverter mina(cx - d);
        FloatConverter maxa(cx + d);
        FloatConverter minb(cy - d);
        FloatConverter maxb(cy + d);
        FloatConverter minc(cz - d);
        FloatConverter maxc(cz + d);

        // compare with main region bound
        FloatConverter a1(maxx - mina.f);
        FloatConverter b1(maxy - minb.f);
        FloatConverter c1(maxz - minc.f);
        maxa.f -= minx;
        maxb.f -= miny;
        maxc.f -= minz;

        // Test if any one of the above main region bound is negative => sphere is completely outside
        // => report the condition on z, so the outer loop in apply.hpp is not entered
        if ((maxa.i | maxb.i | maxc.i | (a1.i-1) | (b1.i-1) | (c1.i-1)) > 0x7FFFFFFF) {
            // ensure mincellz > maxcellz for the loop to fail
            mincellx = 1; maxcellx = 0;
            mincelly = 1; maxcelly = 0;
            mincellz = 1; maxcellz = 0;
            return;
        }

        // finish the convertion from world coordinates to cell space
        mina.f -= minx; minb.f -= miny; minc.f -= minz;
        // Force signed cast from float
        mincellx = static_cast<int32_t>(mina.f * cellSizeInv);
        mincelly = static_cast<int32_t>(minb.f * cellSizeInv);
        mincellz = static_cast<int32_t>(minc.f * cellSizeInv);
        maxcellx = static_cast<int32_t>(maxa.f * cellSizeInv);
        maxcelly = static_cast<int32_t>(maxb.f * cellSizeInv);
        maxcellz = static_cast<int32_t>(maxc.f * cellSizeInv);

        // Intersect by saturating each min/max to the main region bound
        // mincellx = max(mincellx, 0) without branching, idem for y,z
        // generate bitmask from sign => if <0 then AND with 0, if positive then AND with 0xFFFFFFFF
        mincellx &= (uint32_t(mincellx) >> 31) - 1;
        mincelly &= (uint32_t(mincelly) >> 31) - 1;
        mincellz &= (uint32_t(mincellz) >> 31) - 1;
        // saturate the max to the last cell index
        // Note: Works only if there is no overflow, that is, coordmax not above MAXINT - 2^exp2
        // translate to 0
        maxcellx -= (1 << exp2divx) - 1;
        maxcelly -= (1 << exp2divy) - 1;
        maxcellz -= (1 << exp2divz) - 1;
        // generate bitmask from sign => if >=0 then AND with 0, if negative then AND with 0xFFFFFFFF
        maxcellx &= -(uint32_t(maxcellx) >> 31);
        maxcelly &= -(uint32_t(maxcelly) >> 31);
        maxcellz &= -(uint32_t(maxcellz) >> 31);
        // translate back
        maxcellx += (1 << exp2divx) - 1;
        maxcelly += (1 << exp2divy) - 1;
        maxcellz += (1 << exp2divz) - 1;
    }

    // center position in world coordinates, return the inside parallelepiped volume
    NEIGHAND_INLINE uint_fast32_t getInternalParallelepipedVolume(FloatType cx, FloatType cy, FloatType cz, FloatType d) NEIGHAND_ALWAYS_INLINE {

        // a,b,c cube englobing the query sphere
        FloatConverter mina(cx - d);
        FloatConverter maxa(cx + d);
        FloatConverter minb(cy - d);
        FloatConverter maxb(cy + d);
        FloatConverter minc(cz - d);
        FloatConverter maxc(cz + d);

        // compare with main region bound
        FloatConverter a1(maxx - mina.f);
        FloatConverter b1(maxy - minb.f);
        FloatConverter c1(maxz - minc.f);
        maxa.f -= minx;
        maxb.f -= miny;
        maxc.f -= minz;

        // Test if any one of the above main region bound is negative => sphere is completely outside
        if ((maxa.i | maxb.i | maxc.i | (a1.i-1) | (b1.i-1) | (c1.i-1)) > 0x7FFFFFFF) {
            return 0;
        }

        // finish the convertion from world coordinates to cell space
        mina.f -= minx; minb.f -= miny; minc.f -= minz;
        // Force signed cast from float
        int32_t mincellx = static_cast<int32_t>(mina.f * cellSizeInv);
        int32_t mincelly = static_cast<int32_t>(minb.f * cellSizeInv);
        int32_t mincellz = static_cast<int32_t>(minc.f * cellSizeInv);
        int32_t maxcellx = static_cast<int32_t>(maxa.f * cellSizeInv);
        int32_t maxcelly = static_cast<int32_t>(maxb.f * cellSizeInv);
        int32_t maxcellz = static_cast<int32_t>(maxc.f * cellSizeInv);

        // Intersect by saturating each min/max to the main region bound
        // mincellx = max(mincellx, 0) without branching, idem for y,z
        // generate bitmask from sign => if <0 then AND with 0, if positive then AND with 0xFFFFFFFF
        mincellx &= (uint32_t(mincellx) >> 31) - 1;
        mincelly &= (uint32_t(mincelly) >> 31) - 1;
        mincellz &= (uint32_t(mincellz) >> 31) - 1;
        // saturate the max to the last cell index
        // Note: Works only if there is no overflow, that is, coordmax not above MAXINT - 2^exp2
        // translate to 0
        maxcellx -= (1 << exp2divx) - 1;
        maxcelly -= (1 << exp2divy) - 1;
        maxcellz -= (1 << exp2divz) - 1;
        // generate bitmask from sign => if >=0 then AND with 0, if negative then AND with 0xFFFFFFFF
        maxcellx &= -(uint32_t(maxcellx) >> 31);
        maxcelly &= -(uint32_t(maxcelly) >> 31);
        maxcellz &= -(uint32_t(maxcellz) >> 31);
        // translate back, plus 1
        maxcellx += (1 << exp2divx);
        maxcelly += (1 << exp2divy);
        maxcellz += (1 << exp2divz);

        // maxcell - mincell + 1 in each dim
        return (maxcellx - mincellx) * (maxcelly - mincelly) * (maxcellz - mincellz);
    }

// Static functions below

    static NEIGHAND_INLINE bool isInside(uint_fast32_t packedIndex) NEIGHAND_ALWAYS_INLINE {
        return packedIndex < OutsideIndex;
    }

    static NEIGHAND_INLINE int isUnpackedOutside(uint_fast32_t unpackedIndex) NEIGHAND_ALWAYS_INLINE {
        return unpackedIndex & OutPackMask;
    }

    static NEIGHAND_INLINE uint_fast32_t unpack(uint_fast32_t idx) NEIGHAND_ALWAYS_INLINE {
        // internal, no spurious bit, just unpack
        return ((idx & UnpackMask) << ShiftInsert) | (idx & UnpackMaskInv);
    }

    // This function may not return the real outside cell, only the dummy index one (always empty)
    static NEIGHAND_INLINE uint_fast32_t packwrap(uint_fast32_t idx) NEIGHAND_ALWAYS_INLINE {
        idx &= WrapMask;
        return ((idx >> ShiftInsert) | idx) & PackMask;
    }

    // This function always receive valid arguments in range in the ffff case, no need to mask
    static NEIGHAND_INLINE uint_fast32_t xyzToPackedIndex(int_fast32_t cellx, int_fast32_t celly, int_fast32_t cellz) NEIGHAND_ALWAYS_INLINE {
        return cellx | (celly << exp2divx) | (cellz << ShiftZ);
    }

    static NEIGHAND_INLINE void packedIndexToXYZ(uint32_t packedCellIndex, int32_t& cellx, int32_t& celly, int32_t& cellz) NEIGHAND_ALWAYS_INLINE {
        cellx = packedCellIndex & XMask;
        celly = (packedCellIndex & YMask) >> exp2divx;
        cellz = (packedCellIndex & ZMask) >> ShiftZ;
    }

    static NEIGHAND_INLINE int32_t extractX(uint32_t offset) NEIGHAND_ALWAYS_INLINE {
        int32_t ret = offset & EXMask;
        ret |= -(ret & (1<< exp2divx)); // extend bit sign
        return ret;
    }
    static NEIGHAND_INLINE int32_t extractY(uint32_t offset) NEIGHAND_ALWAYS_INLINE {
        int32_t ret = (offset >> ShiftY) & EYMask;
        ret |= -(ret & (1<< exp2divy)); // extend bit sign
        return ret;
    }
    static NEIGHAND_INLINE int32_t extractZ(uint32_t offset) NEIGHAND_ALWAYS_INLINE {
        int32_t ret = (offset >> ShiftZ) & EZMask;
        ret |= -(ret & (1<< exp2divz)); // extend bit sign
        return ret;
    }

    static NEIGHAND_INLINE FloatType wrapCellDeltaX(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return delta;
    }
    static NEIGHAND_INLINE FloatType wrapCellDeltaY(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return delta;
    }
    static NEIGHAND_INLINE FloatType wrapCellDeltaZ(FloatType delta) NEIGHAND_ALWAYS_INLINE {
        return delta;
    }

    static NEIGHAND_INLINE FloatType distCompHeuristicConstant() NEIGHAND_ALWAYS_INLINE {
        return FloatType(0.99999f);
    }

    // take two cells at unpacked index idx1 and idx2, both inside
    // return the minimum squared distance any point in these cells can have, in cell units
    static NEIGHAND_INLINE uint_fast32_t getWrappedDeltaUnpackedIndexDQ(uint_fast32_t idx1, uint_fast32_t idx2) NEIGHAND_ALWAYS_INLINE {
        // 1. Extract X/Y/Z
        uint32_t dx = (idx1 & UnpackedXMask) - (idx2 & UnpackedXMask);
        uint32_t dy = ((idx1 & UnpackedYMask) >> ShiftY) - ((idx2 & UnpackedYMask) >> ShiftY);
        uint32_t dz = ((idx1 & UnpackedZMask) >> ShiftZ) - ((idx2 & UnpackedZMask) >> ShiftZ);
        // 2. bit signs
        uint32_t bsx = dx >> 31;
        uint32_t bsy = dy >> 31;
        uint32_t bsz = dz >> 31;
        // 3. compute abs(xyz) - 1
        dx = (dx ^ (-bsx)) + bsx - 1;
        dy = (dy ^ (-bsy)) + bsy - 1;
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

        // N - version/exp2divx/y/z on 4 bits each = 16 bits - ffff in binary = 0000
        uint32_t version = 1; // binary file format version, not software release version
        uint32_t magic = 'N' | ( ((version<<12)|((exp2divx<<8)|(exp2divy<<4)|exp2divz))<< 8) | ( 0x0 << 24);

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

                    // handle the 8 directions, the distances above are preserved by symmetry.
                    // clumsy way, but it works
                    int xval[] = {x, -x};
                    int yval[] = {y, -y};
                    int zval[] = {z, -z};
                    // should not duplicate the cells at 0 offset in any dim
                    for (int iz = 0; iz <= (z!=0); ++iz)
                    for (int iy = 0; iy <= (y!=0); ++iy)
                    for (int ix = 0; ix <= (x!=0); ++ix) {
                        // Extended x/y/z, possibly negative
                        int ex = xval[ix], ey = yval[iy], ez = zval[iz];

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

                        uint32_t currentCellOffset = (uint32_t(ex) & EXMask)
                            | ((uint32_t(ey) & EYMask) << ShiftY)
                            | ((uint32_t(ez) & EZMask) << ShiftZ);

                        if ((ex!=0) && (ey!=0)) currentCellOffset |= 0x60000000;
                        if ((ex!=0) && (ez!=0)) currentCellOffset |= 0xA0000000;
                        if ((ey!=0) && (ez!=0)) currentCellOffset |= 0xC0000000;

                        baseOffsets[dq].push_back(currentCellOffset);
                        ++counts;

                        for (uint_fast32_t shaving = 0; shaving < 64; ++shaving) {

                            // shave off some offsets
                            if ((dq==0) && ((ex!=0) || (ey!=0) || (ez!=0))) {
                                if ( ((ex<0) && (shaving&ShaveXN)) || ((ex>0) && (shaving&ShaveXP))
                                || ((ey<0) && (shaving&ShaveYN)) || ((ey>0) && (shaving&ShaveYP))
                                || ((ez<0) && (shaving&ShaveZN)) || ((ez>0) && (shaving&ShaveZP))
                                ) {
                                    continue;
                                }
                            } else if (dq>0) {
                                if ((((ex<0) && (shaving&ShaveXN)) || ((ex>0) && (shaving&ShaveXP)) || (ex==0) || (ex==-1) || (ex==1))
                                && (((ey<0) && (shaving&ShaveYN)) || ((ey>0) && (shaving&ShaveYP)) || (ey==0) || (ey==-1) || (ey==1))
                                && (((ez<0) && (shaving&ShaveZN)) || ((ez>0) && (shaving&ShaveZP)) || (ez==0) || (ez==-1) || (ez==1))) {
                                    continue;
                                }
                            }

                            if ((dq>1) && ( ((ex<=-dbase) && (shaving&ShaveXN)) || ((ex>=dbase) && (shaving&ShaveXP))
                            || ((ey<=-dbase) && (shaving&ShaveYN)) || ((ey>=dbase) && (shaving&ShaveYP))
                            || ((ez<=-dbase) && (shaving&ShaveZN)) || ((ez>=dbase) && (shaving&ShaveZP)))) {
                                continue;
                            }

                            sOffsets[shaving][dq].push_back(currentCellOffset);
                            ++sCounts[shaving];
                        }

                    } // end clumsy 8-way switcher

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


