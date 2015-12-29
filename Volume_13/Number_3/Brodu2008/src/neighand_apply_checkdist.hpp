/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines a block of code for checking whether a cell is
    out of range, in the main routine that applies a functor to neighbors.

    Multiple inclusions were preferred instead of error-prone copy/paste.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

if ((offset & 0xE0000000) != 0) {
FloatType dcell(0.0f);
if ((offset & 0x80000000) != 0) {
    int32_t id = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::extractZ(offset);
    FloatType tmp = cachedDeltaZ[id].f;
    if (!cachedDeltaZ[id].i) {
        tmp = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::wrapCellDeltaZ(FloatType(int32_t(id + (uint32_t(id)>>31))) - zcenter);
        tmp *= tmp;
        cachedDeltaZ[id].f = tmp;
    }
    dcell += tmp;
}
if ((offset & 0x40000000) != 0) {
    int32_t id = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::extractY(offset);
    FloatType tmp = cachedDeltaY[id].f;
    if (!cachedDeltaY[id].i) {
        tmp = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::wrapCellDeltaY(FloatType(int32_t(id + (uint32_t(id)>>31))) - ycenter);
        tmp *= tmp;
        cachedDeltaY[id].f = tmp;
    }
    dcell += tmp;
}
if ((offset & 0x20000000) != 0) {
    int32_t id = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::extractX(offset);
    FloatType tmp = cachedDeltaX[id].f;
    if (!cachedDeltaX[id].i) {
        tmp = WrapHelper<NEIGHAND_TEMPLATE_ARGUMENTS>::wrapCellDeltaX(FloatType(int32_t(id + (uint32_t(id)>>31))) - xcenter);
        tmp *= tmp;
        cachedDeltaX[id].f = tmp;
    }
    dcell += tmp;
}
if (dcell > dsq_cellSpace) continue;
}
