/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines a block of code for the processing of one cell,
    in the routine for finding only the N nearest neighbors.

    Multiple inclusions were preferred instead of error-prone copy/paste.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/


// Case where N==1 is optimized compared to N>1 => different implementation
#if defined(NEIGHAND_CLOSEST_N_EQ_1)
#if defined(NEIGHAND_CLOSEST_CHECK_FIRST)
while (plist)
#else
do
#endif
{
    FloatType dist = helper.squaredDistance(plist->x - x, plist->y - y, plist->z - z);
    if ((dist<=neighborSquaredDistance)
#ifndef NEIGHAND_APPLY_XYZ_API
    && (plist->object!=avoidObject)
#endif
    ) {
        neighborSquaredDistance = dist;
        neighborObject = plist->object;
    }
    plist = plist->next;
}
#if !defined(NEIGHAND_CLOSEST_CHECK_FIRST)
while (plist);
#endif

#else
NearestNeighbor<UserObject> currentObject;
#if defined(NEIGHAND_CLOSEST_CHECK_FIRST)
while (plist)
#else
do
#endif
{
    currentObject.squaredDistance = helper.squaredDistance(plist->x - x, plist->y - y, plist->z - z);
    currentObject.object = plist->object;
    // object too far => rejected
    if ((currentObject.squaredDistance<=dsq)
#ifndef NEIGHAND_APPLY_XYZ_API
    // object is the caller => rejected
    && (plist->object!=avoidObject)
#endif
    ){
        uint_fast32_t i=0;
#if defined(NEIGHAND_CLOSEST_CHECK_FOR_DUPS)
        for (uint_fast32_t j=0; j<nfound; ++j) if (neighbor[j].object==plist->object) {
            i=N; break;
        }
        if ((i==0) && (++nfound>N)) nfound=N;
#else
        // if there is some space left, use it
        if (++nfound>N) nfound=N;
#endif
        // bubble up. Complex sorting algo are costly for small N anyway
        for (; i<nfound; ++i) if (currentObject.squaredDistance < neighbor[i].squaredDistance) {
            NearestNeighbor<UserObject> tmp = neighbor[i];
            neighbor[i] = currentObject;
            currentObject = tmp;
        }
        // if array is full, use current max distance to limit the search
        if (nfound==N) dsq = neighbor[N-1].squaredDistance;
    }
    plist = plist->next;
}
#if !defined(NEIGHAND_CLOSEST_CHECK_FIRST)
while (plist);
#endif

#endif
