/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines a block of code for the processing of one cell,
    in the main routine that applies a functor to neighbors.
    It handles the case where the target object should first be checked
    for potential rejection, in the case of cells that fall partially
    outside the query sphere.

    Multiple inclusions were preferred instead of error-prone copy/paste.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/


#if defined(NEIGHAND_APPLY_CHECK_FIRST)
while (plist)
#else
do
#endif
{
    if ((helper.squaredDistance(x-plist->x, y-plist->y, z-plist->z) <= dsq)
#ifndef NEIGHAND_APPLY_XYZ_API
    && (plist->object!=avoidObject)
#endif
    )
#ifdef NEIGHAND_C_CALLBACK_API
    f(plist->object, userData);
#else
    f(plist->object);
#endif
    plist = plist->next;
}
#if !defined(NEIGHAND_APPLY_CHECK_FIRST)
while (plist);
#endif

