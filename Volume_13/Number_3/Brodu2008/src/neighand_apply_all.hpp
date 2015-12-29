/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines the functions for applying a user-defined
    functor or callback to all objects in the region of interest.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

#if defined(NEIGHAND_C_CALLBACK_API)

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
NEIGHAND_INLINE void NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::applyToAll(Callback f, void* userData)

#else

template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator>
template<typename Functor>
NEIGHAND_INLINE Functor NeighborhoodHandler<NEIGHAND_TEMPLATE_ARGUMENTS>::applyToAll(Functor f)

#endif

{
    ObjectProxy<UserObject>* endProxy = allProxies + numProxies;
    for (ObjectProxy<UserObject>* proxy = allProxies; proxy<endProxy; ++proxy) {
#ifdef NEIGHAND_C_CALLBACK_API
        f(proxy->object, userData);
#else
        f(proxy->object);
#endif
    }

#ifndef NEIGHAND_C_CALLBACK_API
    return f;
#endif
}
