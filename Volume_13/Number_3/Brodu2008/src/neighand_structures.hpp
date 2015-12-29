/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines the data structures that are used for this
    project.

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/
#ifndef NEIGHAND_STRUCTURES_HPP
#define NEIGHAND_STRUCTURES_HPP

// concat the arguments
#define COMPILE_TIME_ASSERT_CONCAT_HELPER(X,Y) X ## Y
// expand the macro arguments, including __LINE__
#define COMPILE_TIME_ASSERT_CONCAT(X,Y) COMPILE_TIME_ASSERT_CONCAT_HELPER(X,Y)
template<int line, bool x> struct AssertionFailedLine;
template<int line> struct AssertionFailedLine<line, true> {};
#define COMPILE_TIME_ASSERT(x) \
struct COMPILE_TIME_ASSERT_CONCAT(CompileTimeAssertionFailed_Line,__LINE__) : public AssertionFailedLine<__LINE__,bool(x)> { };

// Assert a few assumptions
COMPILE_TIME_ASSERT(sizeof(FloatType)==4)
// sizeof(char) is always 1, but char is not always 8 bits... => reject these machines
COMPILE_TIME_ASSERT(sizeof(uint32_t)==4)
// We need memory pointers to be 4 bytes too
COMPILE_TIME_ASSERT(sizeof(void*)==4)
// We would need to check that float endianity is the same as integer endianity
// This apparently cannot be done at compile time according to current language specs
// Specific system flags could be used, like FLOAT_BYTE_ORDER and BYTE_ORDER macros.

template <typename UserObject>
struct CellEntry;

// What is passed to the DB user
template <typename UserObject>
struct ObjectProxy {
    FloatType x;                        // align x/y/z on 16-bytes boundary, plan for SSE2 extension
    FloatType y;                        //
    FloatType z;                        //
    ObjectProxy* next;                  // next proxy in the cell list
    UserObject* object;                 // user object
    ObjectProxy* prev;                  // prev proxy in the cell list
    uint32_t cellIndex;                 // index of the top-level cell
    CellEntry<UserObject>* cell;
};
// COMPILE_TIME_ASSERT(sizeof(ObjectProxy)==32)

// Main object for the array of cells. Also maintains the non-empty lists
// DESIGN CHOICE:
// - This cell size is limited to 8 because array addressing *8 is builtin on x86, *16 is not
// - The linked list was initially double-linked, which forces cell size 16
//   It is now single-linked, and ordered in increasing cell indices.
//   => the memory access is much faster when running through the non-empty list
template <typename UserObject>
struct CellEntry {
    ObjectProxy<UserObject>* objects;      // 0 if empty cell
    CellEntry* nextNonEmpty;   // single-linked list
};
// COMPILE_TIME_ASSERT(sizeof(CellEntry)==8)

template <typename UserObject>
struct NearestNeighbor {
    UserObject* object;
    FloatType squaredDistance;
};

// Example of remapper for objects that have a "proxy" field
template<typename UserObject> struct ProxiedObjectRemapper {
    void operator()(UserObject* object, ObjectProxy<UserObject>* updated_proxy){
        object->proxy = updated_proxy;
    }
};

enum QueryMethod {Auto, Sphere, Cube, NonEmpty, Brute};

// Previous version used reinterpret_cast, which is C++ way
// However this badly interferes with aliasing, and -fno-strict-aliasing was necessary
// Using a union is handled by the compiler and allows to assume aliasing
// hence the compiler can optimize better
union FloatConverter {
    FloatType f;
    uint32_t i;
    // C++ support for union constructors is good, avoids declare/affect/retrieve syntax
    NEIGHAND_INLINE FloatConverter(FloatType _f) : f(_f) {}
    // initialized constructor: used for neighand_apply.hpp
    NEIGHAND_INLINE FloatConverter() : i(0) {}
    // avoid confusion when multiple implicit conversions would be possible.
    NEIGHAND_INLINE explicit FloatConverter(uint32_t _i) : i(_i) {}
};
COMPILE_TIME_ASSERT(sizeof(FloatConverter)==4)

#endif
