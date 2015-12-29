#ifndef INTPAIRMAP_H
#define INTPAIRMAP_H

#include <algorithm>
#include <vector>
#include "geom.h"

// map of (int,int) to int
class IntPairMap
{
public:
    IntPairMap() : numEntries(0) {}
    // find: returns -1 if not found; overwrite to add to map
    int& find(int v1, int v2) {
	// grow if needed
	int size = table.size();
	if (numEntries >= size / 2) 
	{ 
	    reserve(size*2);
	    size = table.size(); 
	}
	// start at hash pos and look for entry or empty slot
	unsigned int hash = v1 * 1664525 + v2; // constant from Knuth
	Entry* e = &table[hash%size];
	Entry* end = &table[size];
	while (1) {
	    if (e->v1 == v1 && e->v2 == v2) return e->id;
	    if (e->id == -1) break;
	    if (++e == end) e = &table[0];
	}
	// not found, insert
	e->v1 = v1; e->v2 = v2;
	numEntries++;
	return e->id;
    }
    void reserve(int size)
    {
	size = std::max(size, 16);
	if (size > table.size()) {
	    IntPairMap em; em.table.resize(size);
	    Entry* end = &table[table.size()];
	    for (Entry* e = &table[0]; e != end; e++)
		em.find(e->v1, e->v2) = e->id;
	    std::swap(table, em.table);
	}
    }
    void clear() { table.clear(); numEntries = 0; }
private:
    struct Entry { 
	int v1, v2, id; 
	Entry() : v1(-1), v2(-1), id(-1) {}
    };
    int numEntries;
    std::vector<Entry> table;
};


struct ComparePoints
{
    bool operator() (const Vec3f& a, const Vec3f& b)
    {
	if (a.equals(b, 1e-4)) return 0;
	if (a[0] < b[0]) return 1;
	if (a[0] > b[0]) return 0;
	if (a[1] < b[1]) return 1;
	if (a[1] > b[1]) return 0;
	if (a[2] < b[2]) return 1;
	return 0;
    }
};

typedef std::map<Vec3f, int, ComparePoints> Vec3fMap;

#endif
