#ifndef HAR_TYPES_INCLUDED
#define HAR_TYPES_INCLUDED

/* =========================================================================
 *
 * General Definitions
 *
 * =========================================================================
 */

typedef enum {
    HAR_OK          =  1,
    HAR_ERROR_FAIL  =  0,  /* general error */
    /* specific error enums with negative values can be added here */

    HAR_ERROR_CLIPPED_BEHIND_DETECTOR                                    = -1,
    HAR_ERROR_CLIPPED_BEHIND_SOURCE                                      = -2,
} har_ReturnValue;

/* =========================================================================
 *
 * Geometric Definitions
 *
 * =========================================================================
 */

typedef double har_CoordType;
typedef int    har_IndexType; /* for indexing nodes, edges, face, zones, etc.*/

typedef struct {
    har_CoordType x,y,z;
} har_Point3;


/* We include only the zoo elements.  If general polyhedra 
 * must be supported, the reader must subdivide them into
 * one of the supported cell types below.
 */
typedef enum {
    HAR_ZONETYPE_HEX    = 0,
    HAR_ZONETYPE_PRISM,
    HAR_ZONETYPE_PYRAMID,
    HAR_ZONETYPE_TET
} har_ZoneType;

typedef struct {
    har_ZoneType type; /* Provides interpretation for positions */
    har_IndexType node[8];  /* indices of the nodes in the mesh */
    har_CoordType x[8];     /* positions of the nodes */
    har_CoordType y[8];
    har_CoordType z[8];
} har_Zone;

#endif /* include guard */

