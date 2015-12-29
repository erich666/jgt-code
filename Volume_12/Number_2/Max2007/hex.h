#ifndef HEX_H_INCLUDED
#define HEX_H_INCLUDED

#include "harTypes.h"

/* DLDL: Now zonetypes only contains the Hexahedron struct which may
 * not be needed anymore ,check this and remove it ifpossible.
 */
#include "zonetypes.h"
#include "projection.h"

enum {
    HEX_SUCCESS = 1,

    HEX_FAIL_DEGENERATE                                                 = -1,
    HEX_FAIL_DEGENERATE_TO_SINGLE_POINT                                 = -2,
    HEX_FAIL_BOWTIE                                                     = -3,
    /* returned when count==3 from inside har_projectHex.
     * Its meaning is unclear to me at this time.
     */
    HEX_FAIL_EQ3                                                        = -4,
    HEX_FAIL_INTERSECTING_EDGES_NOT_CONSECUTIVE                         = -5,
    HEX_FAIL_NO_SECOND_CASE8_EDGE_INTERSECTION                          = -6,
    HEX_FAIL_NO_SECOND_TRY_FOR_CASE8_VERTEX10_INTERSECTION              = -7,
    HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_EDGE                               = -8,
    HEX_FAIL_BOTH_SECOND8_AND_SECOND9_USED                              = -9,
    HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_VERTEX10_INTERSECTION              = -10,
    HEX_FAIL_COUNT2_AND_SECOND8_OR_SECOND9                              = -11,
    HEX_FAIL_NEITHER_OCTAGON_DIAGONAL_FOUND_FOR_VERT0                   = -12,
    HEX_FAIL_MISSED_0_3_AND_1_6_INTERSECTIONS_FOR_OCTAGON               = -13,
    HEX_FAIL_MISSED_2_5_AND_3_0_INTERSECTIONS_FOR_OCTAGON               = -14,
    HEX_FAIL_MISSED_2_5_AND_4_7_INTERSECTIONS_FOR_OCTAGON               = -15,
    HEX_FAIL_MISSED_1_6_AND_4_7_INTERSECTIONS_FOR_OCTAGON               = -16,
    HEX_FAIL_TOP_AND_BOTTOM_VERTS_IDENTICAL                             = -17,
    HEX_FAIL_SECOND8_OR_SECOND9_BUT_NO_CROSS                            = -18,
    HEX_FAIL_UNKNOWN_TRIANGLE_FAN_CASE                                  = -19,


};

typedef struct {
    long int nv8 , nvsecond8;
    long int nv9, nvsecond9 , nosecond9;
    long int nv10, sv810 , sv910;
    long int nonconseq , eq3 , bowties;
    long int no8sv10 , no9sv10;
    long int count0 , draw0;
    long int normal1 , normal2 , normal4;

    long int badcount , totalcount ;
    long int ndegen , nalldegen , toteq ;
} har_HexProjectionStatistics;


typedef struct {
    int test_degenerate;
    int tet_only;
    int save_crossed;
    int outlines;
} har_HexProjectionOptions;

int 
har_hexProject (const har_Projection* proj, 
                const har_Zone *hex, int drawNow, char *crossed_count_turn,
                char *save_in, int *lineword, const double atten_per_length[4],
                const har_HexProjectionOptions* options,
                har_HexProjectionStatistics* stats);

const char*
har_hexProjectionResultString(int returnval);

void
har_initHexProjectionStatistics(har_HexProjectionStatistics* stats);

void
har_printHexProjectionStatistics(har_HexProjectionStatistics* stats, const char* str);

void
har_initHexProjectionOptions(har_HexProjectionOptions* options);

#endif
