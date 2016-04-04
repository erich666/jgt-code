//-------------------------------------------------------------------------------
//  Copyright 2002 Mitsubishi Electric Research Laboratories.  
//  All Rights Reserved.
//
//  Permission to use, copy, modify and distribute this software and its 
//  documentation for educational, research and non-profit purposes, without fee, 
//  and without a written agreement is hereby granted, provided that the above 
//  copyright notice and the following three paragraphs appear in all copies.
//
//  To request permission to incorporate this software into commercial products 
//  contact MERL - Mitsubishi Electric Research Laboratories, 201 Broadway, 
//  Cambridge, MA 02139.
//
//  IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, 
//  INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF 
//  THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED 
//  OF THE POSSIBILITY OF SUCH DAMAGES.
//
//  MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
//  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
//  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO 
//  OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR 
//  MODIFICATIONS.
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
//  Generic quadtree cell. Note that the locational codes and the cell level are 
//  only used in neighbor searching; they are not necessary for point or region 
//  location.
//-------------------------------------------------------------------------------
typedef struct _qtCell {
    unsigned int    xLocCode;   // X locational code
    unsigned int    yLocCode;   // Y locational code
    unsigned int    level;      // Cell level in hierarchy (smallest cell has level 0)
    struct _qtCell  *parent;    // Pointer to parent cell
    struct _qtCell  *children;  // Pointer to first of 4 contiguous child cells
    void            *data;      // Application specific cell data
}   qtCell;


//-------------------------------------------------------------------------------
//  Maximum quadtree depth and related constants
//-------------------------------------------------------------------------------
#define QT_N_LEVELS   16        // Number of possible levels in the quadtree
#define QT_ROOT_LEVEL 15        // Level of root cell (QT_N_LEVELS - 1)
#define QT_MAX_VAL    32768.0f  // For converting positions to locational codes 
                                // (QT_MAX_VAL = 2^QT_ROOT_LEVEL)


//-------------------------------------------------------------------------------
//  Macro to traverse a quadtree from a specified cell (typically the root cell) 
//  to a leaf cell by following the x and y locational codes, xLocCode and 
//  yLocCode. Upon entering, cell is the specified cell and nextLevel is one less 
//  than the level of the specified cell. Upon termination, cell is the leaf cell
//  and nextLevel is one less than the level of the leaf cell. 
//-------------------------------------------------------------------------------
#define QT_TRAVERSE(cell,nextLevel,xLocCode,yLocCode)                             \
{                                                                                 \
    while ((cell)->children) {                                                    \
        unsigned int childBranchBit = 1 << (nextLevel);                           \
        unsigned int childIndex = ((((xLocCode) & childBranchBit) >> (nextLevel)) \
        + (((yLocCode) & childBranchBit) >> (--(nextLevel))));                    \
        (cell) = &(((cell)->children)[childIndex]);                               \
    }                                                                             \
}


//-------------------------------------------------------------------------------
//  Macro to traverse a quadtree from a specified cell to an offspring cell by 
//  following the x and y locational codes, xLocCode and yLocCode. The offpring 
//  cell is either at a specified level or is a leaf cell if a leaf cell is 
//  reached before the specified level. Upon entering, cell is the specified 
//  cell and nextLevel is one less than the level of the specified cell. Upon 
//  termination, cell is the offspring cell and nextLevel is one less than the 
//  level of the offspring cell.
//-------------------------------------------------------------------------------
#define QT_TRAVERSE_TO_LEVEL(cell,nextLevel,xLocCode,yLocCode,level)              \
{                                                                                 \
    unsigned int n = (nextLevel) - (level) + 1;                                   \
    while (n--) {                                                                 \
        unsigned int childBranchBit = 1 << (nextLevel);                           \
        unsigned int childIndex = ((((xLocCode) & childBranchBit) >> (nextLevel)) \
        + (((yLocCode) & childBranchBit) >> (--(nextLevel))));                    \
        (cell) = &(((cell)->children)[childIndex]);                               \
        if (!(cell)->children) break;                                             \
    }                                                                             \
}


//-------------------------------------------------------------------------------
//  Macro for traversing a quadtree to a common ancestor of a specified cell 
//  and its neighbor, whose x or y locational code differs from the cell's
//  corresponding x or y locational code by binaryDiff (determined by XOR'ing the 
//  appropriate pair of x or y locational codes). Upon entering, cell is the 
//  specified cell and cellLevel is the cell's level. Upon termination, cell is 
//  the common ancestor and cellLevel is the common ancestor's level.
//-------------------------------------------------------------------------------
#define QT_GET_COMMON_ANCESTOR(cell,cellLevel,binaryDiff)                         \
{                                                                                 \
    while ((binaryDiff) & (1 << (cellLevel))) {                                   \
        (cell) = (cell)->parent;                                                  \
        (cellLevel)++;                                                            \
    }                                                                             \
}


//-------------------------------------------------------------------------------
//  Locate the leaf cell containing the specified point p, where p lies in 
//  [0,1)x[0,1).
//-------------------------------------------------------------------------------
qtCell *qtLocateCell (qtCell *root, float p[2])
{
    //----Determine the x and y locational codes of the point's position. Refer 
    //----to [King2001] for more efficient methods for converting floating point 
    //----numbers to integers.
    unsigned int xLocCode = (unsigned int) (p[0] * QT_MAX_VAL); 
    unsigned int yLocCode = (unsigned int) (p[1] * QT_MAX_VAL); 


    //----Follow the branching patterns of the locational codes from the root cell
    //----to locate the leaf cell containing p
    qtCell *cell = root;
    unsigned int nextLevel = QT_ROOT_LEVEL - 1;
    QT_TRAVERSE(cell,nextLevel,xLocCode,yLocCode);
    return(cell);
}


//-------------------------------------------------------------------------------
//  Locate the smallest cell that entirely contains a rectangular region defined 
//  by its bottom-left vertex v0 and its top-right vertex v1, where v0 and v1 
//  lie in [0,1)x[0,1).
//-------------------------------------------------------------------------------
qtCell *qtLocateRegion (qtCell *root, float v0[2], float v1[2])
{
    //----Determine the x and y locational codes of the region boundaries. Refer 
    //----to [King2001] for more efficient methods for converting floating point 
    //----numbers to integers.
    unsigned int x0LocCode = (unsigned int) (v0[0] * QT_MAX_VAL); 
    unsigned int y0LocCode = (unsigned int) (v0[1] * QT_MAX_VAL); 
    unsigned int x1LocCode = (unsigned int) (v1[0] * QT_MAX_VAL); 
    unsigned int y1LocCode = (unsigned int) (v1[1] * QT_MAX_VAL); 


    //----Determine the XOR'ed pairs of locational codes of the region boundaries
    unsigned int xDiff = x0LocCode ^ x1LocCode;
    unsigned int yDiff = y0LocCode ^ y1LocCode;


    //----Determine the level of the smallest possible cell entirely containing 
    //----the region
    qtCell *cell = root;
    unsigned int level = QT_ROOT_LEVEL;
    unsigned int minLevel = QT_ROOT_LEVEL;
    while (!(xDiff & (1 << level)) && level) level--;
    while (!(yDiff & (1 << minLevel)) && (minLevel > level)) minLevel--;
    minLevel++;


    //----Follow the branching patterns of the locational codes of v0 from the 
    //----root cell to the smallest cell entirely containing the region
    level = QT_ROOT_LEVEL - 1;
    QT_TRAVERSE_TO_LEVEL(cell,level,x0LocCode,y0LocCode,minLevel);
    return(cell);
}


//-------------------------------------------------------------------------------
//  Locate the left edge neighbor of the same size or larger than a specified 
//  cell. A null pointer is returned if no such neighbor exists.
//-------------------------------------------------------------------------------
qtCell *qtLocateLeftNeighbor (qtCell *cell)
{
    //----No left neighbor if this is the left side of the quadtree
    if (cell->xLocCode == 0) return(0);
    else {
        //----Get cell's x and y locational codes and the x locational code of the
        //----cell's smallest possible left neighbor
        unsigned int xLocCode = cell->xLocCode;
        unsigned int yLocCode = cell->yLocCode;
        unsigned int xLeftLocCode = xLocCode - 0x00000001;
        

        //----Determine the smallest common ancestor of the cell and the cell's 
        //----smallest possible left neighbor
        unsigned int cellLevel, nextLevel;
        unsigned int diff = xLocCode ^ xLeftLocCode;
        qtCell *pCell = cell;
        cellLevel = nextLevel = cell->level;
        QT_GET_COMMON_ANCESTOR(pCell,nextLevel,diff);
        

        //----Start from the smallest common ancestor and follow the branching 
        //----patterns of the locational codes downward to the smallest left
        //----neighbor of size greater than or equal to cell
        nextLevel--;
        QT_TRAVERSE_TO_LEVEL(pCell,nextLevel,xLeftLocCode,yLocCode,cellLevel);
        return(pCell);
    }
}


//-------------------------------------------------------------------------------
//  Locate the right edge neighbor of the same size or larger than a specified
//  cell. A null pointer is returned if no such neighbor exists.
//-------------------------------------------------------------------------------
qtCell *qtLocateRightNeighbor (qtCell *cell)
{
    //----No right neighbor if this is the right side of the quadtree
    unsigned int binaryCellSize = 1 << cell->level;
    if ((cell->xLocCode + binaryCellSize) >= (1 << QT_ROOT_LEVEL)) return(0);
    else {
        //----Get cell's x and y locational codes and the x locational code of the
        //----cell's right neighbors
        unsigned int xLocCode = cell->xLocCode;
        unsigned int yLocCode = cell->yLocCode;
        unsigned int xRightLocCode = xLocCode + binaryCellSize;
        

        //----Determine the smallest common ancestor of the cell and the cell's 
        //----right neighbors 
        unsigned int cellLevel, nextLevel;
        unsigned int diff = xLocCode ^ xRightLocCode;
        qtCell *pCell = cell;
        cellLevel = nextLevel = cell->level;
        QT_GET_COMMON_ANCESTOR(pCell,nextLevel,diff);
        

        //----Start from the smallest common ancestor and follow the branching 
        //----patterns of the locational codes downward to the smallest right
        //----neighbor of size greater than or equal to cell
        nextLevel--;
        QT_TRAVERSE_TO_LEVEL(pCell,nextLevel,xRightLocCode,yLocCode,cellLevel);
        return(pCell);
    }
}


//-------------------------------------------------------------------------------
//  Locate the three leaf cell vertex neighbors touching the right-bottom vertex 
//  of a specified cell. bVtxNbr, rVtxNbr, and rbVtxNbr are set to null if the  
//  corresponding neighbor does not exist.
//-------------------------------------------------------------------------------
void qtLocateRBVertexNeighbors (qtCell *cell, qtCell **bVtxNbr, qtCell **rVtxNbr,
qtCell **rbVtxNbr)
{
    //----There are no right neighbors if this is the right side of the quadtree and 
    //----no bottom neighbors if this is the bottom of the quadtree
    unsigned int binCellSize = 1 << cell->level;
    unsigned int noRight = ((cell->xLocCode + binCellSize) >= (1 << QT_ROOT_LEVEL)) ? 1 : 0;
    unsigned int noBottom = (cell->yLocCode == 0) ? 1 : 0;


    //----Get cell's x and y locational codes and the x and y locational codes of 
    //----the cell's right and bottom vertex neighbors
    unsigned int xRightLocCode = cell->xLocCode + binCellSize;
    unsigned int xLocCode = xRightLocCode - 0x00000001;
    unsigned int yLocCode = cell->yLocCode;
    unsigned int yBottomLocCode = yLocCode - 0x00000001;
    unsigned int rightLevel, bottomLevel;
    unsigned int diff;
    qtCell *commonRight, *commonBottom;


    //----Determine the right leaf cell vertex neighbor 
    if (noRight) *rVtxNbr = 0;
    else {
        //----Determine the smallest common ancestor of the cell and the cell's  
        //----right neighbor. Save this right common ancestor and its level for 
        //----determining the right-bottom vertex.
        unsigned int level = cell->level;
        diff = xLocCode ^ xRightLocCode;
        commonRight = cell;
        QT_GET_COMMON_ANCESTOR(commonRight,level,diff);
        rightLevel = level;
        

        //----Follow the branching patterns of the locational codes downward from 
        //----the smallest common ancestor to the right leaf cell vertex neighbor
        *rVtxNbr = commonRight;
        level--;
        QT_TRAVERSE_TO_LEVEL(*rVtxNbr,level,xRightLocCode,cell->yLocCode,0);
    } 


    //----Determine the bottom leaf cell vertex neighbor 
    if (noBottom) *bVtxNbr = 0;
    else {
        //----Determine the smallest common ancestor of the cell and the cell's
        //----bottom neighbor. Save this bottom common ancestor and its level for
        //----determining the right-bottom vertex.
        unsigned int level = cell->level;
        diff = yLocCode ^ yBottomLocCode;
        commonBottom = cell;
        QT_GET_COMMON_ANCESTOR(commonBottom,level,diff);
        bottomLevel = level;
        
        
        //----Follow the branching patterns of the locational codes downward from 
        //----the smallest common ancestor to the bottom leaf cell vertex neighbor
        *bVtxNbr = commonBottom;
        level--;
        QT_TRAVERSE_TO_LEVEL(*bVtxNbr,level,xLocCode,yBottomLocCode,0);
    }


    //----Determine the right-bottom leaf cell vertex neighbor 
    if (noRight || noBottom) *rbVtxNbr = 0;
    else {
        //----Follow the branching patterns of the locational codes downward from 
        //----the smallest common ancestor (the larger of the right common ancestor 
        //----and the bottom common ancestor) to the right-bottom leaf cell vertex 
        //----neighbor
        if (rightLevel >= bottomLevel) {
            *rbVtxNbr = commonRight;
            rightLevel--;
            QT_TRAVERSE_TO_LEVEL(*rbVtxNbr,rightLevel,xRightLocCode,yBottomLocCode,0);

        } else {
            *rbVtxNbr = commonBottom;
            bottomLevel--;
            QT_TRAVERSE_TO_LEVEL(*rbVtxNbr,bottomLevel,xRightLocCode,yBottomLocCode,0);
        }
    }
}
