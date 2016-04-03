/*
 The header/source code in C for traversal of
                            Binary Space Partitioning Trees

 Appendix to the paper:
          Fast Robust Traversal Algorithms for BSP trees

 Authors: Vlastimil Havran, Tomas Kopal, Jiri Bittner, Jiri Zara
 E-mail: havran@fel.cvut.cz, kopal@fel.cvut.cz,
         bittner@fel.cvut.cz, zara@fel.cvut.cz

 Appeared in Journal of Graphics Tools, No.4, 1998
*/

/* -------------------------------------------------------------------
    Header file
*/


/* forward declarations */

/* the representation of an object*/
struct Object3D;
/* the container (list) of objects */
struct ObjectContainer;
/* the axis aligned box in the scene */
struct AxisAlignedBox;

/* this represents the ray */
struct Ray
{
  float loc_x, loc_y, loc_z; /* the coordinates of the origin of a ray */
  float dir_x, dir_y, dir_z; /* the coordinates of the direction of the ray */
}


/* ===================================================================
   Definition of axes of spatial subdivision
*/

enum Axes { X_axis = 0, Y_axis = 1, Z_axis = 2, No_axis = 3};


/* ====================================================================
   Representation of one node of BSP tree
*/

struct BSPNode {
  BSPNode *left; /* pointer to the left child */
  union {
    ObjectContainer* objlist; /* object list for a leaf */
    BSPNode *right; /* pointer to the right child */
  };
  /* the node can contain the list of primitives, when the node is not 
     a leaf .. (ternary) BSP = BSP with tagged object list */
  float splitPlane;  /* the position of splitting plane */
  Axes  splitAxis;   /* the axis, where cut is performed */
};

/* the stack item required for the traversal */
struct SStackElem {
  BSPNode *nodep;   /* pointer to the node */
  float    x, y, z; /* the coordinates of the point */
  float    t;       /* the signed distance of the point */
  SStackElem *prev; /* the pointer to the previous item on the stack (trick) */
};

/* the height of the stack for traversal */
#define MAX_HEIGHT 50

/* the representation has one leaf for all empty leaves in BSP tree */
static struct BSPNode emptyLeaf;

/* the representation of the axis aligned box enclosing the whole scene */
static struct AxisAlignedBox rootBox;

/* the initialization of empty leaf */
void Init();

/* Query functions */
int IsEmptyLeaf_(BSPNode *p);
int IsFullLeaf_(BSPNode *p);
int IsLeaf_(BSPNode *p);
/* Information functions */
float GetSplitValue(BSPNode *nodep);
Axes GetSplitAxis(BSPNode *nodep);
BSPNode* GetLeft(BSPNode *nodep);
BSPNode* GetRight(BSPNode *nodep);
ObjectContainer* GetObjList(BSPNode *nodep);

/* For a given ray and rectangular bor returns minimum and maximum
   signed distance corresponding to intersection of the ray with the box
   Implementation can be found e.g. Graphics Gems .......
*/
int GetMinMaxT(AxisAlignedBox *bbox, Ray *ray, float *tmin, float *tmax)

/* Test objects in full leaf p and if finds the closest intersection
   with object so tmin<= t <= tmax, returns the pointer to that
   object, t is returned in tmax, otherwise NULL
*/
Object3D* TestFullLeaf(Ray *ray, BSPNode *p, float *tmin, float *tmax);

/* Finds the closest objects intersected by a given ray
   If there is no such an object, returns NULL
*/
Object3D* FindNearest(Ray *ray, float *t);


/* -------------------------------------------------------------------
   Source file
*/

/* the initialization of empty leaf */
void Init()
{
  emptyLeaf.left = emptyLeaf.right = NULL;
  emptyLeaf.splitPlane = 0.0;
  emptyLeaf.splitAxis = No_axis;
}

/* Query functions */
int IsEmptyLeaf_(BSPNode *p)
{
  return (p == &emptyLeaf);
}

int IsFullLeaf_(BSPNode *p)
{
  return ((p->splitAxis == No_axis) && (p->objlist != NULL));
}

int IsLeaf_(BSPNode *p)
{
  return (p->splitAxis == No_axis);
}


/* Information functions */
float GetSplitValue(BSPNode *nodep)
{
  return nodep->splitPlane;
}

Axes GetSplitAxis(BSPNode *nodep)
{
  return nodep->splitAxis;
}

BSPNode* GetLeft(BSPNode *nodep)
{
  return nodep->left;
}

BSPNode* GetRight(BSPNode *nodep)
{
  return nodep->right;
}
  
ObjectContainer* GetObjList(BSPNode *nodep)
{
  return nodep->objlist;
}

/* Test all objects in the leaf for intersection with ray
   and returns the pointer to closest one if exists
   and passing through parameter returns in tmax
*/
Object3D* TestFullLeaf(Ray *ray, BSPNode *p, float *tmin, float *tmax)
{
  float t; /* signed distance */

  float tminc = tmin;

  Object3D *retObject = NULL; /* pointer to the intersected object */

  Object3D *obj; /* currently tested object in the leaf */
  
  /* iterate the whole list and find out the nearest intersection */
  for (obj = all objects in the node of BSP tree .. GetObjectList(p))
  {
    /* if the intersection really lies in the node */
    if (obj->NearestInt(ray, t, tmax)) {
      if ((t >= tminc) && (t <= tmax)) {
        tmax = t;
        retObject = obj;
      }
    }
  }
  return retObject;
}


/* Finds the closest objects intersected by a given ray, returns the pointer
   to this object. If there is no such an object, returns NULL.
*/
Object3D* FindNearest(Ray *ray, float *t)
{  
  static struct SStackElem stack[MAX_HEIGHT]; /* the stack of elems */
 
  /* signed distances */
  float tdist, tmin, tmax;
  
  /* test if the whole BSP tree is missed by the input ray or not */
  if (!GetMinMaxT(&bbox, ray, &tmin, &tmax))
    return NULL; /* no object can be intersected */

  BSPNode *currNode = root; /* start from the root node */

  /* exit point setting */
  struct SStackElem *extp = &(stack[1]);
  extp->x = ray->loc_x + ray->dir_x * tmax;
  extp->y = ray->loc_y + ray->dir_y * tmax;
  extp->z = ray->loc_z + ray->dir_z * tmax;
  extp->nodep = NULL;
  extp->prev = NULL;
  extp->t = tmax;

  /* entry point setting */
  struct SStackElem *entp = &(stack[0]);
  entp->nodep = NULL;
  entp->prev = NULL;
  /* entry point setting, tmin > 0.0 */
  if (tmin > 0.0)
  { /* a ray with external origin */
    entp->x = ray->loc_x + ray->dir_x * tmin;
    entp->y = ray->loc_y + ray->dir_y * tmin;
    entp->z = ray->loc_z + ray->dir_z * tmin;
    entp->t = tmin;
  }
  else
  { /* a ray with internal origin */
    entp->x = ray->loc_x;
    entp->y = ray->loc_y;
    entp->z = ray->loc_z;
    entp->t = 0.0;
  }
  /* the pointer to the far child if any */
  BSPNode *farChild;
 
  /* loop .. traverse through whole BSP tree */
  while (1)
  {
    /* loop .. until current node is not the leaf */
    while (1)
    {
      /* the position of the splitting plane */
      float splitVal = GetSplitValue(currNode);
      /* decision based on the axis given by splitting plane */
      switch (GetSplitAxis(currNode))
      {
        case X_axis:
        {
          if (entp->x <= splitVal)
          {
            if (extp->x <= splitVal)
            {
              currNode = GetLeft(currNode); /* cases N1,N2,N3,P5,Z2,Z3 */
              continue;
            }
            /* case N4 */
            farChild = GetRight(currNode);
            currNode = GetLeft(currNode);
          }
          else {
            if (splitVal <= extp->x)
            {
              currNode = GetRight(currNode); /* cases P1,P2,P3,N5,Z1 */
              continue; 
            }
            farChild = GetLeft(currNode); /* case P4 */
            currNode = GetRight(currNode);
          }
          /* case N4 or P4 */
          tdist = (splitVal - ray->loc_x) / ray->dir_x;

          struct SStackElem *tmp = extp;
          if (++extp == entp)
            extp++;

          extp->prev = tmp;
          extp->nodep = farChild;
          extp->t = tdist;
          extp->x = splitVal;
          extp->y = ray->loc_y +  tdist * ray->dir_y;
          extp->z = ray->loc_z +  tdist * ray->dir_z;
          continue;
        }
        
        case Y_axis:
        {
          if (entp->y <= splitVal)
          {
            if (extp->y <= splitVal)
            {
              currNode = GetLeft(currNode); /* case N1,N2,N3,P5,Z2,Z3 */
              continue;
            }
            /* case N4 */
            farChild = GetRight(currNode);
            currNode = GetLeft(currNode);
          }
          else {
            if (splitVal <= extp->y)
            {
              currNode = GetRight(currNode); /* case P1,P2,P3,N5 */
              continue; 
            }
            farChild = GetLeft(currNode); /* case P4 */
            currNode = GetRight(currNode);
          }
          /* case N4 or P4 */
          tdist = (splitVal - ray->loc_y) / ray->dir_y;

          struct SStackElem *tmp = extp;
          if (++extp == entp)
            extp++;
          extp->prev = tmp;
          extp->nodep = farChild;
          extp->t = tdist;
          extp->x = ray->loc_x + tdist * ray->dir_x;
          extp->y = splitVal;
          extp->z = ray->loc_z + tdist * ray->dir_z;
          continue;
        }
        
        case Z_axis:
        {
          if (entp->z <= splitVal)
          {
            if (extp->z <= splitVal)
            {
              currNode = GetLeft(currNode); /* case N1,N2,N3,P5,Z2,Z3 */
              continue;
            }
            /* case N4 */
            farChild = GetRight(currNode);
            currNode = GetLeft(currNode);
          }
          else {
            if (splitVal <= extp->z)
            {
              currNode = GetRight(currNode); /* case P1,P2,P3,N5 */
              continue;
            }
            farChild = GetLeft(currNode); /* case P4 */
            currNode = GetRight(currNode);
          }
          /* case N4 or P4 */
          tdist = (splitVal - ray->loc_z) / ray->dir_z;

          struct SStackElem *tmp = extp;
          if (++extp == entp)
            extp++;
          extp->prev = tmp;
          extp->nodep = farChild;
          extp->t = tdist;
          extp->x = ray->loc_x + tdist * ray->dir_x;
          extp->y = ray->loc_y + tdist * ray->dir_y;
          extp->z = splitVal;
          continue;
        }
        /* test objects for intersection */
        case BSPAxes::No_axis: {
          goto TEST_OBJECTS;
        }
      } /* switch */
    } /* while .. current node is not the leaf */

    /* leaf can be empty or full here */
TEST_OBJECTS:
    if (!IsEmptyLeaf_(currNode)) {
      /* leaf contains the references to some objects */
      Object3D *retObject;
      tmax = extp->t;
      /* test the objects in the full leaf against the ray */
      if ((retObject = TestFullLeaf(ray, currNode, entp->t, &tmax)) != NULL) {
        *t = tmax; /* set the signed distance for the intersection point */
        return retObject; /* the first object intersected was found */
      }
    }

TRAVERSE_UP:
    /* pop farChild from the stack */
    /* restore the current values */
    entp = extp;
    currNode = entp->nodep;
    
    if (currNode == NULL) /* test if the whole BSP tree was traversed */
      return NULL; /* no objects found on the path of a ray */

    extp = extp->prev;
  } /* while .. traverse through whole BSP tree */

} /* FindNearest */

----- End of TA-B algorithm -----
