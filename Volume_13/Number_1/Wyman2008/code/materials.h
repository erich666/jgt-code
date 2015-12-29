/*****************************************
** materials.h                          **
** -----------                          **
**                                      **
** Contains material types available in **
** the materials.c code as well as the  **
** prototype for the function which     **
** sets up one of the materials in the  **
** material.c                           **
**                                      **
** Chris Wyman (2/15/2000)              **
*****************************************/

/* Material Types */

#define MAT_BRASS              0
#define MAT_BRONZE             1
#define MAT_POLISHED_BRASS     2
#define MAT_CHROME             3
#define MAT_COPPER             4
#define MAT_POLISHED_COPPER    5
#define MAT_GOLD               6
#define MAT_POLISHED_GOLD      7
#define MAT_PEWTER             8
#define MAT_SILVER             9
#define MAT_POLISHED_SILVER   10
#define MAT_EMERALD           11
#define MAT_JADE              12
#define MAT_OBSIDIAN          13
#define MAT_PEARL             14
#define MAT_RUBY              15
#define MAT_TURQUOISE         16
#define MAT_BLACK_PLASTIC     17
#define MAT_BLACK_RUBBER      18


/* prototype for material setting function 
**
** "num" should be set to one of the material
** types defined above.
**
** face should be GL_FRONT, GL_BACK, or GL_FRONT_AND_BACK
** to determine which face of the polygon should have these
** properties.
*/
void SetCurrentMaterial( int face, int num );


/* 
** same as SetCurrentMaterial() except it allows
** modification of the preset materials.
*/
void SetCurrentMaterialPlus( int face, int num, GLdouble amb[3], 
			     GLdouble dif[3], GLdouble spec[3] );


/* prototype for material setting function 
** 
** This function sets the material properties based upon an
** RGB color value.  This is useful if you want to use an
** object with RGB values at vertices and don't want to
** get rid of them and use a constant material for the entire
** object or go thru the labor of assigning a good material
** to each point.
**
** pretty naive, no emission, 50 specular.  ambient ranges
** between [0,0.25], diffuse between [0,0.5] and specular
** between [0,1.0] as r/g/b range over [0,1.0]
**
** face should be GL_FRONT, GL_BACK, or GL_FRONT_AND_BACK
** to determine which face of the polygon should have these
** properties.
*/
void SetCurrentMaterialToColor( int face, float r, float g, float b );


/* sets the current material to white --
** ambient, diffuse, emissive, specular are all {1,1,1,1}
*/
void SetCurrentMaterialToWhite( int face );
