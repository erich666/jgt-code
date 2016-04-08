#ifndef __PARSER_H
#define __PARSER_H

#include <fstream>
#include "Datatype.h"

using namespace std;

typedef enum {
  /* BLOCK STRINGS */
  IMAGE_DIMENSION,
  AMBIENT,
  BACKGROUND,
  HYPERTEXTURE,
  ILLUMINATE,
  OBJECT_BEGIN,
  OBJECT_END,
  LIGHT_BEGIN,
  LIGHT_END,
  VIEW_BEGIN,
  VIEW_END,
  SPLAT_BEGIN,
  SPLAT_END,
  COMMENT_BEGIN,
  COMMENT_END,
  SINGLE_COMMENT,
  END,

  /* OBJECT STRINGS */
  DATA_FILE,
  PROCEDURE_DATA,
  SPHERE,
  CUBE,
  RECTANGLE,
  TORUS,
  HIPIPH,
  FUEL,
  HYDROGEN,
  CTHEAD,
  UNCBRAIN,
  FOOT,
  FOOT2,
  SKULL,
  MATERIAL,
  COLOR,
  SCALE,
  ROTATE,
  TRANSLATE,
  FIRE_BALL,
  FIRE_WORKS,
  NOISE_SPHERE,

  /* LIGHT STRINGS */
  LIGHT_POSITION,
  INTENSITY,

  /* VIEW_INFO STRINGS */
  TYPE,
  EYE,
  COI,
  HITHER,
  YON,
  VIEW_ANGLE,
  HEAD_TILT,
  ASPECT_RATIO,

  /* TSPLAT_INFO STRINGS */
  KERNEL_RADIUS,
  TSPLAT_SIZE,
  ATTENUATION_FACTOR,
  SLICE_DEPTH,

  /* string for output file */
  OUTPUT,

  /* error string */
  UNDEFINED

} KEY_STRING;

const int MAX_KEY_STRING_LENGTH = 100;

class CViewer;

class CParser {
 public:
  CParser(CViewer* pviewer);

 public:
  void Parse(char* file_name);

 private:
  KEY_STRING DecodeString(char *string);
  void ParseBackground(ifstream &fin);
  void ParseImageDimension(ifstream &fin);
  void ParseAmbient(ifstream &fin);
  void ParseObject(ifstream &fin);
  void ParseLight(ifstream &fin);
  void ParseView(ifstream &fin);
  void ParseSplat(ifstream &fin);
  void ParseIntTab(ifstream &fin);
  void ParseComment(ifstream &fin, bool bSingle);
  void ParseOutput(ifstream &fin);
  
 private:
  CViewer *viewer;
};

#endif // __PARSER_H
