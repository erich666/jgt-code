#include <assert.h>
#include <string.h>
#include "parser.h"
#include "viewer.h"

//*********************
//  CParser
//*********************

CParser::CParser(CViewer *pviewer) 
{
  viewer = pviewer;
  assert(viewer!=NULL);
}

KEY_STRING CParser::DecodeString(char * string) {
  if (strcmp(string, "background:") == 0)  return BACKGROUND;
  else if (strcmp(string, "ambient:") == 0)  return AMBIENT;
  else if (strcmp(string, "image_dimension:") == 0)  return IMAGE_DIMENSION;
  else if (strcmp(string, "object_begin:") == 0) return OBJECT_BEGIN;
  else if (strcmp(string, "object_end:") == 0) return OBJECT_END;
  else if (strcmp(string, "light_begin:") == 0) return LIGHT_BEGIN;
  else if (strcmp(string, "light_end:") == 0)  return LIGHT_END;
  else if (strcmp(string, "view_begin:") == 0)  return VIEW_BEGIN;
  else if (strcmp(string, "view_end:") == 0)  return VIEW_END;
  else if (strcmp(string, "splat_begin:") == 0)  return SPLAT_BEGIN;
  else if (strcmp(string, "splat_end:") == 0)  return SPLAT_END;
  else if (strcmp(string, "/*") == 0)  return COMMENT_BEGIN;
  else if (strcmp(string, "*/") == 0)  return COMMENT_END;
  else if (strcmp(string, "//") == 0)  return SINGLE_COMMENT;
  else if (strcmp(string, "end:") == 0)  return END;

  else if (strcmp(string, "procedure_data:") == 0)  return PROCEDURE_DATA;
  else if (strcmp(string, "data_file:") == 0) return DATA_FILE;
  else if (strcmp(string, "cube:") == 0) return CUBE;
  else if (strcmp(string, "rectangle:") == 0) return RECTANGLE;
  else if (strcmp(string, "sphere:") == 0) return SPHERE;
  else if (strcmp(string, "torus:") == 0) return TORUS;
  else if (strcmp(string, "hipiph:") == 0) return HIPIPH;
  else if (strcmp(string, "fuel:") == 0) return FUEL;
  else if (strcmp(string, "hydrogen:") == 0) return HYDROGEN;
  else if (strcmp(string, "cthead:") == 0) return CTHEAD;
  else if (strcmp(string, "foot:") == 0) return FOOT;
  else if (strcmp(string, "foot2:") == 0) return FOOT2;
  else if (strcmp(string, "uncbrain:") == 0) return UNCBRAIN;
  else if (strcmp(string, "skull:") == 0) return SKULL;
  else if (strcmp(string, "material:") == 0)  return MATERIAL;
  else if (strcmp(string, "color:") == 0)  return COLOR;
  else if (strcmp(string, "scale:") == 0)  return SCALE;
  else if (strcmp(string, "rotate:") == 0)  return ROTATE;
  else if (strcmp(string, "translate:") == 0)  return TRANSLATE;
  else if (strcmp(string, "fireball:") == 0)  return FIRE_BALL;
  else if (strcmp(string, "fireworks:") == 0)  return FIRE_WORKS;
  else if (strcmp(string, "noisesphere:") == 0)  return NOISE_SPHERE;

  else if (strcmp(string, "light_position:") == 0) return LIGHT_POSITION;
  else if (strcmp(string, "intensity:") == 0) return INTENSITY;

  else if (strcmp(string, "type:") == 0)  return TYPE;
  else if (strcmp(string, "eye:") == 0)  return EYE;
  else if (strcmp(string, "coi:") == 0)  return COI;
  else if (strcmp(string, "hither:") == 0)  return HITHER;
  else if (strcmp(string, "yon:") == 0)  return YON;
  else if (strcmp(string, "view_angle:") == 0)  return VIEW_ANGLE;
  else if (strcmp(string, "head_tilt:") == 0)  return HEAD_TILT;
  else if (strcmp(string, "aspect_ratio:") == 0)  return ASPECT_RATIO;

  else if (strcmp(string, "kernel_radius:") == 0)  return KERNEL_RADIUS;
  else if (strcmp(string, "tsplat_size:") == 0)  return TSPLAT_SIZE;
  else if (strcmp(string, "attenuation_factor:") == 0)  return ATTENUATION_FACTOR;
  else if (strcmp(string, "slice_depth:") == 0)  return SLICE_DEPTH;

  else if (strcmp(string, "output:") == 0)  return OUTPUT;

  return UNDEFINED;
}


void CParser::Parse(char * file_name)  {
  cout << "Parsing file " << file_name << " ..." << endl;

  ifstream fin(file_name);  
  char current_str[MAX_KEY_STRING_LENGTH];
  KEY_STRING current_key;


  if (!fin) {
    cerr << "Unable to open/read file " << file_name << endl;
  }

  fin >> current_str;
  current_key = DecodeString(current_str);

  while(current_key != END) {
    switch(current_key) {
      case BACKGROUND:
        ParseBackground(fin);
        break;
      case IMAGE_DIMENSION:
		ParseImageDimension(fin);
        break;
      case AMBIENT:
		ParseAmbient(fin);
        break;
      case OUTPUT:
        ParseOutput(fin);
        break;
      case OBJECT_BEGIN:
        ParseObject(fin);
        break;
      case LIGHT_BEGIN:
        ParseLight(fin);
        break;
      case VIEW_BEGIN:
        ParseView(fin);
        break;
      case SPLAT_BEGIN:
        ParseSplat(fin);
        break;
     case COMMENT_BEGIN:
        ParseComment(fin, false);
        break;
     case SINGLE_COMMENT:
        ParseComment(fin, true);
        break;
      case UNDEFINED:
        cerr << "Error reading file " << file_name << endl;
        cerr << "\tUnable to decode string " << current_str << endl;
        cerr << "\tAborting read." << endl;
        return;
      default:
        cerr << "Error reading file " << file_name << endl;
        cerr << "\tIncorrect format" << endl;
        cerr << "\tRead " << current_str << endl;
        cerr << "\tAborting read." << endl;
        return;
    }
    fin >> current_str;
    current_key = DecodeString(current_str);
  }
  fin.close();
}

void CParser::ParseBackground(ifstream &fin) {
  float r, g, b;
  
  fin >> r >> g >> b;
  viewer->background[0] = r;
  viewer->background[1] = g;
  viewer->background[2] = b;
}

void CParser::ParseOutput(ifstream &fin) {
 
  fin >> viewer->output_file;
}

void CParser::ParseAmbient(ifstream &fin) {
  float a;

  fin >> a;
  viewer->ambient = a;
}

void CParser::ParseImageDimension(ifstream &fin) {
  int cx, cy;

  fin >> cx >> cy;
  viewer->SetViewport(0, 0, cx-1, cy-1);
}

void CParser::ParseObject(ifstream &fin) {
  char current_str[MAX_KEY_STRING_LENGTH];
  KEY_STRING current_key;

  CVolume *vol;
  INT cx, cy, cz;
  float sx = 1,  sy = 1,  sz = 1,
        rx = 0,  ry = 0,  rz = 0,
        tx = 0,  ty = 0,  tz = 0;
 Loop:
  fin >> current_str;
  switch (DecodeString(current_str)) {
    case DATA_FILE:
      fin >> current_str;
      vol = new CVolume;
      vol->ReadVol(current_str);
      break;
    case SPHERE:
      fin >> cx;
       vol = new CDensSphere(cx);
       //vol = new CSphere(cx);
      break;
    case CUBE:
      fin >> cx;
      vol = new CCube(cx);
      break;
    case RECTANGLE:
      fin >> cx >> cy >> cz;
      vol = new CRectangle(cx, cy, cz);
      break;
    case TORUS:
      fin >> cx >> cy >> cz;
      vol = new CTorus(cx, cy, cz);
      break;
   case HIPIPH:
      fin >> current_str;
      vol = new CHipiph(current_str);
      break;
   case FUEL:
      fin >> current_str;
      vol = new CFuel(current_str);
      break;
   case HYDROGEN:
      fin >> current_str;
      vol = new CHydrogen(current_str);
      break;
   case CTHEAD:
      fin >> current_str;
      vol = new CCThead(current_str);
      break;
   case FOOT:
      fin >> current_str;
      vol = new CFoot(current_str);
      break;
   case FOOT2:
      fin >> current_str;
      vol = new CFoot2(current_str);
      break;
   case UNCBRAIN:
      fin >> current_str;
      vol = new CUncbrain(current_str);
      break;
   case SKULL:
      fin >> current_str;
      vol = new CSkull(current_str);
      break;
     case COMMENT_BEGIN:
      ParseComment(fin, false);
      goto Loop;
      break;
    case SINGLE_COMMENT:
      ParseComment(fin, true);
      goto Loop;
      break;
    default:
      cerr << "Error parsing volume object" << endl;
      cerr << "\tIncorrect format" << endl;
      cerr << "\tRead " << current_str << endl;
      cerr << "\tAborting." << endl;
      return;
  }
  
  fin >> current_str;
  current_key = DecodeString(current_str);

  while (current_key != OBJECT_END) {
    switch(current_key) {
      case MATERIAL:
        float ka, kd, ks, kn;
        fin >> ka >> kd >> ks >> kn;
        vol->SetMaterial(ka, kd, ks, kn);
        break;
      case COLOR:
        float r, g, b;
        fin >> r >> g >> b;
        vol->SetColor(r, g, b);
        break;
      case SCALE:
        fin >> sx >> sy >> sz;
        break;
      case ROTATE:
        fin >> rx >> ry >> rz;
        break;
      case TRANSLATE:
        fin >> tx >> ty >> tz;
        break;
     case COMMENT_BEGIN:
        ParseComment(fin, false);
        break;
     case SINGLE_COMMENT:
        ParseComment(fin, true);
        break;
      default:
        cerr << "Error parsing volume object" << endl;
        cerr << "\tIncorrect format" << endl;
        cerr << "\tRead " << current_str << endl;
        cerr << "\tAborting." << endl;
        return;
    } // switch					 
    fin >> current_str;
    current_key = DecodeString(current_str);
  } // while
  
  vol->SetVolume2World(sx, sy, sz, rx, ry, rz, tx, ty, tz);
  viewer->volume = vol;
}

void CParser::ParseLight(ifstream &fin) {
  char current_str[MAX_KEY_STRING_LENGTH];
  KEY_STRING current_key;
  float x, y, z;
    
  fin >> current_str;
  current_key = DecodeString(current_str);
  
  CLight *light = new CLight;

  while (current_key != LIGHT_END) {
    switch(current_key) {
      case LIGHT_POSITION:
        fin >> x >> y >> z;
        light->SetPosition(VECTOR3(x, y, z));
        break;
      case INTENSITY:
        float i;
        fin >> i;
        light->SetIntensity(i);
        break;
     case COMMENT_BEGIN:
        ParseComment(fin, false);
        break;
     case SINGLE_COMMENT:
        ParseComment(fin, true);
        break;
      default:
        cerr << "Error parsing light" << endl;
        cerr << "\tIncorrect format" << endl;
        cerr << "\tRead " << current_str << endl;
        cerr << "\tAborting." << endl;
        return;
    } // switch					 
    fin >> current_str;
    current_key = DecodeString(current_str);
  } // while

  viewer->light = light;
}

void CParser::ParseView(ifstream &fin) {
  char current_str[MAX_KEY_STRING_LENGTH];
  KEY_STRING current_key;
  VIEW3D_INFO *view_info = &viewer->view_info;


  while (1) {
	fin >> current_str;
	current_key = DecodeString(current_str);
	if (current_key == COMMENT_BEGIN)
	  ParseComment(fin, false);
	else if (current_key == SINGLE_COMMENT)
	  ParseComment(fin, true);
	else
	  break;
  }

  if (current_key == TYPE) {
    fin >> current_str;
    if (strcmp(current_str, "parallel") == 0) {
	  view_info->type = PARALLEL;
	  fin >> view_info->left;
	  fin >> view_info->right;
	  fin >> view_info->bottom;
	  fin >> view_info->top;
    } else if (strcmp(current_str, "perspective") == 0) {
	  view_info->type = PERSPECTIVE;
	}
	else{
      cerr << "Unrecognizable projection type " << current_str << endl;
      cerr << "\tAborting." << endl;
      return;
    }
    fin >> current_str;
	current_key = DecodeString(current_str);
  }
  else{
	  view_info->type = PERSPECTIVE;
  }

  while (current_key != VIEW_END) {
    switch (current_key) {
      case EYE:
        float ex, ey, ez;
        fin >> ex >> ey >> ez;
        view_info->eye = VECTOR3(ex, ey, ez);
        break;
      case COI:
        float cx, cy, cz;
        fin >> cx >> cy >> cz;
        view_info->coi = VECTOR3(cx, cy, cz);
        break;
      case HITHER:
        fin >> view_info->hither;
        break;
      case YON:
        fin >> view_info->yon;
        break;
      case VIEW_ANGLE:
        fin >> view_info->view_angle;
        break;
      case HEAD_TILT:
        fin >> view_info->head_tilt;
        break;
      case ASPECT_RATIO:
        fin >> view_info->aspect_ratio;
        break;
     case COMMENT_BEGIN:
        ParseComment(fin, false);
        break;
     case SINGLE_COMMENT:
        ParseComment(fin, true);
        break;
      default:
        cerr << "Error parsing view" << endl;
        cerr << "\tIncorrect format" << endl;
        cerr << "\tRead " << current_str << endl;
        cerr << "\tAborting." << endl;
        return;
        break;
    }
    fin >> current_str;
    current_key = DecodeString(current_str);
  }

  view_info->image_plane = view_info->hither;
}

void CParser::ParseSplat(ifstream &fin) {
  char current_str[MAX_KEY_STRING_LENGTH];
  KEY_STRING current_key;
  SPLAT_INFO *splat_info= &viewer->splat_info;

  fin >> current_str;
  current_key = DecodeString(current_str);

  while (current_key != SPLAT_END) { 
    switch (current_key) {
      case KERNEL_RADIUS:
        fin >> splat_info->kernel_radius;
        break;
      case ATTENUATION_FACTOR:
        fin >> splat_info->sigma;
        break;
      case TSPLAT_SIZE:
        fin >> splat_info->tsplat_size;
        break;
      case SLICE_DEPTH:
        fin >> splat_info->slice_depth;
        break;
     case COMMENT_BEGIN:
        ParseComment(fin, false);
        break;
     case SINGLE_COMMENT:
        ParseComment(fin, true);
        break;
      default:
        cerr << "Error parsing splatter" << endl;
        cerr << "\tIncorrect format" << endl;
        cerr << "\tRead " << current_str << endl;
        cerr << "\tAborting." << endl;
        return;
        break;
    }
    fin >> current_str;
    current_key = DecodeString(current_str);
  }
}

void CParser::ParseComment(ifstream &fin, bool bSingle)  {
  char current_str[MAX_KEY_STRING_LENGTH];

  if (bSingle){
    fin.getline(current_str, sizeof current_str);
  }
  else {
    fin >> current_str;

    while (DecodeString(current_str) != COMMENT_END)  {
      fin >> current_str;
    }
  }
}


