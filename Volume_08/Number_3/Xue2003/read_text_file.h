#ifndef READ_TEXT_FILE_H__CONVENIENCE
#define READ_TEXT_FILE_H__CONVENIENCE

#include <data_path.h>

data_path get_png_path();
void set_text_path(const data_path & newpath);

char * read_text_file(const char * filename);

#endif