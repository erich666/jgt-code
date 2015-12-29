#ifndef _MOVIEMAKER_H
#define _MOVIEMAKER_H

//  ===============================================
//  MovieMaker class definition.
//  ===============================================

#include <windows.h>
#include <vfw.h>


#define TEXT_HEIGHT	20
#define AVIIF_KEYFRAME	0x00000010L // this frame is a key frame.
#define BUFSIZE 260

class MovieMaker {
private:
    char fname[64];
    int width;
    int height;

  	AVISTREAMINFO strhdr;
	PAVIFILE pfile;
	PAVISTREAM ps;
	PAVISTREAM psCompressed;
	PAVISTREAM psText;
	AVICOMPRESSOPTIONS opts;
	AVICOMPRESSOPTIONS FAR * aopts[1];
	DWORD dwTextFormat;
	char szText[BUFSIZE];
	int nFrames;
	int estFramesPerSecond;
	bool bOK;
	int ready;

	int Snap();
	void PrepareForCapture();

public:
    MovieMaker();
    ~MovieMaker();

    inline bool IsOK() const { return bOK; };
    void StartCapture(const char *name, int framesPerSecond=30 );
	int AddCurrentFrame() { return Snap(); }
    void EndCapture();
};

#endif
