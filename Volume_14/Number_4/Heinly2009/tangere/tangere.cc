
/*
 *  Copyright 2009, 2010 Grove City College
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifdef USE_CHUD
#include <CHUD/CHUD.h>
#endif // USE_CHUD

#ifdef USE_SATURN
#include <Saturn.h>
#endif // USE_SATURN

#include <string>
using std::string;

#include <Common/rtCore/RGB.t>
using rtCoreF::RGB;

#include <tangere/Flags.h>
#include <tangere/Options.h>
#include <tangere/Renderer.t>
#include <tangere/Scene.t>
#include <tangere/Triangle.t>

using tangere::Options;

void usage(int);
void parseCmdLn(Options&, int, char*[]);

void renderAll();

int main(int argc, char* argv[])
{
#if defined(RENDER_ALL)
  renderAll();
  return 0;
#endif // defined(RENDER_ALL)

  Options opt;
  parseCmdLn(opt, argc, argv);

  if (opt.iInput)
  {
    ////////////////////////////////////////////////////////////////////////////
    // Process .int file

    Output(endl);
    Output("--------------------" << endl);
    Output("  Loading \".int\"" << endl);
    Output("--------------------" << endl);
    Output(endl);
    Output("  Loading \"" << opt.bname << ".int\"...");

    tangereI::Scene sceneI;
    sceneI.read(opt);

    Output("  done" << endl);
    Output(endl);
    Output("------------------------------------------" << endl);
    Output("  Generating image with integer renderer" << endl);
    Output("------------------------------------------" << endl);

    tangereI::Renderer rendererI(&sceneI, opt);
    rendererI.render();

    Output(endl);

    return 0;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Load geometry

  Output(endl);
  Output("--------------------" << endl);
  Output("  Loading geometry" << endl);
  Output("--------------------" << endl);
  Output(endl);

  FileIO::Scene ioScene(&opt);
  
  if (opt.iWrite != "")
  {
    Output(endl);

    tangereI::Scene sceneI(&ioScene);
    sceneI.write(opt);

    return 0;
  }

#ifdef USE_CHUD
  chudInitialize();
  chudAcquireRemoteAccess();
#endif // USE_CHUD

#ifdef USE_SATURN
  initSaturn("/Users/cgribble/vissim/build-gnu.saturn/");
#endif // USE_SATURN

  //////////////////////////////////////////////////////////////////////////////
  // Floating-point renderer

  Output(endl);
  Output(endl);
  Output("-------------------------------------------------" << endl);
  Output("  Generating image with floating-point renderer" << endl);
  Output("-------------------------------------------------" << endl);
  Output(endl);

  tangereF::Scene    sceneF(&ioScene);
  tangereF::Renderer rendererF(&sceneF, opt);

  rendererF.render();

  Output(endl);
  Output(endl);

  //////////////////////////////////////////////////////////////////////////////
  // Integer renderer

  Output("------------------------------------------" << endl);
  Output("  Generating image with integer renderer" << endl);
  Output("------------------------------------------" << endl);
  Output(endl);

  tangereI::Scene    sceneI(&ioScene);
  tangereI::Renderer rendererI(&sceneI, opt);

  rendererI.render();

  Output(endl);

#ifdef USE_CHUD
  chudReleaseRemoteAccess();
#endif // USE_CHUD

  return 0;
}

void usage(int ec)
{
  Output(endl);
  Output("usage:  tangere [options] <scene>" << endl);
  Output("options:" << endl);
  Output("  -ground <f> <f> <f>   ground emission" << endl);
  Output("  --help | -h           print this message and exit" << endl);
  Output("  -o <s>                output file basename" << endl);
  Output("  -res <i>x<i>          image resolution" << endl);
  Output("  -sky <f> <f> <f>      sky emission" << endl);
  Output("  -spp <i>              samples per pixel" << endl);
  Output("  -threshold <i>        bvh leaf creation threshold" << endl);
  Output("  -write <s>            write \".int\" file" << endl);
  Output(endl);

  exit(ec);
}

void parseCmdLn(Options& opt, int argc, char* argv[])
{
#if defined(USE_HARDCODED_PATH)
  opt.fname = "..\\..\\scenes\\rtrt_luxjr.txt";
  //opt.fname = "rtrt.int";
  string ext = opt.fname.substr(opt.fname.length()-3);
  opt.thold = 1;
  //opt.iWrite = "rtrt.int";
  if (ext == "txt" || ext == "int")
  {
    size_t b, e;
#if defined(WIN32)
    b         = opt.fname.rfind("\\") + 1;
#else
    b         = opt.fname.rfind("/") + 1;
#endif // defined(WIN32)
    e         = opt.fname.rfind(".");
    opt.path  = opt.fname.substr(0, b);

    if (opt.bname == "")
      opt.bname = opt.fname.substr(b, e-b);

    if (ext == "int")
      opt.iInput = true;
  }
  return;
#endif // defined(USE_HARDCODE_PATH)

  if (argc < 2)
    usage(-1);

  for (int i = 1; i < argc; ++i)
  {
    string arg(argv[i]);
    uint   nremain = argc - i - 1;

    if (arg == "-ground")
    {
      if (nremain < 3)
      {
        Error("\"-ground\" expects 3 arguments" << endl);
        usage(-1);
      }
      
      const float x = float(atof(argv[i+1]));
      const float y = float(atof(argv[i+2]));
      const float z = float(atof(argv[i+3]));

      opt.grnd = RGB(x, y, z);

      i += 3;
    }
    else if (arg == "--help" || arg == "-h")
    {
      usage(0);
    }
    else if (arg == "-o")
    {
      if (nremain < 1)
      {
        Error("\"-o\" expects 1 argument" << endl);
        usage(-1);
      }
      
      opt.bname = string(argv[++i]);
    }
    else if (arg == "-res")
    {
      if (nremain < 1)
      {
        Error("\"-res\" expects 1 argument" << endl);
        usage(-1);
      }

      const string res  = argv[++i];
      const size_t pos  = res.find('x');
      if (pos == 0 || pos == res.length()-1 || pos == string::npos)
      {
        Error("Invalid resolution:  " << res << endl);
        usage(-1);
      }

      const string xres = res.substr(0, pos);
      const string yres = res.substr(pos+1, res.length());

      opt.xres = atoi(xres.c_str());
      opt.yres = atoi(yres.c_str());
    }
    else if (arg == "-sky")
    {
      if (nremain < 3)
      {
        Error("\"-sky\" expects 3 arguments" << endl);
        usage(-1);
      }
      
      const float x = float(atof(argv[i+1]));
      const float y = float(atof(argv[i+2]));
      const float z = float(atof(argv[i+3]));

      opt.sky = RGB(x, y, z);

      i += 3;
    }
    else if (arg == "-spp")
    {
      if (nremain < 1)
      {
        Error("\"-spp\" expects 1 argument" << endl);
        usage(-1);
      }
      
      opt.nspp = atoi(argv[++i]);
    }
    else if (arg == "-threshold")
    {
      if (nremain < 1)
      {
        Error("\"-threshold\" expects 1 argument" << endl);
        usage(-1);
      }
      
      opt.thold = atoi(argv[++i]);
    }
    else if (arg == "-write")
    {
      if (nremain < 1)
      {
        Error("\"-write\" expects 1 argument" << endl);
        usage(-1);
      }
      
      opt.iWrite = string(argv[++i]);
    }
    else
    {
      const uint len = arg.length();
      if (len > 3)
      {
        string ext = arg.substr(len-3);
        if (ext == "txt" || ext == "int")
        {
          size_t b, e;

          opt.fname = string(argv[i]);
#if defined(WIN32)
          b         = opt.fname.rfind("\\") + 1;
#else
          b         = opt.fname.rfind("/") + 1;
#endif // defined(WIN32)
          e         = opt.fname.rfind(".");
          opt.path  = opt.fname.substr(0, b);

          if (opt.bname == "")
            opt.bname = opt.fname.substr(b, e-b);

          if (ext == "int")
            opt.iInput = true;

          continue;
        }
      }

      Error("Unrecognized argument:  " << arg << endl);
      usage(-1);
    }
  }
}

void renderAll()
{
  vector<string> files;
  files.push_back("..\\..\\scenes\\ben_luxjr.txt");
  files.push_back("..\\..\\scenes\\cbox_luxjr.txt");
  files.push_back("..\\..\\scenes\\conf_luxjr.txt");
  files.push_back("..\\..\\scenes\\CubeCave_luxjr.txt");
  files.push_back("..\\..\\scenes\\fairy_luxjr.txt");
  files.push_back("..\\..\\scenes\\hand_luxjr.txt");
  files.push_back("..\\..\\scenes\\kala_luxjr.txt");
  files.push_back("..\\..\\scenes\\poolhall_luxjr.txt");
  files.push_back("..\\..\\scenes\\rtrt_luxjr.txt");
  files.push_back("..\\..\\scenes\\sibenik_luxjr.txt");
  files.push_back("..\\..\\scenes\\spheres_luxjr.txt");
  files.push_back("..\\..\\scenes\\sponza_luxjr.txt");
  files.push_back("..\\..\\scenes\\tank_luxjr.txt");
  files.push_back("..\\..\\scenes\\Toasters_luxjr.txt");
  files.push_back("..\\..\\scenes\\wooddoll_luxjr.txt");

  for (uint i = 0; i < files.size(); ++i)
  {
    ////////////////////////////////////////////////////////////////////////////
    // Loading options

    Output(endl);
    Output("--------------------" << endl);
    Output("  Loading options" << endl);
    Output("--------------------" << endl);
    Output(endl);

    Output("  Loading \"" << files[i] << "\"...");

    Options opt;
    opt.fname = files[i];
    string ext = opt.fname.substr(opt.fname.length()-3);
    opt.thold = 1;
    opt.xres = 512;
    opt.yres = 512;

    size_t b, e;
#ifdef WIN32
    b         = opt.fname.rfind("\\") + 1;
#else
    b         = opt.fname.rfind("/") + 1;
#endif
    e         = opt.fname.rfind(".");
    opt.path  = opt.fname.substr(0, b);
    opt.bname = opt.fname.substr(b, e-b);

    Output("  done" << endl);
    Output(endl);

    ////////////////////////////////////////////////////////////////////////////
    // Load geometry

    Output(endl);
    Output("--------------------" << endl);
    Output("  Loading geometry" << endl);
    Output("--------------------" << endl);
    Output(endl);

    FileIO::Scene ioScene(&opt);

    ////////////////////////////////////////////////////////////////////////////
    // Floating-point renderer

    Output(endl);
    Output(endl);
    Output("-------------------------------------------------" << endl);
    Output("  Generating image with floating-point renderer" << endl);
    Output("-------------------------------------------------" << endl);
    Output(endl);

    tangereF::Scene    sceneF(&ioScene);
    tangereF::Renderer rendererF(&sceneF, opt);
    rendererF.render();

    Output(endl);
    Output(endl);

    ////////////////////////////////////////////////////////////////////////////
    // Integer renderer

    Output("------------------------------------------" << endl);
    Output("  Generating image with integer renderer" << endl);
    Output("------------------------------------------" << endl);
    Output(endl);

    tangere::Triangle<int>::resetStatics();
    tangereI::Scene    sceneI(&ioScene);
    tangereI::Renderer rendererI(&sceneI, opt);
    rendererI.render();

    Output(endl);
  }
}
