
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <string>


#include "argparse/argparse.hpp"

#include "moab/Core.hpp"
#include "moab/Range.hpp"

#include "gprt.h"

#include "sharedCode.h"
#include "mb_util.hpp"

#define LOG(message)                                            \
  std::cout << GPRT_TERMINAL_BLUE;                               \
  std::cout << "#gprt.sample(main): " << message << std::endl;   \
  std::cout << GPRT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << GPRT_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#gprt.sample(main): " << message << std::endl;   \
  std::cout << GPRT_TERMINAL_DEFAULT;

extern GPRTProgram dbl_deviceCode;

#define MOAB_CHECK_ERROR(EC) if (EC != moab::MB_SUCCESS) return 1;

void render();

// initial image resolution
const int2 fbSize = {2560, 1440};

int main(int argc, char** argv) {

  argparse::ArgumentParser args("GPRT H5M READER");

  args.add_argument("filename")
      .help("Path to the DAGMC file to view");
  args.add_argument("--volumes")
      .help("Subset of volume IDs to visualize")
      .nargs(argparse::nargs_pattern::any)
      .default_value(std::vector<double>())
      .scan<'i', int>();
  args.add_argument("--type")
      .help("Floating point primitive representation (one of 'float' or 'double'")
      .default_value("float");

  try {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(0);
  }

  auto filename = args.get<std::string>("filename");

  auto type = args.get<std::string>("type");
  bool useFloats = (type == "float");

  if (type != "double" && type != "float") {
    std::cerr << "Error: primitive representation must be set to either 'float' or 'double'." << std::endl;
    std::exit(1);
  }

  std::shared_ptr<moab::Core> mbi = std::make_shared<moab::Core>();
  moab::ErrorCode rval;

  std::cout << "Loading " << filename << "..." << std::endl;
  rval = mbi->load_file(filename.c_str());
  MOAB_CHECK_ERROR(rval);

  std::vector<int> volumes;
  if (args.is_used("--volumes")) volumes = args.get<std::vector<int>>("volumes");

  // start up GPRT
  gprtRequestWindow(fbSize.x, fbSize.y, "S01 Single Triangle");
  GPRTContext context = gprtContextCreate(nullptr, 1);
  GPRTModule module = gprtModuleCreate(context, dbl_deviceCode);
  // -------------------------------------------------------
  // Setup programs and geometry types
  // -------------------------------------------------------

  // For double precision triangles
  auto DPTriangleType = gprtGeomTypeCreate<DPTriangleData>(context,
                        GPRT_AABBS);
  gprtGeomTypeSetClosestHitProg(DPTriangleType,0,
                                module,"DPTriangle");
  gprtGeomTypeSetIntersectionProg(DPTriangleType,0,
                                  module,"DPTrianglePlucker");

  // For single precision triangles
  auto SPTriangleType = gprtGeomTypeCreate<SPTriangleData>(context, GPRT_TRIANGLES);
  gprtGeomTypeSetClosestHitProg(SPTriangleType,0, module,"SPTriangle");

  // For launching double precision rays
  GPRTRayGenOf<RayGenData> DPRayGen = nullptr;
  if (!useFloats) DPRayGen = gprtRayGenCreate<RayGenData>(context, module, "DPRayGen");

  // For launching single precision rays
  GPRTRayGenOf<RayGenData> SPRayGen = nullptr;
  if (useFloats) SPRayGen = gprtRayGenCreate<RayGenData>(context, module, "SPRayGen");

  // What to do if a ray misses
  GPRTMissOf<MissProgData> miss
    = gprtMissCreate<MissProgData>(context,module,"miss");

  gprtBuildPipeline(context);

  // create colors for each volume
  create_volume_colors(mbi.get(), volumes);

  // compute centroid to look at
  auto bbox = bounding_box(mbi.get());

  // create geometries
  std::vector<SPTriangleSurface> SPTriSurfs;
  std::vector<DPTriangleSurface> DPTriSurfs;

  std::vector<GPRTGeomOf<SPTriangleData>> SPgeoms;
  std::vector<GPRTGeomOf<DPTriangleData>> DPgeoms;

  GPRTAccel blas;

  if (useFloats) {
    SPTriSurfs = setup_surfaces<SPTriangleSurface, SPTriangleData>(context, mbi.get(), SPTriangleType);
    for (auto& ts : SPTriSurfs) {
      ts.set_buffers();
      SPgeoms.push_back(ts.triangle_geom_s);
    }
    blas = gprtTrianglesAccelCreate(context, SPgeoms.size(), SPgeoms.data());
    gprtAccelBuild(context, blas);
  } else {
    DPTriSurfs = setup_surfaces<DPTriangleSurface, DPTriangleData>(context, mbi.get(), DPTriangleType);

    for (auto& ts : DPTriSurfs) ts.aabbs(context, module);

    for (const auto& ts : DPTriSurfs) DPgeoms.push_back(ts.triangle_geom_s);
    blas = gprtAABBAccelCreate(context, DPgeoms.size(), DPgeoms.data());
    gprtAccelBuild(context, blas);
  }

  // empty the mesh library's copy (good riddance)
  rval = mbi->delete_mesh();
  MOAB_CHECK_ERROR(rval);
  mbi.reset();

  if (SPTriSurfs.size() == 0 && DPTriSurfs.size() == 0) {
    std::cerr << "No surfaces visible" << std::endl;
    std::exit(1);
  }

  double3 aabbCentroid = bbox.first + (bbox.second - bbox.first) * 0.5;

  float3 lookFrom = float3(float(aabbCentroid.x), float(aabbCentroid.y)  - 50.f, float(aabbCentroid.z));
  float3 lookAt = float3(float(aabbCentroid.x),float(aabbCentroid.y),float(aabbCentroid.z));
  float3 lookUp = {0.f,0.f,-1.f};
  float cosFovy = 0.66f;

  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  GPRTAccel world = gprtInstanceAccelCreate(context, 1, &blas);
  gprtAccelBuild(context, world);

  // ----------- set variables  ----------------------------
  auto missData = gprtMissGetPointer(miss);
  missData->color0 = float3(0.1f, 0.1f, 0.1f);
  missData->color1 = float3(0.f, 0.f, 0.f);

  // ----------- set raygen variables  ----------------------------
  auto frameBuffer = gprtDeviceBufferCreate<uint32_t>(context, fbSize.x*fbSize.y);

  // need this to communicate double precision rays to intersection program
  // ray origin xyz + tmin, then ray direction xyz + tmax
  GPRTBufferOf<double> doubleRayBuffer = nullptr;
  RayGenData* rayGenData;
  if (useFloats) {
    rayGenData = gprtRayGenGetPointer(SPRayGen);
  }
  else {
    rayGenData = gprtRayGenGetPointer(DPRayGen);
    doubleRayBuffer = gprtDeviceBufferCreate<double>(context,fbSize.x*fbSize.y*8);
    rayGenData->dpRays = gprtBufferGetHandle(doubleRayBuffer);


    for (auto& ts : DPTriSurfs) {
      auto dpGeomData = gprtGeomGetPointer(ts.triangle_geom_s);
      dpGeomData->fbSize = fbSize;
      dpGeomData->dpRays = gprtBufferGetHandle(doubleRayBuffer);
    }
  }
  rayGenData->fbPtr = gprtBufferGetHandle(frameBuffer);
  rayGenData->fbSize = fbSize;
  rayGenData->world = gprtAccelGetHandle(world);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  gprtBuildPipeline(context);
  gprtBuildShaderBindingTable(context, GPRT_SBT_ALL);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  LOG("launching ...");

  bool firstFrame = true;
  double xpos = 0.f, ypos = 0.f;
  double lastxpos, lastypos;
  while (!gprtWindowShouldClose(context))
  {
    float speed = .001f;
    lastxpos = xpos;
    lastypos = ypos;
    gprtGetCursorPos(context, &xpos, &ypos);
    if (firstFrame) {
      lastxpos = xpos;
      lastypos = ypos;
    }
    int state = gprtGetMouseButton(context, GPRT_MOUSE_BUTTON_LEFT);
    int rstate = gprtGetMouseButton(context, GPRT_MOUSE_BUTTON_RIGHT);

    int w_state = gprtGetKey(context, GPRT_KEY_W);
    int c_state = gprtGetKey(context, GPRT_KEY_C);
    int ctrl_state = gprtGetKey(context, GPRT_KEY_LEFT_CONTROL);

    // close window on Ctrl-W press
    if (w_state && ctrl_state) { break; }
    // close window on Ctrl-C press
    if (c_state && ctrl_state) { break; }

    // If we click the mouse, we should rotate the camera
    if (state == GPRT_PRESS || firstFrame)
    {
      firstFrame = false;
      float4 position = {lookFrom.x, lookFrom.y, lookFrom.z, 1.f};
      float4 pivot = {lookAt.x, lookAt.y, lookAt.z, 1.0};
      #ifndef M_PI
      #define M_PI 3.1415926f
      #endif

      // step 1 : Calculate the amount of rotation given the mouse movement.
      float deltaAngleX = (2 * M_PI / fbSize.x);
      float deltaAngleY = (M_PI / fbSize.y);
      float xAngle = (lastxpos - xpos) * deltaAngleX;
      float yAngle = (lastypos - ypos) * deltaAngleY;

      // step 2: Rotate the camera around the pivot point on the first axis.
      float4x4 rotationMatrixX = rotation_matrix(rotation_quat(lookUp, xAngle));
      position = (mul(rotationMatrixX, (position - pivot))) + pivot;

      // step 3: Rotate the camera around the pivot point on the second axis.
      float3 lookRight = cross(lookUp, normalize(pivot - position).xyz());
      float4x4 rotationMatrixY = rotation_matrix(rotation_quat(lookRight, yAngle));
      lookFrom = ((mul(rotationMatrixY, (position - pivot))) + pivot).xyz();

      // ----------- compute variable values  ------------------
      float3 camera_pos = lookFrom;
      float3 camera_d00
        = normalize(lookAt-lookFrom);
      float aspect = float(fbSize.x) / float(fbSize.y);
      float3 camera_ddu
        = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
      float3 camera_ddv
        = cosFovy * normalize(cross(camera_ddu,camera_d00));
      camera_d00 -= 0.5f * camera_ddu;
      camera_d00 -= 0.5f * camera_ddv;

      // ----------- set variables  ----------------------------
      rayGenData->camera.pos = camera_pos;
      rayGenData->camera.dir_00 = camera_d00;
      rayGenData->camera.dir_du = camera_ddu;
      rayGenData->camera.dir_dv = camera_ddv;
      gprtBuildShaderBindingTable(context, GPRT_SBT_RAYGEN);
    }

    if (rstate == GPRT_PRESS) {
      gprtGetCursorPos(context, &xpos, &ypos);
      float dy = ypos - lastypos;

      float3 view_vec = lookFrom - lookAt;

      if (dy > 0.0) {
        view_vec.x *= 0.95;
        view_vec.y *= 0.95;
        view_vec.z *= 0.95;
      } else if (dy < 0.0) {
        view_vec.x *= 1.05;
        view_vec.y *= 1.05;
        view_vec.z *= 1.05;
      }

      lookFrom = lookAt + view_vec;

      rayGenData->camera.pos = lookFrom;
      gprtBuildShaderBindingTable(context, GPRT_SBT_RAYGEN);
    }

    // Now, trace rays
    gprtBeginProfile(context);

    if (useFloats) {
      gprtRayGenLaunch2D(context,SPRayGen,fbSize.x,fbSize.y);
    }
    else  {
      gprtRayGenLaunch2D(context,DPRayGen,fbSize.x,fbSize.y);
    }
    float ms = gprtEndProfile(context) * 1.e-06;
    std::cout << "RF Time: " << ms << " ms" << std::endl;
    std::cout << "Time per ray: " << ms / (1080 * 720) << " ms" << std::endl;

    // Render results to screen
    gprtBufferPresent(context, frameBuffer);
  }

  // ##################################################################
  // and finally, clean up
  // ##################################################################

  LOG("cleaning up ...");

  // if (singleVertexBuffer) gprtBufferDestroy(singleVertexBuffer);
  // if (doubleVertexBuffer) gprtBufferDestroy(doubleVertexBuffer);
  // gprtBufferDestroy(indexBuffer);
  // if (aabbPositionsBuffer) gprtBufferDestroy(aabbPositionsBuffer);
  gprtBufferDestroy(frameBuffer);
  // if (doubleRayBuffer) gprtBufferDestroy(doubleRayBuffer);
  if (SPRayGen) gprtRayGenDestroy(SPRayGen);
  // if (DPRayGen) gprtRayGenDestroy(DPRayGen);
  gprtMissDestroy(miss);
  gprtAccelDestroy(blas);
  gprtAccelDestroy(world);
  // if (dpGeom) gprtGeomDestroy(dpGeom);
  // for (auto& spGeom : spGeoms) gprtGeomDestroy(spGeom);
  for (auto& g : SPTriSurfs) { g.cleanup(); }
  for (auto& g : DPTriSurfs) { g.cleanup();}
  gprtGeomTypeDestroy(DPTriangleType);
  gprtGeomTypeDestroy(SPTriangleType);
  gprtModuleDestroy(module);
  gprtContextDestroy(context);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");

  return 0;
}


// void render() {
//   // ##################################################################
//   // create a window we can use to display and interact with the image
//   // ##################################################################
//   if (!glfwInit())
//     // Initialization failed
//     throw std::runtime_error("Can't initialize GLFW");

//   auto error_callback = [](int error, const char* description)
//   {
//     fprintf(stderr, "Error: %s\n", description);
//   };
//   glfwSetErrorCallback(error_callback);

//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
//   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
//   GLFWwindow* window = glfwCreateWindow(fbSize.x, fbSize.y,
//     "Int02 Simple AABBs", NULL, NULL);
//   if (!window) throw std::runtime_error("Window or OpenGL context creation failed");
//   glfwMakeContextCurrent(window);

//   // ##################################################################
//   // now that everything is ready: launch it ....
//   // ##################################################################

//   LOG("launching ...");

//   bool firstFrame = true;
//   double xpos = 0.f, ypos = 0.f;
//   double lastxpos, lastypos;
//   while (!glfwWindowShouldClose(window))
//   {
//     float speed = .001f;
//     lastxpos = xpos;
//     lastypos = ypos;
//     glfwGetCursorPos(window, &xpos, &ypos);
//     if (firstFrame) {
//       lastxpos = xpos;
//       lastypos = ypos;
//     }
//     int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
//     int rstate = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

//     int w_state = glfwGetKey(window, GLFW_KEY_W);
//     int c_state = glfwGetKey(window, GLFW_KEY_C);
//     int ctrl_state = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL);

//     // close window on Ctrl-W press
//     if (w_state && ctrl_state) { break; }
//     // close window on Ctrl-C press
//     if (c_state && ctrl_state) { break; }

//     // If we click the mouse, we should rotate the camera
//     if (state == GLFW_PRESS || firstFrame)
//     {
//       firstFrame = false;
//       float4 position = {lookFrom.x, lookFrom.y, lookFrom.z, 1.f};
//       float4 pivot = {lookAt.x, lookAt.y, lookAt.z, 1.0};
//       #define M_PI 3.1415926f

//       // step 1 : Calculate the amount of rotation given the mouse movement.
//       float deltaAngleX = (2 * M_PI / fbSize.x);
//       float deltaAngleY = (M_PI / fbSize.y);
//       float xAngle = (lastxpos - xpos) * deltaAngleX;
//       float yAngle = (lastypos - ypos) * deltaAngleY;

//       // step 2: Rotate the camera around the pivot point on the first axis.
//       float4x4 rotationMatrixX = rotation_matrix(rotation_quat(lookUp, xAngle));
//       position = (mul(rotationMatrixX, (position - pivot))) + pivot;

//       // step 3: Rotate the camera around the pivot point on the second axis.
//       float3 lookRight = cross(lookUp, normalize(pivot - position).xyz());
//       float4x4 rotationMatrixY = rotation_matrix(rotation_quat(lookRight, yAngle));
//       lookFrom = ((mul(rotationMatrixY, (position - pivot))) + pivot).xyz();

//       // ----------- compute variable values  ------------------
//       float3 camera_pos = lookFrom;
//       float3 camera_d00
//         = normalize(lookAt-lookFrom);
//       float aspect = float(fbSize.x) / float(fbSize.y);
//       float3 camera_ddu
//         = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
//       float3 camera_ddv
//         = cosFovy * normalize(cross(camera_ddu,camera_d00));
//       camera_d00 -= 0.5f * camera_ddu;
//       camera_d00 -= 0.5f * camera_ddv;

//       // ----------- set variables  ----------------------------
//       gprtRayGenSet3fv    (rayGen,"camera.pos",   (float*)&camera_pos);
//       gprtRayGenSet3fv    (rayGen,"camera.dir_00",(float*)&camera_d00);
//       gprtRayGenSet3fv    (rayGen,"camera.dir_du",(float*)&camera_ddu);
//       gprtRayGenSet3fv    (rayGen,"camera.dir_dv",(float*)&camera_ddv);
//       gprtBuildShaderBindingTable(context, GPRT_SBT_RAYGEN);
//     }

//     if (rstate == GLFW_PRESS) {
//       glfwGetCursorPos(window, &xpos, &ypos);
//       float dy = ypos - lastypos;

//       float3 view_vec = lookFrom - lookAt;

//       if (dy > 0.0) {
//         view_vec.x *= 0.95;
//         view_vec.y *= 0.95;
//         view_vec.z *= 0.95;
//       } else {
//         view_vec.x *= 1.05;
//         view_vec.y *= 1.05;
//         view_vec.z *= 1.05;
//       }

//       lookFrom = lookAt + view_vec;

//       gprtRayGenSet3fv(rayGen, "camera.pos", (float*)&lookFrom);
//       gprtBuildShaderBindingTable(context, GPRT_SBT_RAYGEN);

//     }

//     // Now, trace rays
//     gprtRayGenLaunch2D(context,rayGen,fbSize.x,fbSize.y);

//     // Render results to screen
//     void* pixels = gprtBufferGetPointer(frameBuffer);
//     if (fbTexture == 0)
//       glGenTextures(1, &fbTexture);

//     glBindTexture(GL_TEXTURE_2D, fbTexture);
//     GLenum texFormat = GL_RGBA;
//     GLenum texelType = GL_UNSIGNED_BYTE;
//     glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
//                   texelType, pixels);

//     glDisable(GL_LIGHTING);
//     glColor3f(1, 1, 1);

//     glMatrixMode(GL_MODELVIEW);
//     glLoadIdentity();

//     glEnable(GL_TEXTURE_2D);
//     glBindTexture(GL_TEXTURE_2D, fbTexture);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//     glDisable(GL_DEPTH_TEST);

//     glViewport(0, 0, fbSize.x, fbSize.y);

//     glMatrixMode(GL_PROJECTION);
//     glLoadIdentity();
//     glOrtho(0.f, (float)fbSize.x, (float)fbSize.y, 0.f, -1.f, 1.f);

//     glBegin(GL_QUADS);
//     {
//       glTexCoord2f(0.f, 0.f);
//       glVertex3f(0.f, 0.f, 0.f);

//       glTexCoord2f(0.f, 1.f);
//       glVertex3f(0.f, (float)fbSize.y, 0.f);

//       glTexCoord2f(1.f, 1.f);
//       glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

//       glTexCoord2f(1.f, 0.f);
//       glVertex3f((float)fbSize.x, 0.f, 0.f);
//     }
//     glEnd();

//     glfwSwapBuffers(window);
//     glfwPollEvents();
//   }

//   // ##################################################################
//   // and finally, clean up
//   // ##################################################################

//   LOG("cleaning up ...");

//   glfwDestroyWindow(window);
//   glfwTerminate();
// }
