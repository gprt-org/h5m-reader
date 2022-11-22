
#include <iostream>
#include <memory>
#include <vector>
#include <string>


#include "argparse/argparse.hpp"

#include "moab/Core.hpp"
#include "moab/Range.hpp"

#include "MOABDirectAccess.h"

#include "gprt.h"

#include "deviceCode.h"

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

/* forward declarations to double precision cube.
  See gprt_data/double_cube.cpp for details */
extern uint32_t double_indices[];
extern double double_vertices[];

float transform[3][4] =
  {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f
  };

// initial image resolution
const int num_rays = 1000000;

int main(int argc, char** argv) {

  argparse::ArgumentParser args("GPRT H5M READER");

  args.add_argument("filename");

  try {
  args.parse_args(argc, argv);                  // Example: ./main -abc 1.95 2.47
  }
  catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << args;
  exit(0);
  }

  auto filename = args.get<std::string>("filename");

  std::shared_ptr<moab::Core> mbi = std::make_shared<moab::Core>();
  moab::ErrorCode rval;

  std::cout << "Loading " << filename << "..." << std::endl;
  rval = mbi->load_file(filename.c_str());
  MOAB_CHECK_ERROR(rval);

  // create a direct access manager
  MBDirectAccess mdam (mbi.get());
  // setup datastructs storing internal information
  mdam.setup();

  int n_vertices = mdam.xyz().size() / 3;
  int n_tris = mdam.conn().size() / 3;

  // clear out the MOAB interface, we don't need it anymore
  rval = mbi->delete_mesh();
  MOAB_CHECK_ERROR(rval);

  mbi.reset();

  // start up GPRT
  GPRTContext context = gprtContextCreate(nullptr, 1);
  GPRTModule module = gprtModuleCreate(context, dbl_deviceCode);
  // -------------------------------------------------------
  // Setup programs and geometry types
  // -------------------------------------------------------
  GPRTVarDecl DPTriangleVars[] = {
    { "vertex",  GPRT_BUFFER, GPRT_OFFSETOF(DPTriangleData, vertex)},
    { "index" ,  GPRT_BUFFER, GPRT_OFFSETOF(DPTriangleData, index)},
    { "aabbs" ,  GPRT_BUFFER, GPRT_OFFSETOF(DPTriangleData, aabbs)},
    { "dpRays" , GPRT_BUFFER, GPRT_OFFSETOF(DPTriangleData, dpRays)},
    { /* sentinel to mark end of list */ }
  };

    GPRTGeomType DPTriangleType
    = gprtGeomTypeCreate(context,
                        GPRT_AABBS,
                        sizeof(DPTriangleVars),
                        DPTriangleVars);
  GPRTCompute DPTriangleBoundsProgram
    = gprtComputeCreate(context,module,"DPTriangle",
                        sizeof(DPTriangleVars),
                        DPTriangleVars);
  gprtGeomTypeSetClosestHitProg(DPTriangleType,0,
                           module,"DPTriangle");
  gprtGeomTypeSetIntersectionProg(DPTriangleType,0,
                           module,"DPTrianglePlucker");


  GPRTVarDecl rayGenVars[] = {
    { "frameId",       GPRT_INT,    GPRT_OFFSETOF(RayGenData, frameId)},
    { "dpRays",        GPRT_BUFFER, GPRT_OFFSETOF(RayGenData, dpRays)},
    { "world",         GPRT_ACCEL,  GPRT_OFFSETOF(RayGenData, world)},
    { "distances",     GPRT_BUFFER, GPRT_OFFSETOF(RayGenData, distances)},
    { /* sentinel to mark end of list */ }
  };
  GPRTRayGen rayGen
    = gprtRayGenCreate(context, module, "AABBRayGen", sizeof(RayGenData), rayGenVars, -1);

  GPRTBuffer distances =
    gprtHostBufferCreate(context, GPRT_DOUBLE, num_rays);

  gprtRayGenSetBuffer(rayGen, "distances", distances);

  GPRTVarDecl missVars[]
    = {
    {"temp", GPRT_INT, GPRT_OFFSETOF(MissProgData, temp)},
    { /* sentinel to mark end of list */ }
  };
  GPRTMiss miss
    = gprtMissCreate(context, module, "miss", sizeof(MissProgData),
                        missVars, -1);

  gprtBuildPipeline(context);

  // ------------------------------------------------------------------
  // aabb mesh
  // ------------------------------------------------------------------
  GPRTBuffer vertexBuffer
    = gprtDeviceBufferCreate(context, GPRT_DOUBLE3, n_vertices, mdam.xyz().data());
  GPRTBuffer indexBuffer
    = gprtDeviceBufferCreate(context, GPRT_INT3, n_tris, mdam.conn().data());
  GPRTBuffer aabbPositionsBuffer
    = gprtDeviceBufferCreate(context, GPRT_FLOAT3, 2*n_tris, nullptr);

  // clear out mdam data now that it's been transferred to device
  mdam.clear();

  GPRTGeom dpCubeGeom
    = gprtGeomCreate(context, DPTriangleType);
  gprtAABBsSetPositions(dpCubeGeom, aabbPositionsBuffer,
                        n_tris, 2 * sizeof(float3), 0);

  gprtGeomSetBuffer(dpCubeGeom, "vertex", vertexBuffer);
  gprtGeomSetBuffer(dpCubeGeom, "index", indexBuffer);
  gprtGeomSetBuffer(dpCubeGeom, "aabbs", aabbPositionsBuffer);

  gprtComputeSetBuffer(DPTriangleBoundsProgram, "vertex", vertexBuffer);
  gprtComputeSetBuffer(DPTriangleBoundsProgram, "index", indexBuffer);
  gprtComputeSetBuffer(DPTriangleBoundsProgram, "aabbs", aabbPositionsBuffer);

  // compute AABBs in parallel with a compute shader
  gprtBuildShaderBindingTable(context, GPRT_SBT_COMPUTE);
  gprtComputeLaunch1D(context, DPTriangleBoundsProgram, n_tris);

  GPRTAccel aabbAccel = gprtAABBAccelCreate(context, 1, &dpCubeGeom);
  gprtAccelBuild(context, aabbAccel);

  // compute centroid to look at
  const auto& xyz = mdam.xyz();
  double3 aabbmin = double3(xyz[0],xyz[1],xyz[2]);
  double3 aabbmax = aabbmin;
  for (uint32_t i = 1; i < n_vertices; ++i) {
    aabbmin = linalg::min(aabbmin, double3(xyz[i * 3 + 0],
                                           xyz[i * 3 + 1],
                                           xyz[i * 3 + 2]));
    aabbmax = linalg::max(aabbmax, double3(xyz[i * 3 + 0],
                                           xyz[i * 3 + 1],
                                           xyz[i * 3 + 2]));
  }
  double3 aabbCentroid = aabbmin + (aabbmax - aabbmin) * 0.5;

  float3 lookFrom = float3(float(aabbCentroid.x), float(aabbCentroid.y)  - 50.f, float(aabbCentroid.z));
  float3 lookAt = float3(float(aabbCentroid.x),float(aabbCentroid.y),float(aabbCentroid.z));
  float3 lookUp = {0.f,0.f,-1.f};
  float cosFovy = 0.66f;

  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  GPRTBuffer transformBuffer
    = gprtDeviceBufferCreate(context,GPRT_TRANSFORM,1,transform);
  GPRTAccel world = gprtInstanceAccelCreate(context, 1, &aabbAccel);
  gprtInstanceAccelSet3x4Transforms(world, transformBuffer);
  gprtAccelBuild(context, world);

  // ----------- set raygen variables  ----------------------------

  // need this to communicate double precision rays to intersection program
  // ray origin xyz + tmin, then ray direction xyz + tmax
  GPRTBuffer doubleRayBuffer
    = gprtDeviceBufferCreate(context, GPRT_DOUBLE, num_rays*8);
  gprtRayGenSetBuffer(rayGen, "dpRays", doubleRayBuffer);
  gprtRayGenSetAccel(rayGen, "world", world);

  // Also set on geometry for intersection program
  gprtGeomSetBuffer(dpCubeGeom,"dpRays", doubleRayBuffer);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  gprtBuildPipeline(context);
  gprtBuildShaderBindingTable(context, GPRT_SBT_ALL);

  // ----------- set variables  ----------------------------
  int n_launches = 1000;
  float avg_time = 0.0;
  for (int i = 0; i < n_launches; ++i) {
    gprtRayGenSet1i(rayGen, "frameId", i);
    gprtBuildShaderBindingTable(context, GPRT_SBT_RAYGEN);
    gprtBeginProfile(context);
    gprtRayGenLaunch1D(context, rayGen, num_rays);
    float t = gprtEndProfile(context);
    float ms = t * 1.e-06;
    avg_time += t;
    std::cout << "RF Time: " << ms << " ms" << std::endl;
    std::cout << "Time per ray: " << ms / num_rays << " ms" << std::endl;
  }

  float ms_per_ray = avg_time * 1.e-06 / (num_rays * n_launches);
  std::cout <<  "======================================" << std::endl;
  std::cout << "Avg RF time: " << ms_per_ray << " ms " << std::endl;
  std::cout <<  "======================================" << std::endl;

  std::cout <<  "======================================" << std::endl;
  std::cout << "Gigarays: " << 1 / (ms_per_ray / 1e3) / 1e9 << std::endl;
  std::cout <<  "======================================" << std::endl;



  gprtBufferDestroy(vertexBuffer);
  gprtBufferDestroy(indexBuffer);
  gprtBufferDestroy(aabbPositionsBuffer);
  gprtBufferDestroy(distances);
  gprtBufferDestroy(doubleRayBuffer);
  gprtBufferDestroy(transformBuffer);
  gprtRayGenDestroy(rayGen);
  gprtMissDestroy(miss);
  gprtComputeDestroy(DPTriangleBoundsProgram);
  gprtAccelDestroy(aabbAccel);
  gprtAccelDestroy(world);
  gprtGeomDestroy(dpCubeGeom);
  gprtGeomTypeDestroy(DPTriangleType);
  gprtModuleDestroy(module);
  gprtContextDestroy(context);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");

  return 0;
}