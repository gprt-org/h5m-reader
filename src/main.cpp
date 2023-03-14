
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "argparse/argparse.hpp"

#include "DagMC.hpp"
#include "moab/Core.hpp"
#include "moab/Range.hpp"

#include "gprt.h"

#include "sharedCode.h"
#include "mb_util.hpp"

#include "imgui.h"
#include <imgui_gradient/imgui_gradient.hpp>

#define LOG(message)                                            \
  std::cout << GPRT_TERMINAL_BLUE;                               \
  std::cout << "#gprt.sample(main): " << message << std::endl;   \
  std::cout << GPRT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << GPRT_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#gprt.sample(main): " << message << std::endl;   \
  std::cout << GPRT_TERMINAL_DEFAULT;

extern GPRTProgram dbl_deviceCode;
extern GPRTProgram voxelize_deviceCode;

#define MOAB_CHECK_ERROR(EC) if (EC != moab::MB_SUCCESS) return 1;

void render();

// initial image resolution
// const int2 fbSize = {2560, 1440};
const int2 fbSize = {1000, 1000};

int main(int argc, char** argv) {

  argparse::ArgumentParser args("GPRT H5M READER");

  args.add_argument("filename")
      .help("Path to the DAGMC file to view");
  args.add_argument("--volumes")
      .help("Subset of volume IDs to visualize")
      .nargs(argparse::nargs_pattern::any)
      .scan<'i', int>();
  args.add_argument("--type")
      .help("Floating point primitive representation (one of 'float' or 'double'")
      .default_value(std::string("float"));
  args.add_argument("--write")
      .help("Write out the DAGMC file as rendered using GPRT in this app")
      .default_value(false)
      .implicit_value(true);
  args.add_argument("--grid")
      .help("number of voxels per grid dimension")
      .nargs(3)
      .scan<'i', uint32_t>()
      .default_value(std::vector<uint32_t>(3, 256));

  try {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(0);
  }

  auto filename = args.get<std::string>("filename");

  std::string type = args.get<std::string>("type");

  bool useFloats = (type == "float");
  if (type != "double" && type != "float") {
    std::cerr << "Error: primitive representation must be set to either 'float' or 'double'." << std::endl;
    std::exit(1);
  }

  std::vector<uint32_t> _gridDims = args.get<std::vector<uint32_t>>("grid");
  uint3 gridDims = {_gridDims[0], _gridDims[1], _gridDims[2]};

  moab::ErrorCode rval;
  std::shared_ptr<moab::DagMC> dag = std::make_shared<moab::DagMC>();

  auto mbi = dag->moab_instance_sptr();

  rval = dag->load_file(filename.c_str());
  MOAB_CHECK_ERROR(rval);

  rval = dag->setup_indices();
  MOAB_CHECK_ERROR(rval);

  if (dag->num_entities(2) == 0) {
    std::cerr << "No surfaces were found in the model" << std::endl;
    std::exit(1);
  }

  rval = dag->setup_impl_compl();
  MOAB_CHECK_ERROR(rval);

  // always create graveyard, overwriting the current graveyard if present
  if ( !dag->has_graveyard() ) {
    rval = dag->create_graveyard(true);
    MOAB_CHECK_ERROR(rval);
  }

  // write geometry as-rendered for verification
  if (args.get<bool>("write")) {
    std::string file_out{"as-is.h5m"};
    dag->write_mesh(file_out.c_str(), file_out.size());
  }

  // get the graveyard group
  EntityHandle graveyard_group;
  rval = dag->get_graveyard_group(graveyard_group);
  MOAB_CHECK_ERROR(rval);

  // get the entity handle of the graveyard volume
  Range graveyard_vols;
  rval = mbi->get_entities_by_handle(graveyard_group, graveyard_vols);
  MOAB_CHECK_ERROR(rval);

  // should be one graveyard vol
  if (graveyard_vols.size() != 1) {
    std::cout << "ERROR: More than one graveyard volume is present" << std::endl;
    std::exit(1);
  }

  int graveyard_id = dag->get_entity_id(graveyard_vols[0]);

  EntityHandle implicit_complement;
  rval = dag->geom_tool()->get_implicit_complement(implicit_complement);
  MOAB_CHECK_ERROR(rval);

  int implicit_complement_id = dag->get_entity_id(implicit_complement);

  std::vector<int> volumes;
  if (args.is_used("--volumes")) {
    volumes = args.get<std::vector<int>>("volumes");
  } else {
    for (int i = 0; i < dag->num_entities(3); i++) volumes.push_back(dag->id_by_index(3, i+1));
  }

  // start up GPRT
  gprtRequestWindow(fbSize.x, fbSize.y, "H5M");
  GPRTContext context = gprtContextCreate(nullptr, 1);
  GPRTModule module = gprtModuleCreate(context, dbl_deviceCode);
  GPRTModule voxelizeModule = gprtModuleCreate(context, voxelize_deviceCode);
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
  gprtGeomTypeSetClosestHitProg(SPTriangleType,0, module, "SPTriangle");

  // To test DDA and point queries, to voxelize parts
  GPRTRayGenOf<RayGenData> Voxelize;

  // For launching double precision rays
  GPRTRayGenOf<RayGenData> DPRayGen = nullptr;
  if (!useFloats) DPRayGen = gprtRayGenCreate<RayGenData>(context, module, "DPRayGen");
  if (!useFloats) Voxelize = gprtRayGenCreate<RayGenData>(context, voxelizeModule, "DPVoxelize");

  // For launching single precision rays
  GPRTRayGenOf<RayGenData> SPRayGen = nullptr;
  GPRTRayGenOf<RayGenData> SPVolVis = nullptr;
  if (useFloats) SPRayGen = gprtRayGenCreate<RayGenData>(context, module, "SPRayGen");
  if (useFloats) SPVolVis = gprtRayGenCreate<RayGenData>(context, module, "SPVolVis");
  if (useFloats) Voxelize = gprtRayGenCreate<RayGenData>(context, voxelizeModule, "SPVoxelize");

  // What to do if a ray misses
  GPRTMissOf<MissProgData> miss
    = gprtMissCreate<MissProgData>(context,module,"miss");

  // compute centroid to look at
  auto bbox = bounding_box(mbi.get());

  // create volumes
  MBVolumes<SPTriangleSurface, SPTriangleData> spvols(volumes);
  MBVolumes<DPTriangleSurface, DPTriangleData> dpvols(volumes);

  if (useFloats) {
    spvols.populate_surfaces(dag.get());
    spvols.create_geoms(context, SPTriangleType);
    spvols.setup(context, module, fbSize);
    spvols.create_accel_structures(context);
  } else {
    dpvols.populate_surfaces(dag.get());
    dpvols.create_geoms(context, DPTriangleType);
    dpvols.setup(context, module, fbSize);
    dpvols.create_accel_structures(context);
  }

  // create geometries
  uint32_t numVols = dag->num_entities(3);

  int max_vol_id = 0;
  for (int i = 0; i < numVols; i++) {
    max_vol_id = std::max(max_vol_id, dag->id_by_index(3, i));
  }

  // empty the mesh library's copy (good riddance)
  rval = mbi->delete_mesh();
  MOAB_CHECK_ERROR(rval);
  mbi.reset();

  if (volumes.size() == 0) {
    std::cerr << "No surfaces visible" << std::endl;
    std::exit(1);
  }

  double3 aabbCentroid = bbox.first + (bbox.second - bbox.first) * 0.5;

  float3 lookFrom = float3(float(aabbCentroid.x), float(aabbCentroid.y)  - 100.f, float(aabbCentroid.z));
  float3 lookAt = float3(float(aabbCentroid.x),float(aabbCentroid.y),float(aabbCentroid.z));
  float3 lookUp = {0.f,0.f,-1.f};
  float cosFovy = 0.66f;

  // Allocate a grid for DDA to hold physics kernel results
  auto ddaGrid = gprtDeviceBufferCreate<float>(context, gridDims.x * gridDims.y * gridDims.z);
  gprtBufferClear(ddaGrid);

  // ----------- set variables  ----------------------------
  auto missData = gprtMissGetParameters(miss);
  missData->color0 = float3(0.1f, 0.1f, 0.1f);
  missData->color1 = float3(0.f, 0.f, 0.f);

  // ----------- set raygen variables  ----------------------------
  auto frameBuffer = gprtDeviceBufferCreate<uint32_t>(context, fbSize.x*fbSize.y);
  auto accumBuffer = gprtDeviceBufferCreate<float4>(context, fbSize.x*fbSize.y);

  // This is new, setup GUI frame buffer. We'll rasterize the GUI to this texture, then composite the GUI on top of the
  // rendered scene.
  auto guiColorAttachment = gprtDeviceTextureCreate<uint32_t>(
      context, GPRT_IMAGE_TYPE_2D, GPRT_FORMAT_R8G8B8A8_SRGB, fbSize.x, fbSize.y, 1, false, nullptr);
  auto guiDepthAttachment = gprtDeviceTextureCreate<float>(
      context, GPRT_IMAGE_TYPE_2D, GPRT_FORMAT_D32_SFLOAT, fbSize.x, fbSize.y, 1, false, nullptr);
  gprtGuiSetRasterAttachments(context, guiColorAttachment, guiDepthAttachment);

  // Colormaps for visualization
  auto surfaceColormap = gprtDeviceTextureCreate<uint32_t>(
    context, GPRT_IMAGE_TYPE_1D, GPRT_FORMAT_R8G8B8A8_SRGB, 256, 1, 1, false, nullptr
  );
  auto ddaColormap = gprtDeviceTextureCreate<uint32_t>(
    context, GPRT_IMAGE_TYPE_1D, GPRT_FORMAT_R8G8B8A8_SRGB, 256, 1, 1, false, nullptr
  );

  auto sampler = gprtSamplerCreate(context, GPRT_FILTER_LINEAR, GPRT_FILTER_LINEAR, GPRT_FILTER_LINEAR, 1, GPRT_SAMPLER_ADDRESS_MODE_CLAMP);

  // need this to communicate double precision rays to intersection program
  // ray origin xyz + tmin, then ray direction xyz + tmax
  GPRTBufferOf<double> doubleRayBuffer = nullptr;
  RayGenData* rayGenData = nullptr;
  RayGenData* volVisData = nullptr;
  RayGenData* voxelizeData = gprtRayGenGetParameters(Voxelize);

  if (useFloats) {
    rayGenData = gprtRayGenGetParameters(SPRayGen);
    volVisData = gprtRayGenGetParameters(SPVolVis);
  }
  else {
    rayGenData = gprtRayGenGetParameters(DPRayGen);
    doubleRayBuffer = gprtDeviceBufferCreate<double>(context,fbSize.x*fbSize.y*8);
    rayGenData->dpRays = gprtBufferGetHandle(doubleRayBuffer);
  }

  rayGenData->fbPtr = gprtBufferGetHandle(frameBuffer);
  rayGenData->accumPtr = gprtBufferGetHandle(accumBuffer);
  rayGenData->guiTexture = gprtTextureGetHandle(guiColorAttachment);
  rayGenData->fbSize = fbSize;
  rayGenData->moveOrigin = false;
  rayGenData->world = gprtAccelGetHandle(spvols.world_tlas_);
  rayGenData->partTrees = gprtBufferGetHandle(spvols.tlas_buffer_);
  rayGenData->aabbMin = float3(bbox.first.x, bbox.first.y, bbox.first.z);
  rayGenData->aabbMax = float3(bbox.second.x, bbox.second.y, bbox.second.z);
  rayGenData->unit = 1000.f;
  rayGenData->surfaceColormap = gprtTextureGetHandle(surfaceColormap);
  rayGenData->ddaColormap = gprtTextureGetHandle(ddaColormap);
  rayGenData->numVolumes = numVols;
  rayGenData->maxVolID = max_vol_id;
  rayGenData->graveyardID = graveyard_id;
  rayGenData->complementID = implicit_complement_id;
  rayGenData->frameID = 0;
  rayGenData->colormapSampler = gprtSamplerGetHandle(sampler);
  rayGenData->ddaGrid = gprtBufferGetHandle(ddaGrid);
  rayGenData->gridDims = gridDims;

  if (volVisData) *volVisData = *rayGenData;
  *voxelizeData = *rayGenData;

  gprtBuildShaderBindingTable(context, GPRT_SBT_ALL);

  // Voxelize into DDA grid
  gprtRayGenLaunch3D(context, Voxelize, gridDims.x, gridDims.y, gridDims.z);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  LOG("launching ...");

  ImGG::GradientWidget surfaceColormapWidget{};
  ImGG::GradientWidget ddaColormapWidget{};

  bool firstFrame = true;
  double xpos = 0.f, ypos = 0.f;
  double lastxpos, lastypos;
  while (!gprtWindowShouldClose(context))
  {
    ImGuiIO &io = ImGui::GetIO();
    ImGui::NewFrame();

    rayGenData->frameID++;

    if (surfaceColormapWidget.widget("Surface Colormap") || firstFrame) {
      auto make_8bit = [](const float f) -> uint32_t {
        return std::min(255, std::max(0, int(f * 256.f)));
      };

      auto make_rgba = [make_8bit](float4 color) -> uint32_t {
        float gamma = 2.2;
        color = pow(color, float4(1.0f / gamma, 1.0f / gamma, 1.0f / gamma, 1.0f));
        return (make_8bit(color.x) << 0) + (make_8bit(color.y) << 8) + (make_8bit(color.z) << 16) + (make_8bit(color.w) << 24);
      };

      gprtTextureMap(surfaceColormap);
      uint32_t *ptr = gprtTextureGetPointer(surfaceColormap);
      for (uint32_t i = 0; i < 256; ++i) {
        auto result = surfaceColormapWidget.gradient().at(ImGG::RelativePosition(float(i + 1) / 256.f));
        ptr[i] = make_rgba(float4(result.x, result.y, result.z, result.w));
      }
      gprtTextureUnmap(surfaceColormap);
      rayGenData->frameID = 1;
    }

    if (ddaColormapWidget.widget("Grid Colormap") || firstFrame) {
      auto make_8bit = [](const float f) -> uint32_t {
        return std::min(255, std::max(0, int(f * 256.f)));
      };

      auto make_rgba = [make_8bit](float4 color) -> uint32_t {
        float gamma = 2.2;
        color = pow(color, float4(1.0f / gamma, 1.0f / gamma, 1.0f / gamma, 1.0f));
        return (make_8bit(color.x) << 0) + (make_8bit(color.y) << 8) + (make_8bit(color.z) << 16) + (make_8bit(color.w) << 24);
      };

      gprtTextureMap(ddaColormap);
      uint32_t *ptr = gprtTextureGetPointer(ddaColormap);
      for (uint32_t i = 0; i < 256; ++i) {
        auto result = ddaColormapWidget.gradient().at(ImGG::RelativePosition(float(i + 1) / 256.f));
        ptr[i] = make_rgba(float4(result.x, result.y, result.z, result.w));
      }
      gprtTextureUnmap(ddaColormap);
      rayGenData->frameID = 1;
    }

    static bool moveOrigin = rayGenData->moveOrigin;
    if (ImGui::Checkbox("Move Origin", &moveOrigin)) {
      rayGenData->moveOrigin = moveOrigin;
    }

    static float unit = 1000.f;
    if (ImGui::SliderFloat("unit", &unit, 1., 1000.f)) {
      rayGenData->unit = unit;
      if (volVisData) *volVisData = *rayGenData;
      rayGenData->frameID = 1;
    }

    float speed = .001f;
    lastxpos = xpos;
    lastypos = ypos;
    gprtGetCursorPos(context, &xpos, &ypos);
    if (firstFrame) {
      lastxpos = xpos;
      lastypos = ypos;
    }

    float dx = xpos - lastxpos;
    float dy = ypos - lastypos;

    int state = gprtGetMouseButton(context, GPRT_MOUSE_BUTTON_LEFT);
    int rstate = gprtGetMouseButton(context, GPRT_MOUSE_BUTTON_RIGHT);
    int mstate = gprtGetMouseButton(context, GPRT_MOUSE_BUTTON_MIDDLE);

    int w_state = gprtGetKey(context, GPRT_KEY_W);
    int c_state = gprtGetKey(context, GPRT_KEY_C);
    int ctrl_state = gprtGetKey(context, GPRT_KEY_LEFT_CONTROL);

    // close window on Ctrl-W press
    if (w_state && ctrl_state) { break; }
    // close window on Ctrl-C press
    if (c_state && ctrl_state) { break; }

    // If we click the mouse, we should rotate the camera
    if (state == GPRT_PRESS && !io.WantCaptureMouse || firstFrame)
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
      float xAngle = -dx * deltaAngleX;
      float yAngle = -dy * deltaAngleY;

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

      rayGenData->frameID = 1;
    }

    if (rstate == GPRT_PRESS && !io.WantCaptureMouse) {
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

      rayGenData->frameID = 1;

    }

    if (mstate == GPRT_PRESS && !io.WantCaptureMouse) {
      float4 position = {lookFrom.x, lookFrom.y, lookFrom.z, 1.f};
      float4 pivot = {lookAt.x, lookAt.y, lookAt.z, 1.0};
      float3 lookRight = cross(lookUp, normalize(pivot - position).xyz());

      float3 translation = lookRight * dx + lookUp * -dy;

      lookFrom = lookFrom + translation;
      lookAt = lookAt + translation;

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

      rayGenData->frameID = 1;
    }

    if (volVisData) *volVisData = *rayGenData;
    gprtBuildShaderBindingTable(context, GPRT_SBT_RAYGEN);

    static int choice = 0;
    ImGui::RadioButton("Show Surfaces", &choice, 0);
    ImGui::RadioButton("Show Volumes", &choice, 1);
    ImGui::EndFrame();

    gprtTextureClear(guiDepthAttachment);
    gprtTextureClear(guiColorAttachment);
    gprtGuiRasterize(context);

    // Now, trace rays
    gprtBeginProfile(context);

    if (useFloats) {

      if (choice == 0) {
        gprtRayGenLaunch2D(context,SPRayGen,fbSize.x,fbSize.y);
      }
      else if (choice == 1) {
        gprtRayGenLaunch2D(context,SPVolVis,fbSize.x,fbSize.y);
      }
    }
    else  {
      gprtRayGenLaunch2D(context,DPRayGen,fbSize.x,fbSize.y);
    }
    float ms = gprtEndProfile(context) * 1.e-06;
    std::string perf = "RF Time: " + std::to_string(ms) + " ms, " +
                       "Time per ray: "
                       + std::to_string(ms / ( fbSize.x * fbSize.y)) + " ms";

    gprtSetWindowTitle(context, perf.c_str());

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
  gprtTextureDestroy(guiColorAttachment);
  gprtTextureDestroy(guiDepthAttachment);
  if (doubleRayBuffer) gprtBufferDestroy(doubleRayBuffer);
  if (SPRayGen) gprtRayGenDestroy(SPRayGen);
  if (SPVolVis) gprtRayGenDestroy(SPVolVis);
  if (DPRayGen) gprtRayGenDestroy(DPRayGen);
  gprtRayGenDestroy(Voxelize);
  gprtMissDestroy(miss);
  if (useFloats) spvols.cleanup();
  else dpvols.cleanup();
  gprtGeomTypeDestroy(DPTriangleType);
  gprtGeomTypeDestroy(SPTriangleType);
  gprtModuleDestroy(module);
  gprtModuleDestroy(voxelizeModule);
  gprtContextDestroy(context);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");

  return 0;
}
