#include <vector>
#include <map>
#include <set>

#include "moab/Core.hpp"
#include "moab/Range.hpp"
#include "MBTagConventions.hpp"

#include "gprt.h"

using namespace moab;

int DEBUG_SURF = -4;

std::set<EntityHandle> visible_surfs;

float3 rnd_color() {
  return normalize(float3( std::rand(), std::rand(), std::rand()));
}

template<class T, typename R>
struct MBTriangleSurface {

  typedef R vertex_type;

  int id;
  int n_tris;
  int frontface_vol;
  int backface_vol;
  std::vector<R> vertices;
  std::vector<uint3> connectivity;
  GPRTBufferOf<float3> aabb_buffer;
  GPRTGeomOf<T> triangle_geom_s;
  int2 parent_ids;
  bool aabbs_present {false};

  struct SurfaceData {
    std::vector<double> coords;
    std::vector<uint3> connectivity;
  };

MBTriangleSurface(DagMC* dagmc, EntityHandle surf_handle, EntityHandle vol_handle) {
  ErrorCode rval;

  moab::Interface* mbi = dagmc->moab_instance();

  // keep surface and volume IDs for debugging
  int surface_id = dagmc->get_entity_id(surf_handle);
  int volume_id = dagmc->get_entity_id(vol_handle);

  // get the triangles for this surface
  std::vector<EntityHandle> surf_tris;
  rval = mbi->get_entities_by_dimension(surf_handle, 2, surf_tris);
  MB_CHK_SET_ERR_CONT(rval, "Failed to get surface " << surface_id << "'s triangles");

  n_tris = surf_tris.size();

  std::vector<EntityHandle> conn;
  rval = mbi->get_connectivity(surf_tris.data(), surf_tris.size(), conn);
  MB_CHK_SET_ERR_CONT(rval, "Failed to get surface connectivity");

  Range verts;
  verts.insert<std::vector<EntityHandle>>(conn.begin(), conn.end());

  std::vector<double> coords(3*verts.size());

  rval = mbi->get_coords(verts, coords.data());
  MB_CHK_SET_ERR_CONT(rval, "Failed to get vertex coordinates for surface " << surface_id);

  vertices.resize(3*verts.size());

  for (int i = 0; i < verts.size(); i++) {
      vertices[i] = R(coords[3*i], coords[3*i+1], coords[3*i+2]);
  }

  connectivity.resize(surf_tris.size());

  for (int i = 0; i < surf_tris.size(); i++) {
      connectivity[i] = uint3(verts.index(conn[3*i]), verts.index(conn[3*i+1]), verts.index(conn[3*i+2]));
  }

  // get the geom sense tag
  Tag geom_sense;
  const char GEOM_SENSE_2_TAG_NAME[] = "GEOM_SENSE_2";
  rval = mbi->tag_get_handle(GEOM_SENSE_2_TAG_NAME, geom_sense);
  MB_CHK_SET_ERR_CONT(rval, "Failed to get the geometry sense tag");

  std::array<EntityHandle, 2> parent_vols;
  rval = mbi->tag_get_data(geom_sense, &surf_handle, 1, parent_vols.data());
  MB_CHK_SET_ERR_CONT(rval, "Failed to get the geometry sense of surface " << surface_id);

  std::array<int, 2> parent_ids = {-1 , -1};
  int parents_size = parent_vols[1] == 0 ? 1 : 2;
  Tag id_tag = mbi->globalId_tag();
  rval = mbi->tag_get_data(id_tag, parent_vols.data(), parents_size, parent_ids.data());
  MB_CHK_SET_ERR_CONT(rval, "Failed to get parent volume IDs");

  // if we're building up this surface and it has a reverse sense relative to the volume provided,
  // flip the triangle normals by changing the triangle connectivity
  bool sense_reverse = parent_ids[1] == volume_id;
  id = sense_reverse ? -surface_id : surface_id;
  if (sense_reverse) {
    for (int i = 0; i < surf_tris.size(); i++) {
      auto& conn = connectivity[i];
      std::swap(conn[1], conn[2]);
    }
  }

  // set forward/reverse volume information
  if (sense_reverse) {
    this->parent_ids[0] = parent_ids[1];
    this->parent_ids[1] = parent_ids[0];
  } else {
    this->parent_ids[0] = parent_ids[0];
    this->parent_ids[1] = parent_ids[1];
  }
}


};

using SPTriangleSurface = MBTriangleSurface<SPTriangleData, float3>;
using DPTriangleSurface = MBTriangleSurface<DPTriangleData, double3>;

template<class T, class G>
struct MBVolume {
  // Constructor
  MBVolume(int id) : id_(id) {};

  // Methods
  void populate_surfaces(DagMC* dagmc) {
    EntityHandle vol = dagmc->entity_by_id(3, id_);

    Range moab_surf_handles;
    ErrorCode rval = dagmc->moab_instance()->get_child_meshsets(vol, moab_surf_handles);

    for (auto surface_handle : moab_surf_handles)
      surfaces_.emplace_back(std::move(T(dagmc, surface_handle, vol)));
  }

  void create_geoms(GPRTContext context, GPRTGeomTypeOf<G> g_type) {
    for (auto& surf : surfaces_) {
      vertex_buffers_.push_back(gprtDeviceBufferCreate<typename T::vertex_type>(context, surf.vertices.size(), surf.vertices.data()));
      connectivity_buffers_.push_back(gprtDeviceBufferCreate<uint3>(context, surf.connectivity.size(), surf.connectivity.data()));
      gprt_geoms_.push_back(gprtGeomCreate<G>(context, g_type));
      G* geom_data = gprtGeomGetParameters(gprt_geoms_.back());
      geom_data->vertex = gprtBufferGetHandle(vertex_buffers_.back());
      geom_data->index = gprtBufferGetHandle(connectivity_buffers_.back());
      geom_data->id = surf.id;
      geom_data->vols = surf.parent_ids;
    }
  }

  void setup(GPRTContext context, GPRTModule module, int2 fbSize) {
    for (int i = 0; i < surfaces_.size(); i++) {
      auto& surf = surfaces_[i];
      gprtTrianglesSetVertices(gprt_geoms_[i], vertex_buffers_[i], surf.vertices.size());
      gprtTrianglesSetIndices(gprt_geoms_[i], connectivity_buffers_[i], surf.connectivity.size());
    }
  }

  void create_accel_structures(GPRTContext context) {
    blas_ = gprtTrianglesAccelCreate(context, gprt_geoms_.size(), gprt_geoms_.data());
    gprtAccelBuild(context, blas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
    tlas_ = gprtInstanceAccelCreate(context, 1, &blas_);
    gprtAccelBuild(context, tlas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);
  }

  void cleanup () {
    gprtAccelDestroy(tlas_);
    gprtAccelDestroy(blas_);
    for (auto& vert_buff : vertex_buffers_) gprtBufferDestroy(vert_buff);
    for (auto& conn_buff : connectivity_buffers_) gprtBufferDestroy(conn_buff);
    for (auto& geom : gprt_geoms_) gprtGeomDestroy(geom);
    // for (auto& surf : surfaces_) {
    //   if (surf.aabb_buffer) gprtBufferDestroy(surf.aabb_buffer);
    // }
  }

  // Data members
  int id_;
  std::vector<T> surfaces_;
  std::vector<GPRTBufferOf<typename T::vertex_type>> vertex_buffers_;
  std::vector<GPRTBufferOf<uint3>> connectivity_buffers_;
  std::vector<GPRTGeomOf<G>> gprt_geoms_;
  GPRTAccel blas_;
  GPRTAccel tlas_;
};

template<>
void MBVolume<DPTriangleSurface, DPTriangleData>::setup(GPRTContext context, GPRTModule module, int2 fbSize) {
  // populate AABB buffer
  for (int i = 0; i < surfaces_.size(); i++) {
    auto& surf = surfaces_[i];
    auto& geom = gprt_geoms_[i];

    auto geom_data = gprtGeomGetParameters(geom);
    geom_data->fbSize = fbSize;
    // might need this too?
    // dpGeomData->dpRays = gprtBufferGetHandle(doubleRayBuffer);

    surf.aabb_buffer = gprtDeviceBufferCreate<float3>(context, 2*surf.n_tris, nullptr);
    gprtAABBsSetPositions(geom, surf.aabb_buffer, surf.n_tris, 2*sizeof(float3), 0);
    GPRTComputeOf<DPTriangleData> boundsProg = gprtComputeCreate<DPTriangleData>(context, module, "DPTriangle");
    auto boundsProgData = gprtComputeGetParameters(boundsProg);
    boundsProgData->vertex = gprtBufferGetHandle(vertex_buffers_[i]);
    boundsProgData->index = gprtBufferGetHandle(connectivity_buffers_[i]);
    boundsProgData->aabbs = gprtBufferGetHandle(surf.aabb_buffer);
    gprtBuildShaderBindingTable(context, GPRT_SBT_COMPUTE);
    gprtComputeLaunch1D(context, boundsProg, surf.n_tris);
    surf.aabbs_present = true;
  }
}

template<class T, class G>
struct MBVolumes {
  // Constructors
  MBVolumes(std::vector<int> ids) {
    for (auto id : ids) {
      volumes().push_back(MBVolume<T, G>(id));
    }
  }

  // Methods
  void populate_surfaces(DagMC* dagmc) {
    for (auto& volume : volumes()) {
      volume.populate_surfaces(dagmc);
    }
  }

  void create_geoms(GPRTContext context, GPRTGeomTypeOf<G> g_type) {
    for (auto& volume : volumes()) {
      volume.create_geoms(context, g_type);

    }
  }

  void setup(GPRTContext context, GPRTModule module, int2 fbSize) {
    for (auto& volume : volumes()) {
      volume.setup(context, module, fbSize);
    }
  }

  void create_accel_structures(GPRTContext context) {
    // gather up all BLAS and join into a single TLAS
    std::vector<GPRTAccel> blass;
    for (auto& vol : volumes()) {
      vol.create_accel_structures(context);
      blass.push_back(vol.blas_);
    }
    world_tlas_ = gprtInstanceAccelCreate(context, blass.size(), blass.data());
    gprtAccelBuild(context, world_tlas_, GPRT_BUILD_MODE_FAST_TRACE_NO_UPDATE);

    std::vector<gprt::Accel> accel_ptrs;
    for (auto& vol : volumes()) accel_ptrs.push_back(gprtAccelGetHandle(vol.tlas_));
    // map acceleration pointers to a device buffer
    tlas_buffer_ = gprtDeviceBufferCreate<gprt::Accel>(context, accel_ptrs.size(), accel_ptrs.data());

    // create a map of volume ID to index
    std::map<int, int> vol_id_to_idx_map;
    for (int i = 0; i < volumes().size(); i++) {
      vol_id_to_idx_map[volumes()[i].id_] = i;
    }

    // set surface parent indices into the tlas buffer for each surface
    for (auto& vol : volumes()) {
      for (auto& geom : vol.gprt_geoms_) {
        auto geom_data = gprtGeomGetParameters(geom);
        geom_data->ff_vol = vol_id_to_idx_map[geom_data->vols[0]];
        geom_data->bf_vol = vol_id_to_idx_map[geom_data->vols[1]];
      }
    }
  }

  void cleanup() {
    gprtAccelDestroy(world_tlas_);
    gprtBufferDestroy(tlas_buffer_);
    for (auto& vol : volumes()) {
      vol.cleanup();
    }
  }

  // Accessors
  const auto& volumes() const { return volumes_; }
  auto& volumes() { return volumes_; }

  // Data members
  std::vector<MBVolume<T, G>> volumes_;
  GPRTAccel world_tlas_;
  GPRTBufferOf<gprt::Accel> tlas_buffer_;
};

template<class T>
struct MBTriangleSurfaces {

  // Data members
  std::vector<T> surfaces_;
  std::vector<GPRTAccel> blass_;
};



// Create an object that is a collection of SPTriangle surface objects and can
//  - call any necessary methods for final setup (set_buffers, aabbs, etc.)
//  - create it's own BLAS for all surfaces in the container

// TLAS creation and index mapping into the TLAS should be able to occur on each of these containers
template<class T, class G>
std::map<int, std::vector<T>> setup_surfaces(GPRTContext context, GPRTModule module, std::shared_ptr<moab::DagMC> dag, GPRTGeomTypeOf<G> g_type, std::vector<int> visible_vol_ids = {}) {
    ErrorCode rval;

    int n_surfs = dag->num_entities(2);

    if (n_surfs == 0) {
      std::cerr << "No surfaces were found in the model" << std::endl;
      std::exit(1);
    }

    if (visible_vol_ids.size() == 0) {
      for (int i = 0; i < dag->num_entities(3); i++) visible_vol_ids.push_back(dag->id_by_index(3, i+1));
      // add graveyard explicitly
      EntityHandle gyg;
      Range gys;
      dag->get_graveyard_group(gyg);
      dag->moab_instance()->get_entities_by_type(gyg, MBENTITYSET, gys);
    }

    std::map<int, std::vector<T>> out;
    for (auto vol_id : visible_vol_ids) {
      EntityHandle vol = dag->entity_by_id(3, vol_id);

      if (std::find(visible_vol_ids.begin(), visible_vol_ids.end(), vol_id) == visible_vol_ids.end()) continue;

      Range vol_surfs;
      rval = dag->moab_instance()->get_child_meshsets(vol, vol_surfs);

      std::vector<T> surf_geoms;
      for (auto surf : vol_surfs) {
        int surf_id = dag->id_by_index(2, dag->index_by_handle(surf));
        surf_geoms.emplace_back(std::move(T(context, dag->moab_instance(), g_type, surf_id, vol_id)));
      }
      out[vol_id] = surf_geoms;
    }
    return out;
}

std::pair<double3, double3> bounding_box(Interface* mbi) {
  ErrorCode rval;

  Range all_verts;
  rval = mbi->get_entities_by_dimension(0, 0, all_verts, true);
  MB_CHK_SET_ERR_CONT(rval, "Failed to retrieve all vertices");

  std::vector<double> coords(3*all_verts.size());
  rval = mbi->get_coords(all_verts, coords.data());
  MB_CHK_SET_ERR_CONT(rval, "Failed to get vertex coordinates");

  double3 aabbmin = double3(coords[0], coords[1], coords[2]);
  double3 aabbmax = aabbmin;
  for (uint32_t i = 1; i < all_verts.size(); ++i) {
    aabbmin = linalg::min(aabbmin, double3(coords[i * 3 + 0],
                                           coords[i * 3 + 1],
                                           coords[i * 3 + 2]));
    aabbmax = linalg::max(aabbmax, double3(coords[i * 3 + 0],
                                           coords[i * 3 + 1],
                                           coords[i * 3 + 2]));
  }

  return {aabbmin, aabbmax};
}