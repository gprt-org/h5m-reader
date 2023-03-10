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

  int id;
  int n_tris;
  int frontface_vol;
  int backface_vol;
  std::vector<R> vertices;
  std::vector<uint3> connectivity;
  GPRTBufferOf<R> vertex_buffer_s;
  GPRTBufferOf<uint3> conn_buffer;
  GPRTBufferOf<float3> aabb_buffer;
  GPRTGeomOf<T> triangle_geom_s;
  bool aabbs_present {false};

  struct SurfaceData {
    std::vector<double> coords;
    std::vector<uint3> connectivity;
  };

  MBTriangleSurface(GPRTContext context, moab::Interface* mbi, GPRTGeomTypeOf<T> g_type, int surface_id, int vol_id=-1) {
    ErrorCode rval;

    // get this surface's handle
    Tag dim_tag;
    rval = mbi->tag_get_handle(GEOM_DIMENSION_TAG_NAME, dim_tag);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get the geom dim tag");

    Tag id_tag = mbi->globalId_tag();

    int dim = 2;
    const Tag tags[] = {id_tag, dim_tag};
    const void* const vals[] = {&surface_id, &dim};

    Range surf_sets;
    rval = mbi->get_entities_by_type_and_tag(0, MBENTITYSET, tags, vals, 2, surf_sets);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get surface with id " << surface_id);

    if (surf_sets.size() != 1) {
        std::cerr << "Incorrect number of surfaces found (" << surf_sets.size() << ") with ID " << surface_id << std::endl;
        std::exit(1);
    }

    EntityHandle surf_handle = surf_sets[0];

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
    rval = mbi->tag_get_data(id_tag, parent_vols.data(), parents_size, parent_ids.data());
    MB_CHK_SET_ERR_CONT(rval, "Failed to get parent volume IDs");

    // if we're building up this surface and it has a reverse sense relative to the volume provided,
    // flip the triangle normals by changing the triangle connectivity
    bool sense_reverse = parent_ids[1] == vol_id;
    id = sense_reverse ? -surface_id : surface_id;
    if (sense_reverse) {
      for (int i = 0; i < surf_tris.size(); i++) {
        auto& conn = connectivity[i];
        std::swap(conn[1], conn[2]);
      }
    }

    if (id == DEBUG_SURF) {
    std::cout << "----------------" << std::endl;
    std::cout << "Surface Creation Stage" << std::endl;
    std::cout << "Surface " << id << ", natural parent vols: " << parent_ids[0] << ", " << parent_ids[1] << std::endl;
    }

    vertex_buffer_s = gprtDeviceBufferCreate<R>(context, vertices.size(), vertices.data());
    conn_buffer = gprtDeviceBufferCreate<uint3>(context, connectivity.size(), connectivity.data());

    triangle_geom_s = gprtGeomCreate<T>(context, g_type);

    T* geom_data = gprtGeomGetParameters(triangle_geom_s);
    geom_data->vertex = gprtBufferGetHandle(vertex_buffer_s);
    geom_data->index = gprtBufferGetHandle(conn_buffer);
    geom_data->id = id;

    if (surface_id == DEBUG_SURF) std::cout << "SPTriangleGeom ID: " << geom_data->id << std::endl;

    if (sense_reverse) {
      geom_data->vols[0] = parent_ids[1];
      geom_data->vols[1] = parent_ids[0];
    } else {
      geom_data->vols[0] = parent_ids[0];
      geom_data->vols[1] = parent_ids[1];
    }

    if (id == DEBUG_SURF)
      std::cout << "Geom data vols: " << geom_data->vols << std::endl;

  }

  void aabbs(GPRTContext context, GPRTModule module) {
    aabb_buffer = gprtDeviceBufferCreate<float3>(context, 2*n_tris, nullptr);
    gprtAABBsSetPositions(triangle_geom_s, aabb_buffer, n_tris, 2*sizeof(float3), 0);

    T* geom_data = gprtGeomGetParameters(triangle_geom_s);
    geom_data->aabbs = gprtBufferGetHandle(aabb_buffer);

    GPRTComputeOf<T> boundsProg = gprtComputeCreate<T>(context, module, "DPTriangle");
    auto boundsProgData = gprtComputeGetParameters(boundsProg);
    boundsProgData->vertex = gprtBufferGetHandle(vertex_buffer_s);
    boundsProgData->index = gprtBufferGetHandle(conn_buffer);
    boundsProgData->aabbs = gprtBufferGetHandle(aabb_buffer);

    gprtBuildShaderBindingTable(context, GPRT_SBT_COMPUTE);
    gprtComputeLaunch1D(context, boundsProg, n_tris);
    aabbs_present = true;
  }

  void set_buffers() {
    gprtTrianglesSetVertices(triangle_geom_s, vertex_buffer_s, vertices.size());
    gprtTrianglesSetIndices(triangle_geom_s, conn_buffer, connectivity.size());
  }

  void cleanup() {
    gprtGeomDestroy(triangle_geom_s);
    gprtBufferDestroy(vertex_buffer_s);
    gprtBufferDestroy(conn_buffer);
  }
};

using SPTriangleSurface = MBTriangleSurface<SPTriangleData, float3>;
using DPTriangleSurface = MBTriangleSurface<DPTriangleData, double3>;

template<class T, class G>
std::map<int, std::vector<T>> setup_surfaces(GPRTContext context, std::shared_ptr<moab::DagMC> dag, GPRTGeomTypeOf<G> g_type, std::vector<int> visible_vol_ids = {}) {
    ErrorCode rval;

    // get this surface's handle
    Tag dim_tag;
    rval = dag->moab_instance()->tag_get_handle(GEOM_DIMENSION_TAG_NAME, dim_tag);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get the geom dim tag");

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
//      visible_vol_ids.push_back(dag->get_entity_id(gys[0]));
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