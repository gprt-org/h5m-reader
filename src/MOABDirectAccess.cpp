#include <iostream>
#include <sstream>

// MOAB
#include "moab/Range.hpp"

#include "MOABDirectAccess.h"
#include "MBTagConventions.hpp"

void
MBDirectAccess::setup(std::vector<int> vol_ids) {
  ErrorCode rval;

  Range tris;

  if (vol_ids.size() != 0) {
    // get the category tag handle
    Tag dim_tag;
    rval = mbi->tag_get_handle(GEOM_DIMENSION_TAG_NAME, dim_tag);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get category tag handle");

    Tag id_tag = mbi->globalId_tag();

    for (int vol_id : vol_ids) {
      int dim = 3;
      const Tag tags[] = {id_tag, dim_tag};
      const void* const vals[] = {&vol_id, &dim};

      // retrieve all volume sets from the MOAB model
      Range vol_sets;
      rval = mbi->get_entities_by_type_and_tag(0, MBENTITYSET, tags, vals, 2, vol_sets  );
      MB_CHK_SET_ERR_CONT(rval, "Failed to retrieve volume sets");

      if (vol_sets.size() == 0) {
        std::cerr << "Could not find volume with ID: " << vol_id << std::endl;
        std::exit(1);
      }

      if (vol_sets.size() > 1) {
        std::cerr << "Found more than one volume with ID: " << vol_id << std::endl;
      }

      std::cout << "Vol sets size: " <<  vol_sets.size() << std::endl;
      // get the ID of the volume
      // std::vector<int> vol_ids(vol_sets.size());

      // // display the volume ID of each discovered volume
      // rval = mbi->tag_get_data(id_tag, vol_sets, vol_ids.data());
      // MB_CHK_SET_ERR_CONT(rval, "Failed to get volume ID tag data");

      Range children;
      rval = mbi->get_child_meshsets(vol_sets[0], children);
      MB_CHK_SET_ERR_CONT(rval, "Failed to get child meshsets (surface sets) of volume");

      for (auto child : children) {
        rval = mbi->get_entities_by_dimension(child, 2, tris, true);
        MB_CHK_SET_ERR_CONT(rval, "Failed to get triangle elements on surface");
      }
    }
  } else {
    // setup triangles
    rval = mbi->get_entities_by_dimension(0, 2, tris, true);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get all elements of dimension 2 (tris)");
    num_elements_ = tris.size();
  }

  // get the first global triangle
  Range all_tris;
  rval = mbi->get_entities_by_dimension(0, 2, all_tris, true);
  MB_CHK_SET_ERR_CONT(rval, "Failed to get all elements of dimension 2 (tris)");

  // only supporting triangle elements for now
  if (!tris.all_of_type(MBTRI)) { throw std::runtime_error("Not all 2D elements are triangles"); }

  conn_.resize(3*tris.size());

  moab::Range::iterator tris_it = tris.begin();
  while(tris_it != tris.end()) {
    // set connectivity pointer, element stride and the number of elements
    EntityHandle* conntmp;
    int n_elements;
    rval = mbi->connect_iterate(tris_it, tris.end(), conntmp, element_stride_, n_elements);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get direct access to triangle elements");

    // add first element/length pair to the set of first elements
    first_elements_.push_back({*tris_it, n_elements});

    // offset is always relative to first global triangle
    int offset = tris_it - tris.begin();
    offsets_.push_back({offset, n_elements});
    for (int i = 0; i < 3*n_elements; i++) conn_[3*offset + i] = conntmp[i] - all_tris.front();

    // move iterator forward by the number of triangles in this contiguous memory block
    tris_it += n_elements;
  }

  // setup vertices
  Range verts;
  rval = mbi->get_entities_by_dimension(0, 0, verts, true);
  MB_CHK_SET_ERR_CONT(rval, "Failed to get all elements of dimension 0 (vertices)");
  num_vertices_ = verts.size();

  xyz_.resize(3*num_vertices_);

  moab::Range::iterator verts_it = verts.begin();
  while (verts_it != verts.end()) {
    // set vertex coordinate pointers
    double* xtmp;
    double* ytmp;
    double* ztmp;
    int n_vertices;
    rval = mbi->coords_iterate(verts_it, verts.end(), xtmp, ytmp, ztmp, n_vertices);
    MB_CHK_SET_ERR_CONT(rval, "Failed to get direct access to vertex elements");

    // add the vertex coordinate arrays to their corresponding vector of array pointers
    int offset = 3 * (verts_it - verts.begin());
    for (int i = 0; i < n_vertices; i++) {
      xyz_[offset + 3 * i] = xtmp[i];
      xyz_[offset + 3 * i + 1] = ytmp[i];
      xyz_[offset + 3 * i + 2] = ztmp[i];
    }
    // move iterator forward by the number of vertices in this contiguous memory block
    verts_it += n_vertices;
  }
}

void
MBDirectAccess::clear()
{
  num_elements_ = -1;
  num_vertices_ = -1;
  element_stride_ = -1;

  first_elements_.clear();

  xyz_.clear();
  conn_.clear();
}

void
MBDirectAccess::update() {
  clear();
  setup();
}
