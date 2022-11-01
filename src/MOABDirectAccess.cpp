#include <sstream>

// MOAB
#include "moab/Range.hpp"

#include "MOABDirectAccess.h"

void
MBDirectAccess::setup() {
  ErrorCode rval;

  // setup triangles
  Range tris;
  rval = mbi->get_entities_by_dimension(0, 2, tris, true);
  MB_CHK_SET_ERR_CONT(rval, "Failed to get all elements of dimension 2 (tris)");
  num_elements_ = tris.size();

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

    int offset = tris_it - tris.begin();
    for (int i = 0; i < 3*n_elements; i++) conn_[offset + i] = conntmp[i] - 1;

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
