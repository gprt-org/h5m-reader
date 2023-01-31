#ifndef _MBDIRECTACCESS_
#define _MBDIRECTACCESS_

#include <array>
#include <memory>

// MOAB
#include "moab/Core.hpp"
#include "moab/CartVect.hpp"

using namespace moab;

/*! Class to manage direct access of triangle connectivity and coordinates */
class MBDirectAccess {

public:
  // constructor
  MBDirectAccess(Interface* mbi) : mbi(mbi) {};

  MBDirectAccess(std::shared_ptr<Interface> mbi) : mbi(mbi.get()) {};

  //! \brief Initialize internal structures
  void setup(std::vector<int> vol_ids = {});

  //! \brief Reset internal data structures, but maintain MOAB isntance
  void clear();

  //! \brief Update internal data structures to account for changes in the MOAB instance
  void update();

  //! \brief Check that a triangle is part of the managed coordinates here
  inline bool accessible(EntityHandle tri) {
    // determine the correct index to use
    int idx = 0;
    auto fe = first_elements_[idx];
    while(true) {
      if (tri - fe.first < fe.second) { break; }
      idx++;
      if (idx >= first_elements_.size()) { return false; }
      fe = first_elements_[idx];
    }
    return true;
  }

  const double** get_coord_ptrs(const EntityHandle &elem) {

    auto fe_it = first_elements_.begin();
    size_t conn_offset {0};

    while(true) {
      if (elem - fe_it->first < fe_it->second) { break; }
      conn_offset += fe_it->second;
      fe_it++;
    }

    conn_offset += elem - fe_it->first;

    const double* i0 = xyz_.data() + conn_[conn_offset];
    const double* i1 = xyz_.data() + conn_[conn_offset + 1];
    const double* i2 = xyz_.data() + conn_[conn_offset + 2];

    return nullptr;
  }

  // Accessors
  //! \brief return the number of elements being managed
  inline int n_elements() { return num_elements_; }
  //! \brief return the number of vertices being managed
  inline int n_vertices() { return num_vertices_; }
  //! \brief return the stride between elements in the coordinate arrays
  inline int stride() { return element_stride_;}

  inline std::vector<double>& xyz() { return xyz_; }
  inline const std::vector<double>& xyz() const { return xyz_; }
  inline std::vector<int>& conn() { return conn_; }
  inline const std::vector<int>& conn() const { return conn_; }

private:
  Interface* mbi {nullptr}; //!< MOAB instance for the managed data
  int num_elements_ {-1}; //!< Number of elements in the manager
  int num_vertices_ {-1}; //!< Number of vertices in the manager
  int element_stride_ {-1}; //!< Number of vertices used by each element
  std::vector<std::pair<EntityHandle, size_t>> first_elements_; //!< Pairs of first element and length pairs for contiguous blocks of memorys
  std::vector<double> xyz_; //!< Storage location for vertex data (optionally used)
  std::vector<int32_t> conn_; //!< Storage location for connectivity data (optionally used)
  bool internal_storage_; //!< indicates whether or not information is stored internally
};

#endif // include guard