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
  void setup(bool store_internal=false);

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

  //! \brief Get the coordinates of a triangle as MOAB CartVect's
  inline std::array<moab::CartVect, 3> get_mb_coords(const EntityHandle& tri) {

    // determine the correct index to use
    int idx = 0;
    auto fe = first_elements_[idx];

    while(true) {
      if (tri - fe.first < fe.second) { break; }
      idx++;
      fe = first_elements_[idx];
    }

    size_t conn_idx = element_stride_ * (tri - fe.first);
    size_t i0 = vconn_[idx][conn_idx] - 1;
    size_t i1 = vconn_[idx][conn_idx + 1] - 1;
    size_t i2 = vconn_[idx][conn_idx + 2] - 1;

    moab::CartVect v0(tx_[idx][i0], ty_[idx][i0], tz_[idx][i0]);
    moab::CartVect v1(tx_[idx][i1], ty_[idx][i1], tz_[idx][i1]);
    moab::CartVect v2(tx_[idx][i2], ty_[idx][i2], tz_[idx][i2]);

    return {v0, v1, v2};
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
  std::vector<std::pair<EntityHandle, size_t>> first_elements_; //!< Pairs of first element and length pairs for contiguous blocks of memory
  std::vector<const EntityHandle*> vconn_; //!< Storage array(s) for the connectivity array
  std::vector<double*> tx_; //!< Storage array(s) for vertex x coordinates
  std::vector<double*> ty_; //!< Storage array(s) for vertex y coordinates
  std::vector<double*> tz_; //!< Storage array(s) for vertex z coordinates

  std::vector<double> xyz_; //!< Storage location for vertex data (optionally used)
  std::vector<int32_t> conn_; //!< Storage location for connectivity data (optionally used)
};

#endif // include guard