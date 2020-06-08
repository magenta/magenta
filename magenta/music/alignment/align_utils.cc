#include <assert.h>
#include <stdlib.h>
#include <random>
#include <iostream>

#include "./align_utils.h"

#include "absl/container/flat_hash_map.h"

using std::cerr;
using std::cout;
using std::endl;

namespace magenta {

namespace {

enum MoveType {
  DIAGONAL = 0,
  HORIZONTAL = 1,
  VERTICAL = 2,
};

typedef Eigen::Matrix<MoveType, Eigen::Dynamic, Eigen::Dynamic> EigenMatrixXMT;

typedef std::pair<int, int> Coord;
typedef absl::flat_hash_map<Coord, double> CoordMap;

// Computes the cost and backtrace matrices using DTW.
void ComputeDtwCosts(const Eigen::MatrixXd& distance_matrix, double penalty,
                     Eigen::MatrixXd* cost_matrix,
                     EigenMatrixXMT* backtrace_matrix) {
  *cost_matrix = distance_matrix;
  backtrace_matrix->resize(distance_matrix.rows(), distance_matrix.cols());

  // Initialize first row and column so as to force all paths to start from
  // (0, 0) by making all moves on top and left edges lead back to the origin.
  for (int i = 1; i < distance_matrix.rows(); ++i) {
    (*cost_matrix)(i, 0) += (*cost_matrix)(i - 1, 0) + penalty;
    (*backtrace_matrix)(i, 0) = VERTICAL;
  }
  for (int j = 1; j < distance_matrix.cols(); ++j) {
    (*cost_matrix)(0, j) += (*cost_matrix)(0, j - 1) + penalty;
    (*backtrace_matrix)(0, j) = HORIZONTAL;
  }

  for (int i = 1; i < distance_matrix.rows(); ++i) {
    for (int j = 1; j < distance_matrix.cols(); ++j) {
      const double diagonal_cost = (*cost_matrix)(i - 1, j - 1);
      const double horizontal_cost = (*cost_matrix)(i, j - 1) + penalty;
      const double vertical_cost = (*cost_matrix)(i - 1, j) + penalty;
      if (diagonal_cost <= std::min(horizontal_cost, vertical_cost)) {
        // Diagonal move (which has no penalty) is lowest.
        (*cost_matrix)(i, j) += diagonal_cost;
        (*backtrace_matrix)(i, j) = DIAGONAL;
      } else if (horizontal_cost <= vertical_cost) {
        // Horizontal move (which has penalty) is lowest.
        (*cost_matrix)(i, j) += horizontal_cost;
        (*backtrace_matrix)(i, j) = HORIZONTAL;
      } else if (vertical_cost < horizontal_cost) {
        // Vertical move (which has penalty) is lowest.
        (*cost_matrix)(i, j) += vertical_cost;
        (*backtrace_matrix)(i, j) = VERTICAL;
      } else {
        cerr << "Invalid state." << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
}

int GetSafeBandRadius(const Eigen::MatrixXd& sequence_1,
                      const Eigen::MatrixXd& sequence_2,
                      const int band_radius) {
  double slope = static_cast<double>(sequence_2.cols()) / sequence_1.cols();

  return std::max(band_radius, static_cast<int>(ceil(slope)));
}

bool IsInBand(const Eigen::MatrixXd& sequence_1,
              const Eigen::MatrixXd& sequence_2, const int band_radius,
              Coord coord) {
  if (band_radius < 0) {
    return true;
  }
  double slope = static_cast<double>(sequence_2.cols()) / sequence_1.cols();

  double y = coord.first * slope;
  return std::abs(y - coord.second) <= band_radius;
}

double GetDistance(const Eigen::MatrixXd& sequence_1,
                   const Eigen::MatrixXd& sequence_2, const Coord& coord) {
  auto s1 = sequence_1.transpose().block(coord.first, 0, 1,
                                         sequence_1.transpose().cols());
  auto s2 = sequence_2.block(0, coord.second, sequence_2.rows(), 1);
  auto distance = 1.0 - (s1.array() * s2.transpose().array()).sum();
  // cout << coord.first << "," << coord.second << ": " << distance << endl;
  return distance;
}

double GetDistanceOrInfinity(const Eigen::MatrixXd& sequence_1,
                             const Eigen::MatrixXd& sequence_2,
                             const int band_radius, const Coord& coord) {
  if (!IsInBand(sequence_1, sequence_2, band_radius, coord)) {
    return std::numeric_limits<double>::infinity();
  }

  return GetDistance(sequence_1, sequence_2, coord);
}

double GetCostOrInfinity(const CoordMap& cost_matrix, const Coord& coord,
                         double penalty_mul, double penalty_add) {
  if (cost_matrix.count(coord) == 0) {
    return std::numeric_limits<double>::infinity();
  } else {
    return cost_matrix.at(coord) * penalty_mul + penalty_add;
  }
}

// Computes the cost and backtrace matrices using DTW.
void ComputeDtwCostsOnDemand(const Eigen::MatrixXd& sequence_1,
                             const Eigen::MatrixXd& sequence_2,
                             const int band_radius, const double penalty_mul,
                             const double penalty_add, CoordMap* cost_matrix,
                             CoordMap* backtrace_matrix) {
  cout << "Computing DTW costs with penalty " << penalty_add
       << " and penalty multiplier " << penalty_mul << endl;
  int safe_band_radius = -1;
  if (band_radius >= 0) {
    safe_band_radius = GetSafeBandRadius(sequence_1, sequence_2, band_radius);
    if (safe_band_radius != band_radius) {
      cout << "Increasing band radius from " << band_radius << " to "
           << safe_band_radius << " to ensure a continuous path." << endl;
    }

    cost_matrix->reserve(sequence_1.cols() * safe_band_radius * 2);
  } else {
    cost_matrix->reserve(sequence_1.cols() * sequence_2.cols());
  }

  // Initialize starting point.
  (*cost_matrix)[Coord(0, 0)] = GetDistanceOrInfinity(
      sequence_1, sequence_2, safe_band_radius, Coord(0, 0));

  // Initialize first row and column so as to force all paths to start from
  // (0, 0) by making all moves on top and left edges lead back to the origin.
  for (int i = 1; i < sequence_1.cols(); ++i) {
    (*cost_matrix)[Coord(i, 0)] =
        GetDistanceOrInfinity(sequence_1, sequence_2, safe_band_radius,
                              Coord(i, 0)) +
        cost_matrix->at(Coord(i - 1, 0)) * penalty_mul + penalty_add;
    (*backtrace_matrix)[Coord(i, 0)] = VERTICAL;
  }
  for (int j = 1; j < sequence_2.cols(); ++j) {
    (*cost_matrix)[Coord(0, j)] =
        GetDistanceOrInfinity(sequence_1, sequence_2, safe_band_radius,
                              Coord(0, j)) +
        cost_matrix->at(Coord(0, j - 1)) * penalty_mul + penalty_add;
    (*backtrace_matrix)[Coord(0, j)] = HORIZONTAL;
  }

  int cur_percent = -1;
  for (int i = 1; i < sequence_1.cols(); ++i) {
    int new_percent =
        static_cast<int>(100.0 * static_cast<double>(i) / sequence_1.cols());
    if (new_percent != cur_percent && new_percent % 10 == 0) {
      cout << "Processing... " << new_percent << "%" << endl;
      cur_percent = new_percent;
    }
    for (int j = 1; j < sequence_2.cols(); ++j) {
      if (!IsInBand(sequence_1, sequence_2, safe_band_radius, Coord(i, j))) {
        // cout << "skipping " << i << "," << j << endl;
        continue;
      } else {
        // cout << "calculating " << i << "," << j << endl;
      }
      const double diagonal_cost =
          GetCostOrInfinity(*cost_matrix, Coord(i - 1, j - 1),
                            /*penalty_mul=*/1, /*penalty_add=*/0);
      const double horizontal_cost = GetCostOrInfinity(
          *cost_matrix, Coord(i, j - 1), penalty_mul, penalty_add);
      const double vertical_cost = GetCostOrInfinity(
          *cost_matrix, Coord(i - 1, j), penalty_mul, penalty_add);
      auto cur_distance = GetDistanceOrInfinity(sequence_1, sequence_2,
                                                safe_band_radius, Coord(i, j));
      if (diagonal_cost <= std::min(horizontal_cost, vertical_cost)) {
        // Diagonal move (which has no penalty) is lowest.
        (*cost_matrix)[Coord(i, j)] = diagonal_cost + cur_distance;
        (*backtrace_matrix)[Coord(i, j)] = DIAGONAL;
        // cout << i << "," << j << ": diagonal - "
        //      << (*cost_matrix)[Coord(i, j)] << endl;
      } else if (horizontal_cost <= vertical_cost) {
        // Horizontal move (which has penalty) is lowest.
        (*cost_matrix)[Coord(i, j)] = horizontal_cost + cur_distance;
        (*backtrace_matrix)[Coord(i, j)] = HORIZONTAL;
        // cout << i << "," << j << ": horizontal - "
        //      << (*cost_matrix)[Coord(i, j)] << endl;
      } else if (vertical_cost < horizontal_cost) {
        // Vertical move (which has penalty) is lowest.
        (*cost_matrix)[Coord(i, j)] = vertical_cost + cur_distance;
        (*backtrace_matrix)[Coord(i, j)] = VERTICAL;
        // cout << i << "," << j << ": vertical - "
        //      << (*cost_matrix)[Coord(i, j)] << endl;
      } else {
        cerr << "Invalid state at " << i << ", " << j
             << ". diagonal_cost: " << diagonal_cost
             << ", horizontal_cost: " << horizontal_cost
             << ", vertical_cost: " << vertical_cost << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
}

// Backtracks from the specified endpoint to fill the indices along the lowest-
// cost path.
void GetDtwAlignedIndices(const Eigen::MatrixXd& cost_matrix,
                          const EigenMatrixXMT& backtrace_matrix,
                          std::vector<int>* i_indices,
                          std::vector<int>* j_indices) {
  assert(i_indices != nullptr);
  assert(j_indices != nullptr);

  int i = cost_matrix.rows() - 1;
  int j = cost_matrix.cols() - 1;

  // Start from the end of the path.
  *i_indices = {i};
  *j_indices = {j};

  // Until we reach the origin.
  while (i > 0 || j > 0) {
    if (backtrace_matrix(i, j) == DIAGONAL) {
      --i;
      --j;
    } else if (backtrace_matrix(i, j) == HORIZONTAL) {
      --j;
    } else if (backtrace_matrix(i, j) == VERTICAL) {
      --i;
    }
    // Add new indices into the path arrays.
    i_indices->push_back(i);
    j_indices->push_back(j);
  }
  // Reverse the path index arrays.
  std::reverse(i_indices->begin(), i_indices->end());
  std::reverse(j_indices->begin(), j_indices->end());
}

// Backtracks from the specified endpoint to fill the indices along the lowest-
// cost path.
void GetDtwAlignedIndicesOnDemand(const CoordMap& cost_matrix,
                                  const CoordMap& backtrace_matrix,
                                  const int s1_cols, const int s2_cols,
                                  std::vector<int>* i_indices,
                                  std::vector<int>* j_indices) {
  assert(i_indices != nullptr);
  assert(j_indices != nullptr);

  int i = s1_cols - 1;
  int j = s2_cols - 1;

  // Start from the end of the path.
  i_indices->assign({i});
  j_indices->assign({j});
  // We'll need at least as many steps as i.
  i_indices->reserve(i);
  j_indices->reserve(i);

  // Until we reach the origin.
  while (i > 0 || j > 0) {
    if (backtrace_matrix.at(Coord(i, j)) == DIAGONAL) {
      --i;
      --j;
    } else if (backtrace_matrix.at(Coord(i, j)) == HORIZONTAL) {
      --j;
    } else if (backtrace_matrix.at(Coord(i, j)) == VERTICAL) {
      --i;
    }
    // Add new indices into the path arrays.
    i_indices->push_back(i);
    j_indices->push_back(j);
  }
  // Reverse the path index arrays.
  std::reverse(i_indices->begin(), i_indices->end());
  std::reverse(j_indices->begin(), j_indices->end());
}

}  // namespace

double AlignWithDynamicTimeWarping(const Eigen::MatrixXd& distance_matrix,
                                   double penalty, std::vector<int>* i_indices,
                                   std::vector<int>* j_indices) {
  assert(i_indices != nullptr);
  assert(j_indices != nullptr);

  Eigen::MatrixXd cost_matrix;
  EigenMatrixXMT backtrace_matrix;
  ComputeDtwCosts(distance_matrix, penalty, &cost_matrix, &backtrace_matrix);
  GetDtwAlignedIndices(cost_matrix, backtrace_matrix, i_indices, j_indices);
  return cost_matrix(cost_matrix.rows() - 1, cost_matrix.cols() - 1);
}

double AlignWithDynamicTimeWarpingOnDemand(
    const Eigen::MatrixXd& sequence_1, const Eigen::MatrixXd& sequence_2,
    const int band_radius, const double penalty_mul, const double penalty_add,
    std::vector<int>* i_indices, std::vector<int>* j_indices,
    int distance_samples) {
  assert(sequence_1.rows() == sequence_2.rows());
  assert(i_indices != nullptr);
  assert(j_indices != nullptr);

  // Normalize both sequences.
  auto s1_norm = sequence_1;
  auto s2_norm = sequence_2;
  s1_norm.colwise().normalize();
  s2_norm.colwise().normalize();

  double mean_distance = 0;
  if (distance_samples > 0) {
    cout << "Finding mean distance with " << distance_samples
         << " distance samples..." << endl;

    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    // Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> s1col_dis(0, s1_norm.cols() - 1);
    std::uniform_int_distribution<> s2col_dis(0, s2_norm.cols() - 1);

    double total_distance = 0;
    for (int i = 0; i < distance_samples; i++) {
      int s1col = s1col_dis(gen);
      int s2col = s2col_dis(gen);
      total_distance += GetDistance(s1_norm, s2_norm, Coord(s1col, s2col));
    }
    mean_distance = total_distance / distance_samples;
    cout << "Mean distance: " << mean_distance << endl;
  }

  CoordMap cost_matrix;
  CoordMap backtrace_matrix;
  ComputeDtwCostsOnDemand(s1_norm, s2_norm, band_radius, penalty_mul,
                          penalty_add + mean_distance, &cost_matrix,
                          &backtrace_matrix);

  GetDtwAlignedIndicesOnDemand(cost_matrix, backtrace_matrix, s1_norm.cols(),
                               s2_norm.cols(), i_indices, j_indices);
  return cost_matrix[Coord(s1_norm.cols() - 1, s2_norm.cols() - 1)];
}

Eigen::MatrixXd GetCosineDistanceMatrix(const Eigen::MatrixXd& sequence_1,
                                        const Eigen::MatrixXd& sequence_2) {
  assert(sequence_1.rows() == sequence_2.rows());

  Eigen::MatrixXd distance_matrix = sequence_1.transpose() * sequence_2;
  distance_matrix.array() /=
      (sequence_1.colwise().norm().transpose() * sequence_2.colwise().norm())
          .array();
  distance_matrix.array() = 1 - distance_matrix.array();
  return distance_matrix;
}

}  // namespace magenta
