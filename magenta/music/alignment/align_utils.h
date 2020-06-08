#ifndef MAGENTA_MUSIC_ALIGNMENT_ALIGN_UTILS_H_
#define MAGENTA_MUSIC_ALIGNMENT_ALIGN_UTILS_H_

#include <vector>
#include "Eigen/Core"

namespace magenta {

// Aligns two sequences using Dynamic Time Warping given their distance matrix,
// returning the alignment score and filling vectors with the aligned indices.
// The score for non-diagonal moves is computed as the value of the non-diagonal
// position plus the penalty. The path is to include the entirety of both
// sequences.
double AlignWithDynamicTimeWarping(const Eigen::MatrixXd& distance_matrix,
                                   double penalty, std::vector<int>* i_indices,
                                   std::vector<int>* j_indices);

// Computes and fills the cosine distance matrix between two sequences whose
// columns represent time and rows frequencies. The sequences must have an
// equivalent number of rows.
Eigen::MatrixXd GetCosineDistanceMatrix(const Eigen::MatrixXd& sequence_1,
                                        const Eigen::MatrixXd& sequence_2);

// Aligns two sequences using Dynamic Time Warping.
// The distance between the two sequences is calculated on demand, which is
// more memory-efficient when a band radius is used. The band_radius specifies
// how many steps away from the diagonal path the warp path is allowed to go.
// If it is < 0, it is ignored.
// When alignment is complete, the alignment score is returned and the vectors
// are filled with the aligned indices.
// The score for non-diagonal moves is computed as the value of the non-diagonal
// position plus the penalty. The path is to include the entirety of both
// sequences.
double AlignWithDynamicTimeWarpingOnDemand(const Eigen::MatrixXd& sequence_1,
                                           const Eigen::MatrixXd& sequence_2,
                                           int band_radius, double penalty_mul,
                                           double penalty_add,
                                           std::vector<int>* i_indices,
                                           std::vector<int>* j_indices,
                                           int distance_samples = 100000);
}  // namespace magenta

#endif  // MAGENTA_MUSIC_ALIGNMENT_ALIGN_UTILS_H_
