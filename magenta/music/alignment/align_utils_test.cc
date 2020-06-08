#include "./align_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

namespace magenta {

class AlignUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_distance_matrix_.resize(3, 4);
    test_distance_matrix_.row(0) << 3.0, 1.5, 5.0, 2.0;
    test_distance_matrix_.row(1) << 0.5, 1.0, 2.0, 0.5;
    test_distance_matrix_.row(2) << 5.0, 0.2, 2.1, 2.0;
  }

  Eigen::MatrixXd test_distance_matrix_;
};

TEST_F(AlignUtilsTest, YesPenalty) {
  std::vector<int> i_indices;
  std::vector<int> j_indices;
  double score = AlignWithDynamicTimeWarping(
      test_distance_matrix_, /*penalty=*/2.0, &i_indices, &j_indices);

  EXPECT_DOUBLE_EQ(10.0, score);
  EXPECT_THAT(i_indices, ElementsAre(0, 1, 1, 2));
  EXPECT_THAT(j_indices, ElementsAre(0, 1, 2, 3));
}

TEST_F(AlignUtilsTest, NoPenalty) {
  std::vector<int> i_indices;
  std::vector<int> j_indices;
  double score = AlignWithDynamicTimeWarping(
      test_distance_matrix_, /*penalty=*/0.0, &i_indices, &j_indices);

  EXPECT_DOUBLE_EQ(7.8, score);
  EXPECT_THAT(i_indices, ElementsAre(0, 1, 2, 2, 2));
  EXPECT_THAT(j_indices, ElementsAre(0, 0, 1, 2, 3));
}

TEST_F(AlignUtilsTest, NoPenalty_Transposed) {
  std::vector<int> i_indices;
  std::vector<int> j_indices;
  double score =
      AlignWithDynamicTimeWarping(test_distance_matrix_.transpose(),
                                  /*penalty=*/0.0, &i_indices, &j_indices);

  EXPECT_DOUBLE_EQ(7.8, score);
  EXPECT_THAT(i_indices, ElementsAre(0, 0, 1, 2, 3));
  EXPECT_THAT(j_indices, ElementsAre(0, 1, 2, 2, 2));
}

TEST_F(AlignUtilsTest, FillCosineDistanceMatrix) {
  Eigen::MatrixXd test_sequence_1(2, 3);
  test_sequence_1.row(0) << -1, 0, 1;
  test_sequence_1.row(1) << 3, 4, 5;

  Eigen::MatrixXd test_sequence_2(2, 4);
  test_sequence_2.row(0) << 6, 5, 4, 3;
  test_sequence_2.row(1) << 1, 0, -1, -2;

  Eigen::MatrixXd distance_matrix =
      GetCosineDistanceMatrix(test_sequence_1, test_sequence_2);

  Eigen::MatrixXd expected_distance_matrix(3, 4);
  expected_distance_matrix.row(0) << 1.156, 1.316, 1.537, 1.789;
  expected_distance_matrix.row(1) << 0.836, 1.000, 1.243, 1.555;
  expected_distance_matrix.row(2) << 0.645, 0.804, 1.048, 1.381;
  ASSERT_TRUE(distance_matrix.isApprox(expected_distance_matrix, 0.001));
}

TEST_F(AlignUtilsTest, OnDemandGlobalDiagonal) {
  std::vector<int> i_indices;
  std::vector<int> j_indices;

  Eigen::MatrixXd sequence_1(1, 10);
  Eigen::MatrixXd sequence_2(1, 10);
  for (int i = 0; i < 10; i += 1) {
    sequence_1.col(i) << (i + 1.0) / 10.0;
    sequence_2.col(i) << (i + 1.0) / 10.0;
  }

  AlignWithDynamicTimeWarpingOnDemand(
      sequence_1, sequence_2, /*band_radius=*/-1, /*penalty_mul=*/1.0,
      /*penalty_add=*/1.0, &i_indices, &j_indices);

  EXPECT_THAT(i_indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
  EXPECT_THAT(j_indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST_F(AlignUtilsTest, OnDemandBandRad) {
  std::vector<int> i_indices;
  std::vector<int> j_indices;

  Eigen::MatrixXd sequence_1(2, 7);
  Eigen::MatrixXd sequence_2(2, 7);
  for (int i = 0; i < sequence_1.cols(); i += 1) {
    sequence_1.col(i) << (i % 5) + 1, (i % 5) + 2;
    sequence_2.col(i) << ((i + 2) % 5) + 1, ((i + 2) % 5) + 2;
  }

  AlignWithDynamicTimeWarpingOnDemand(
      sequence_1, sequence_2, /*band_radius=*/-1, /*penalty_mul=*/1.0,
      /*penalty_add=*/0.0, &i_indices, &j_indices, /*distance_samples=*/0);

  EXPECT_THAT(i_indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 6, 6));
  EXPECT_THAT(j_indices, ElementsAre(0, 0, 0, 1, 2, 3, 4, 5, 6));

  // Again, but with a band radius of 1, which forces a narrower path.
  i_indices.clear();
  j_indices.clear();
  AlignWithDynamicTimeWarpingOnDemand(
      sequence_1, sequence_2, /*band_radius=*/1, /*penalty_mul=*/1.0,
      /*penalty_add=*/0.0, &i_indices, &j_indices, /*distance_samples=*/0);

  EXPECT_THAT(i_indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 6));
  EXPECT_THAT(j_indices, ElementsAre(0, 0, 1, 2, 3, 4, 5, 6));
}

TEST_F(AlignUtilsTest, OnDemandBandRadDifferentSizes) {
  std::vector<int> i_indices;
  std::vector<int> j_indices;

  Eigen::MatrixXd sequence_1(2, 5);
  Eigen::MatrixXd sequence_2(2, 49);  // Does not divide cleanly into 5.

  sequence_1.setOnes();
  sequence_2.setOnes();

  // Just verify that this doesn't crash.
  AlignWithDynamicTimeWarpingOnDemand(
      sequence_1, sequence_2, /*band_radius=*/0, /*penalty_mul=*/1.0,
      /*penalty_add=*/0.0, &i_indices, &j_indices);
}

}  // namespace magenta
