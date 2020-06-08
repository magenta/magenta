// C++ program for processing alignment tasks created by the 'align_fine.py'
// Python program.
// Tasks are described using alignment.proto.

#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

#include "Eigen/Core"
#include "google/protobuf/repeated_field.h"

#include "./alignment.pb.h"
#include "./align_utils.h"


using std::cerr;
using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::string;

Eigen::MatrixXd readSequence(const magenta::Sequence& sequence) {
  cout << "Creating matrix of size: " << sequence.x() << ", "
       << sequence.y() << endl;
  assert(sequence.content_size() == sequence.x() * sequence.y());
  Eigen::MatrixXd matrix(sequence.x(), sequence.y());
  for (int i = 0; i < sequence.x(); i++) {
    for (int j = 0; j < sequence.y(); j++) {
      matrix(i, j) = sequence.content(i * sequence.y() + j);
    }
  }
  return matrix;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " ALIGNMENT_TASK_FILE" << endl;
    return -1;
  }
  auto alignment_file = string(argv[1]);

  magenta::AlignmentTask alignment_task;
  fstream input(alignment_file, ios::in | ios::binary);
  if (!input) {
    cerr << alignment_file << ": File not found." << endl;
    return -1;
  } else if (!alignment_task.ParseFromIstream(&input)) {
    cerr << "Failed to parse alignment task." << endl;
    return -1;
  }

  cout << "band radius: " << alignment_task.band_radius() << endl;
  cout << "penalty_mul: " << alignment_task.penalty_mul() << endl;
  cout << "penalty: " << alignment_task.penalty() << endl;

  auto sequence_1 = readSequence(alignment_task.sequence_1());
  auto sequence_2 = readSequence(alignment_task.sequence_2());

  std::vector<int> i_indices;
  std::vector<int> j_indices;
  double score = magenta::AlignWithDynamicTimeWarpingOnDemand(
      sequence_1, sequence_2, alignment_task.band_radius(),
      alignment_task.penalty_mul(), alignment_task.penalty(), &i_indices,
      &j_indices);

  magenta::AlignmentResult result;
  result.set_score(score);
  std::copy(i_indices.begin(), i_indices.end(),
            google::protobuf::RepeatedFieldBackInserter(result.mutable_i()));
  std::copy(j_indices.begin(), j_indices.end(),
            google::protobuf::RepeatedFieldBackInserter(result.mutable_j()));


  fstream output(alignment_file + ".result",
                 ios::out | ios::trunc | ios::binary);
  if (!result.SerializeToOstream(&output)) {
    cerr << "Failed to write alignment result." << endl;
    return -1;
  }

  return 0;
}
