#ifndef FUSION_SPACE_SOLVER_H__
#define FUSION_SPACE_SOLVER_H__

#include <vector>

#include "CostFunctor.h"
#include "ProposalGenerator.h"


class FusionSpaceSolver
{
 public:
  
  FusionSpaceSolver(const int NUM_NODES, const std::vector<std::vector<int> > &node_neighbors, CostFunctor &cost_functor, ProposalGenerator &proposal_generator, const int NUM_ITERATIONS = 1000, const bool CONSIDER_LABEL_COST = false);
  
  //  void setNeighbors();
  //void setNeighbors(const int width, const int height, const int neighbor_system = 8);
  
  std::vector<int> solve(const int NUM_ITERATIONS, const std::vector<int> &initial_solution);
  
 private:
  const int NUM_NODES_;
  const int NUM_ITERATIONS_;
  const bool CONSIDER_LABEL_COST_;
  
  const std::vector<std::vector<int> > node_neighbors_;
  CostFunctor &cost_functor_;
  ProposalGenerator &proposal_generator_;
  
  std::vector<int> fuse(const std::vector<std::vector<int> > &proposal_labels, std::vector<double> &energy_info);
};

#endif
