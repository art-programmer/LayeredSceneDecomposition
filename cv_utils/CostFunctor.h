#ifndef COST_FUNCTOR_H__
#define COST_FUNCTOR_H__

class CostFunctor
{
 public:
  virtual double operator()(const int node_index, const int label) const = 0;
  virtual double operator()(const int node_index_1, const int node_index_2, const int label_1, const int label_2) const = 0;
  virtual void setCurrentSolution(const std::vector<int> &current_solution) {};
  virtual double getLabelCost() const { return 0; };
  virtual double getLabelIndicatorConflictCost() const { return 0; };
};

#endif
