#ifndef BGM_DNN_WRAPPER_HPP_
#define BGM_DNN_WRAPPER_HPP_

#include "dnn_unit.hpp"

namespace bgm
{


class DNNWrapper
{
 public:
  virtual void Process() = 0;

  void set_input(int idx, const DNNUnit& unit);
  void set_input(const std::vector<DNNUnit>& unit);

  const DNNUnit& input(int idx) const;
  const std::vector<DNNUnit>& input() const;
  const DNNUnit& output(int idx) const;
  const std::vector<DNNUnit>& output() const;
  //virtual void SetInput(int idx, const DNNUnit& unit) = 0;
  //virtual void SetInput(const std::vector<DNNUnit>& unit) = 0;
  //virtual void GetOutput(int idx, DNNUnit* unit) = 0;
  //virtual void GetOutput(std::vector<DNNUnit>* unit) = 0;

 protected:
  //void SetInputSize(int size);
  //void SetOutputSize(int size);

 //private:
  std::vector<DNNUnit> input_;
  std::vector<DNNUnit> output_;
};

// inline functions
inline void DNNWrapper::set_input(int idx, const DNNUnit& unit) {
  input_[idx] = unit;
}

inline void DNNWrapper::set_input(const std::vector<DNNUnit>& unit) {
  input_.assign(unit.begin(), unit.end());
}

inline const DNNUnit& DNNWrapper::input(int idx) const {
  return input_[idx];
}

inline const std::vector<DNNUnit>& DNNWrapper::input() const {
  return input_;
}

inline const DNNUnit& DNNWrapper::output(int idx) const {
  return output_[idx];
}

inline const std::vector<DNNUnit>& DNNWrapper::output() const {
  return output_;
}

//inline void DNNWrapper::SetInputSize(int size) {
//  input_.resize(size);
//}
//
//inline void DNNWrapper::SetOutputSize(int size) {
//  output_.resize(size);
//}

} // namespace bgm
#endif // !BGM_DNN_WRAPPER_HPP_
