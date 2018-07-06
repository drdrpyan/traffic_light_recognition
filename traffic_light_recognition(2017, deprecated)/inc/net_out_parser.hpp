#ifndef TLR_NET_OUT_PARSER_HPP_
#define TLR_NET_OUT_PARSER_HPP_

#include "detector_common.hpp"

#include "caffe/blob.hpp"

#include <set>
#include <vector>

namespace bgm
{

class NetOutParser
{
 public:
  virtual void Parse(const std::vector<caffe::Blob<float>*>& net_out,
                     std::vector<std::vector<Detection> >* detection) = 0;
  //virtual void Parse(const caffe::Blob<float>& net_label_out,
  //                   std::vector<std::vector<Detection> >* detection) = 0;
  //virtual void Parse(const caffe::Blob<float>& net_label_out,
  //                   const caffe::Blob<float>& net_bbox_out,
  //                   std::vector<std::vector<Detection> >* detection) = 0;
  
  void AddIgnoredLabel(int label);

 protected:
  bool IsIgnoredLabel(int label) const;

 private:
  std::set<int> ignored_label_;
};

// inline functions
inline bool NetOutParser::IsIgnoredLabel(int label) const {
  return ignored_label_.find(label) != ignored_label_.cend();
}

inline void NetOutParser::AddIgnoredLabel(int label) {
  ignored_label_.insert(label);
}

} // namespace bgm

#endif // !TLR_NET_OUT_PARSER_HPP_
