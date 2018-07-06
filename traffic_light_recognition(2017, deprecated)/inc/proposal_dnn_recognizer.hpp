#ifndef BGM_PROPOSAL_DNN_RECOGNIZER_HPP_
#define BGM_PROPOSAL_DNN_RECOGNIZER_HPP_

#include "proposal_recognizer.hpp"

#include "dnn_wrapper.hpp"

#include <functional>
#include <memory>

namespace bgm
{

class ProposalDNNRecognizer : public ProposalRecognizer
{
 public:
  enum OutType {ARG_MAX, SORTED_CONF, CONF, BBOX, CONF_BBOX};
  enum {NOT_USED = -1};

  class NetOutHandler
  {
    public:
     NetOutHandler(float bbox_w_scale = 1,
                   float bbox_h_scale = 1);
     virtual void operator()(
         const std::vector<DNNUnit>& net_out, int top_k,
         std::vector<std::vector<int> >* label,
         std::vector<std::vector<cv::Rect2f> >* bbox = 0,
         std::vector<std::vector<float> >* conf = 0) = 0;
    protected:
     void ParseArgMaxOut(
         const DNNUnit& arg_max_out,
         std::vector<std::vector<int> >* arg_max) const;
     void ParseConfOut(
         const DNNUnit& conf_out,
         std::vector<std::vector<float> >* conf) const;
     void ParseBBoxOut(
         const DNNUnit& bbox_out,
         std::vector<std::vector<cv::Rect2f> >* bbox) const;
     template <typename T>
     void GetTopKIdx(const std::vector<T>& conf,
                     int top_k,
                     std::vector<int>* top_k_idx) const;
     template <typename T>
     void GetTopKIdx(
         const std::vector<T>& conf, int top_k,
         std::function<bool(const T&, const T&)> compare,
         std::vector<int>* top_k_idx) const;
     template <typename T>
     void GetTopK(const std::vector<T>& parsed,
                  const std::vector<int>& top_k_idx,
                  std::vector<T>* picked) const;
     //void ParseConfBBoxOut(
     //    const DNNUnit& conf_bbox_out,
     //    std::vector<std::vector<float> >* conf,
     //    std::vector<std::vector<cv::Rect> >* bbox) const;

    private:
     float bbox_w_scale_;
     float bbox_h_scale_;
  };

  class ReturnSorted : public NetOutHandler
  {
    public:
     ReturnSorted(int arg_max_out_idx,
                  int conf_out_idx = NOT_USED,
                  int bbox_out_idx = NOT_USED,
                  float bbox_w_scale = 1,
                  float bbox_h_scale = 1);
     virtual void operator()(
         const std::vector<DNNUnit>& net_out, int top_k,
         std::vector<std::vector<int> >* label,
         std::vector<std::vector<cv::Rect2f> >* bbox = 0,
         std::vector<std::vector<float> >* conf = 0) override;

    private:
     int arg_max_out_idx_;
     int conf_out_idx_;
     int bbox_out_idx_;
  };

  class SortAndReturn : public NetOutHandler
  {
    public:
     SortAndReturn(int conf_out_idx, 
                   int bbox_out_idx = NOT_USED,
                   float bbox_w_scale = 1,
                   float bbox_h_scale = 1);
     virtual void operator()(
         const std::vector<DNNUnit>& net_out, int top_k,
         std::vector<std::vector<int> >* label,
         std::vector<std::vector<cv::Rect2f> >* bbox = 0,
         std::vector<std::vector<float> >* conf = 0) override;

    private:
     int conf_out_idx_;
     int bbox_out_idx_;
  };

  //class ParseAndReturn : public NetOutHandler
  //{
  //  public:
  //   ParseAndReturn(int conf_bbox_out_idx);
  //   virtual void operator()(
  //       const std::vector<DNNUnit>& net_out, int top_k,
  //       std::vector<std::vector<int> >* label,
  //       std::vector<std::vector<cv::Rect> >* bbox = 0,
  //       std::vector<std::vector<float> >* conf = 0) override;

  //  private:
  //   int conf_bbox_out_idx_;
  //};

 public:
  //ProposalDNNRecognizer(DNNWrapper* dnn,
  //                      int num_label,
  //                      const std::vector<OutType>& list);
  ProposalDNNRecognizer(DNNWrapper* dnn,
                        NetOutHandler* net_out_handler);

  virtual void Recognize(
       const std::vector<cv::Mat>& proposal, int top_k,
       std::vector<std::vector<int> >* label,
       std::vector<std::vector<cv::Rect2f> >* bbox = 0,
       std::vector<std::vector<float> >* conf = 0) override;

  //void set_dnn(DNNWrapper* dnn);
  //void set_num_label(int num_label);
  //void set_out_type_list(const std::vector<OutType>& list);

 private:
  //int num_label_;
  //std::vector<OutType> out_type_list_;
  //bool net_out_sorted_;
  std::unique_ptr<DNNWrapper> dnn_;
  std::unique_ptr<NetOutHandler> net_out_handler_;

  //int arg_max_idx_;
  //int sorted_conf_idx_;
  //int conf_idx_;
  //int bbox_idx_;
  //int conf_bbox_idx_;
};

// inline functions
//inline void ProposalDNNRecognizer::set_dnn(DNNWrapper* dnn) {
//  assert(dnn);
//  dnn_.reset(dnn);
//}
//
//inline void ProposalDNNRecognizer::set_num_label(int num_label) {
//  assert(num_label > 0);
//  num_label_ = num_label;
//}

// template functions
template <typename T>
void ProposalDNNRecognizer::NetOutHandler::GetTopKIdx(
    const std::vector<T>& conf, int top_k,
    std::vector<int>* top_k_idx) const {
  assert(conf.size() >= top_k);
  assert(top_k_idx);

  std::vector<std::pair<T, int> > pair_vec(conf.size());
  for (int i = 0; i < conf.size(); ++i)
    pair_vec[i] = std::make_pair(conf[i], i);

  std::sort(pair_vec.begin(), pair_vec.end(),
            [](const std::pair<T, int>& p1,
               const std::pair<T, int>& p2) {
              return p1.first > p2.first; 
            });

  top_k_idx->resize(top_k);
  for (int i = 0; i < top_k; ++i)
    (*top_k_idx)[i] = pair_vec[i].second;
}

template <typename T>
void ProposalDNNRecognizer::NetOutHandler::GetTopKIdx(
    const std::vector<T>& conf, int top_k,
    std::function<bool(const T&, const T&)> compare,
    std::vector<int>* top_k_idx) const {
  assert(conf.size() >= top_k);
  assert(top_k_idx);

  std::vector<std::pair<T, int> > pair_vec(conf.size());
  for (int i = 0; i < conf.size(); ++i)
    pair_vec[i] = std::make_pair(conf[i], i);

  std::sort(pair_vec.begin(), pair_vec.end(),
            [](const std::pair<T, int>& p1,
               const std::pair<T, int>& p2) {
              return compare(p1.first, p2.first); 
            });

  top_k_idx->resize(top_k);
  for (int i = 0; i < top_k; ++i)
    (*top_k_idx)[i] = pair_vec[i].second;
}

template <typename T>
inline void ProposalDNNRecognizer::NetOutHandler::GetTopK(
    const std::vector<T>& parsed,
    const std::vector<int>& top_k_idx,
    std::vector<T>* picked) const {
  assert(parsed.size() >= top_k_idx.size());
  assert(picked);

  picked->resize(top_k_idx.size());
  for (int i = 0; i < top_k_idx.size(); ++i)
    (*picked)[i] = parsed[top_k_idx[i]];
}

}
#endif // !BGM_PROPOSAL_DNN_RECOGNIZER_HPP_
