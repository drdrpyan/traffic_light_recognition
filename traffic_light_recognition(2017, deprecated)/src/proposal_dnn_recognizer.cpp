#include "proposal_dnn_recognizer.hpp"

namespace bgm
{

//ProposalDNNRecognizer::ProposalDNNRecognizer(
//    DNNWrapper* dnn, int num_label, 
//    const std::vector<OutType>& list) {
//  set_dnn(dnn);
//  set_num_label(num_label);
//  set_out_type_list(list);
//}

ProposalDNNRecognizer::ProposalDNNRecognizer(
    DNNWrapper* dnn, NetOutHandler* net_out_handler) 
  : dnn_(dnn), net_out_handler_(net_out_handler) {
  assert(dnn);
  assert(net_out_handler);
}

void ProposalDNNRecognizer::Recognize(
    const std::vector<cv::Mat>& proposal, int top_k,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox,
    std::vector<std::vector<float> >* conf) {
  assert(top_k > 0);

  dnn_->set_input(0, DNNUnit(proposal));
  dnn_->Process();

  (*net_out_handler_)(dnn_->output(), top_k,
                      label, bbox, conf);
  //std::vector<std::vector<int> > parsed_label;
  //std::vector<std::vector<cv::Rect> > parsed_bbox;
  //std::vector<std::vector<float> > parsed_conf;
  //std::vector<int> top_k_idx;

  //if (bbox && bbox_idx_ != NOT_USED)
  //  ParseBBoxOut(dnn_->output(bbox_idx_), &parsed_bbox);
  //if()

  //if (arg_max_idx_ != NOT_USED) {
  //  ParseArgMaxOut(dnn_->output(arg_max_idx_),
  //                 &parsed_label);


  //  if(bbox)


  //  GetDefaultTopKIdx(top_k, &top_k_idx);

  //}
  //else if (sorted_conf_idx_ != NOT_USED)
  //  assert(("Not implemented yet", 0));
  //else if (conf_idx_ != NOT_USED)
  //  assert(("Not implemented yet", 0));
  //else if (conf_bbox_idx_ != NOT_USED)
  //  assert(("Not implemented yet", 0));
  //else
  //  assert(("Illegal net out setting", 0));
}

//void ProposalDNNRecognizer::set_out_type_list(
//    const std::vector<OutType>& list) {
//  int arg_max_idx_ = NOT_USED;
//  int sorted_conf_idx_ = NOT_USED;
//  int conf_idx_ = NOT_USED;
//  int bbox_idx_ = NOT_USED;
//  int conf_bbox_idx_ = NOT_USED;
//
//  for (int i = 0; i < list.size(); ++i) {
//    switch (list[i]) {
//      case ARG_MAX:
//        assert(arg_max_idx_ == NOT_USED);
//        arg_max_idx_ = i;
//        break;
//      case SORTED_CONF:
//        assert(sorted_conf_idx_ == NOT_USED);
//        sorted_conf_idx_ = i;
//        break;
//      case CONF:
//        assert(conf_idx_ == NOT_USED);
//        conf_idx_ = i;
//        break;
//      case BBOX:
//        assert(bbox_idx_ == NOT_USED);
//        bbox_idx_ = i;
//        break;
//      case CONF_BBOX:
//        assert(conf_bbox_idx_ == NOT_USED);
//        conf_bbox_idx_ = i;
//        break;
//      default:
//        assert(("Illegal out type", 0));
//    }
//  }
//}

ProposalDNNRecognizer::NetOutHandler::NetOutHandler(
    float bbox_w_scale, float bbox_h_scale) 
  : bbox_w_scale_(bbox_w_scale), bbox_h_scale_(bbox_h_scale) {

}

void ProposalDNNRecognizer::NetOutHandler::ParseArgMaxOut(
    const DNNUnit& arg_max_out,
    std::vector<std::vector<int> >* arg_max) const {
  assert(arg_max_out.Area() == 1);
  assert(arg_max);

  arg_max->resize(arg_max_out.batch());
  if (arg_max_out.elem_type() == DNNUnit::SINGLE) {
    const float *data_iter = reinterpret_cast<const float*>(arg_max_out.data());
    const int step = arg_max_out.Volume();

    for (int i = 0; i < arg_max_out.batch(); ++i) {
      (*arg_max)[i].resize(step);
      for (int j = 0; j < step; ++j)
        (*arg_max)[i][j] = *data_iter++;
    }
  }
  else
    assert(("Not implemented yet", 0));
}

void ProposalDNNRecognizer::NetOutHandler::ParseConfOut(
    const DNNUnit& conf_out,
    std::vector<std::vector<float> >* conf) const {
  assert(conf_out.Area() == 1);
  assert(conf);

  conf->resize(conf_out.batch());
  if (conf_out.elem_type() == DNNUnit::SINGLE) {
    const float *data_iter = reinterpret_cast<const float*>(conf_out.data());
    const int step = conf_out.Volume();

    for (int i = 0; i < conf_out.batch(); ++i) {
      (*conf)[i].assign(data_iter, data_iter + step);
      data_iter += step;
    }
  }
  else
    assert(("Not implemented yet", 0));
}

void ProposalDNNRecognizer::NetOutHandler::ParseBBoxOut(
    const DNNUnit& bbox_out,
    std::vector<std::vector<cv::Rect2f> >* bbox) const {
  assert(bbox_out.Area() == 1);
  assert(bbox_out.channel() == 4);
  assert(bbox);

  bbox->resize(bbox_out.batch());
  if (bbox_out.elem_type() == DNNUnit::SINGLE) {
    const float *data_iter = reinterpret_cast<const float*>(bbox_out.data());

    for (int i = 0; i < bbox_out.batch(); ++i) {
      float center_x = data_iter[0] * bbox_w_scale_;
      float center_y = data_iter[1] * bbox_h_scale_;
      float width = data_iter[2] * bbox_w_scale_;
      float height = data_iter[3] * bbox_h_scale_;
      
      (*bbox)[i].resize(1);
      (*bbox)[i][0] = cv::Rect2f(center_x - width/2,
                                 center_y - height/2,
                                 width, height);
      data_iter += 4;
    }
  }
  else
    assert(("Not implemented yet", 0));
}

ProposalDNNRecognizer::ReturnSorted::ReturnSorted(
    int arg_max_out_idx, int conf_out_idx, int bbox_out_idx,
    float bbox_w_scale, float bbox_h_scale) 
  : ProposalDNNRecognizer::NetOutHandler::NetOutHandler(bbox_w_scale, bbox_h_scale),
    arg_max_out_idx_(arg_max_out_idx),
    conf_out_idx_(conf_out_idx), bbox_out_idx_(bbox_out_idx) {
  assert(arg_max_out_idx >= 0);
  //assert(conf_out_idx >= 0);
  //assert(bbox_out_idx >= 0);
}

void ProposalDNNRecognizer::ReturnSorted::operator()(
    const std::vector<DNNUnit>& net_out, int top_k,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox,
    std::vector<std::vector<float> >* conf) {
  assert(top_k >= 0);
  assert(label);
  assert(net_out[arg_max_out_idx_].channel() >= top_k);
  if (bbox) {
    assert(bbox_out_idx_ != NOT_USED);
    assert(net_out[arg_max_out_idx_].batch()
           == net_out[bbox_out_idx_].batch());
  }
  if (conf) {
    assert(conf_out_idx_ != NOT_USED);
    assert(net_out[arg_max_out_idx_].batch()
           == net_out[conf_out_idx_].batch());
    assert(net_out[conf_out_idx_].channel() >= top_k);
  }

  
  
  if (net_out[arg_max_out_idx_].channel() == top_k) {
    ParseArgMaxOut(net_out[arg_max_out_idx_], label);
    if (conf) {
      ParseConfOut(net_out[conf_out_idx_], conf);
      assert(label->size(), conf->size());
    }
  }
  else {
    std::vector<std::vector<int> > arg_max;
    ParseArgMaxOut(net_out[arg_max_out_idx_], &arg_max);
    label->resize(arg_max.size());
    for (int i = 0; i < arg_max.size(); ++i) {
      (*label)[i].assign(arg_max[i].begin(),
                         arg_max[i].begin() + top_k);
    }

    if (conf) {
      std::vector<std::vector<float> > conf_all;
      ParseConfOut(net_out[conf_out_idx_], &conf_all);
      conf->resize(conf_all.size());
      for (int i = 0; i < conf_all.size(); ++i) {
        (*conf)[i].assign(conf_all[i].begin(),
                           conf_all[i].begin() + top_k);
      }
    }
  }
  if (bbox)
    ParseBBoxOut(net_out[bbox_out_idx_], bbox);
}

ProposalDNNRecognizer::SortAndReturn::SortAndReturn(
    int conf_out_idx, int bbox_out_idx,
    float bbox_w_scale, float bbox_h_scale)
  : ProposalDNNRecognizer::NetOutHandler::NetOutHandler(bbox_w_scale, bbox_h_scale),
    conf_out_idx_(conf_out_idx), bbox_out_idx_(bbox_out_idx) {
  assert(conf_out_idx_ >= 0);
}

void ProposalDNNRecognizer::SortAndReturn::operator()(
    const std::vector<DNNUnit>& net_out, int top_k,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox,
    std::vector<std::vector<float> >* conf) {
  assert(top_k >= 0);
  assert(label);
  assert(net_out[conf_out_idx_].channel() >= top_k);
  if (bbox) {
    assert(bbox_out_idx_ != NOT_USED);
    assert(net_out[bbox_out_idx_].channel() >= top_k);
  }
  if (conf) {
    assert(conf_out_idx_ != NOT_USED);
    assert(net_out[bbox_out_idx_].batch()
           == net_out[conf_out_idx_].batch());
    assert(net_out[conf_out_idx_].channel() >= top_k);
    conf->resize(net_out[conf_out_idx_].batch());
  }

  label->resize(net_out[conf_out_idx_].batch());

  std::vector<std::vector<float> > conf_all;

  ParseConfOut(net_out[conf_out_idx_], &conf_all);
  //label->resize(conf_all.size());

  for (int i = 0; i < conf_all.size(); ++i) {
    GetTopKIdx<float>(conf_all[i], top_k, &((*label)[i]));
    if (conf)
      GetTopK<float>(conf_all[i], (*label)[i], &((*conf)[i]));
  }

  if (bbox)
    ParseBBoxOut(net_out[bbox_out_idx_], bbox);
}

} // namespace bgm