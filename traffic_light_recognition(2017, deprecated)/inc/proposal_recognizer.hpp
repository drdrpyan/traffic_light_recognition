#ifndef BGM_PROPOSAL_RECOGNIZER_HPP_
#define BGM_PROPOSAL_RECOGNIZER_HPP_

#include <opencv2/core.hpp>

namespace bgm
{

class ProposalRecognizer
{
 public:
  void Recognize(const cv::Mat& proposal,
                 int* label,
                 cv::Rect2f* bbox = 0, float* conf = 0);
  void Recognize(const cv::Mat& proposal, int top_k,
                 std::vector<int>* label,
                 std::vector<cv::Rect2f>* bbox = 0,
                 std::vector<float>* conf = 0);
  void Recognize(
      const std::vector<cv::Mat>& proposal,
      std::vector<int>* label, 
      std::vector<cv::Rect2f>* bbox = 0,
      std::vector<float>* conf = 0);
  virtual void Recognize(
      const std::vector<cv::Mat>& proposal, int top_k,
      std::vector<std::vector<int> >* label,
      std::vector<std::vector<cv::Rect2f> >* bbox = 0,
      std::vector<std::vector<float> >* conf = 0) = 0;

  void Recognize(const cv::Mat& img,
                 const std::vector<cv::Rect>& proposal,
                 std::vector<int>* label,
                 std::vector<cv::Rect2f>* bbox = 0,
                 std::vector<float>* conf = 0);
  virtual void Recognize(
      const cv::Mat& img, int top_k,
      const std::vector<cv::Rect>& proposal,
      std::vector<std::vector<int> >* label,
      std::vector<std::vector<cv::Rect2f> >* bbox = 0,
      std::vector<std::vector<float> >* conf = 0);

 private:
  cv::Mat ExtractProposal(const cv::Mat& img, 
                          const cv::Rect& proposal) const;

}; // class ProposalRecognizer

// inline functions

} // namespace bgm
#endif // !BGM_PROPOSAL_RECOGNIZER_HPP_
