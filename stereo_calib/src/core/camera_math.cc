#include "core/camera_math.h"

#include <opencv2/calib3d.hpp>

namespace stereocalib {
namespace camera_math {

cv::Mat ToRotation(const std::vector<double>& rvec)
{
  const cv::Mat rv = (cv::Mat_<double>(3, 1) << rvec[0], rvec[1], rvec[2]);
  cv::Mat R;
  cv::Rodrigues(rv, R);
  return R;
}

std::vector<double> ToRodrigues(const cv::Mat& R)
{
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  return {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
}

bool TriangulatePoint(const cv::Point2f& pt_left,
                      const cv::Point2f& pt_right,
                      const cv::Mat& K_left,
                      const cv::Mat& K_right,
                      const cv::Mat& dist_left,
                      const cv::Mat& dist_right,
                      const cv::Mat& R_rl,
                      const cv::Mat& t_rl,
                      cv::Point3d& point_3d)
{
  // Undistort points
  std::vector<cv::Point2f> pts_l = {pt_left};
  std::vector<cv::Point2f> pts_r = {pt_right};
  std::vector<cv::Point2f> pts_l_undist, pts_r_undist;
  
  cv::undistortPoints(pts_l, pts_l_undist, K_left, dist_left, cv::Mat(), K_left);
  cv::undistortPoints(pts_r, pts_r_undist, K_right, dist_right, cv::Mat(), K_right);
  
  // Create projection matrices
  cv::Mat P_left = K_left * cv::Mat::eye(3, 4, CV_64F);
  cv::Mat P_right_temp = (cv::Mat_<double>(3, 4) <<
    R_rl.at<double>(0,0), R_rl.at<double>(0,1), R_rl.at<double>(0,2), t_rl.at<double>(0),
    R_rl.at<double>(1,0), R_rl.at<double>(1,1), R_rl.at<double>(1,2), t_rl.at<double>(1),
    R_rl.at<double>(2,0), R_rl.at<double>(2,1), R_rl.at<double>(2,2), t_rl.at<double>(2));
  cv::Mat P_right = K_right * P_right_temp;
  
  // Triangulate
  cv::Mat points4D;
  std::vector<cv::Point2f> pts_l_vec = {pts_l_undist[0]};
  std::vector<cv::Point2f> pts_r_vec = {pts_r_undist[0]};
  cv::triangulatePoints(P_left, P_right, pts_l_vec, pts_r_vec, points4D);
  
  if (points4D.cols != 1) {
    return false;
  }
  
  // Convert to 3D point
  double w = points4D.at<double>(3, 0);
  if (std::abs(w) < 1e-6) {
    return false;
  }
  
  point_3d.x = points4D.at<double>(0, 0) / w;
  point_3d.y = points4D.at<double>(1, 0) / w;
  point_3d.z = points4D.at<double>(2, 0) / w;
  
  return true;
}

}  // namespace camera_math
}  // namespace stereocalib
