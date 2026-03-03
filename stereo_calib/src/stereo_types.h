/*
 * Stereo camera parameter types.
 *
 * Camera model (left camera is the reference frame):
 *   Left:  x_l = K_l * dist_l(X_cam)          where X_cam in left camera frame
 *   Right: x_r = K_r * dist_r(R_rl * X_cam + t_rl)
 *
 * Intrinsics param layout  [9]:  fx, fy, cx, cy, k1, k2, p1, p2, k3
 * Extrinsics param layout  [6]:  r1, r2, r3  (Rodrigues), t1, t2, t3
 * 3D point param layout    [3]:  X, Y, Z  (left camera frame)
 */

#ifndef STEREO_CALIB_SRC_STEREO_TYPES_H
#define STEREO_CALIB_SRC_STEREO_TYPES_H

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>

namespace stereocalib {

// ─── Intrinsics ──────────────────────────────────────────────────────────────

struct Intrinsics {
  double fx = 0, fy = 0;
  double cx = 0, cy = 0;
  double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;

  cv::Mat K() const
  {
    return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  }

  // OpenCV distortion order: k1, k2, p1, p2, k3
  cv::Mat dist() const
  {
    return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
  }

  // [fx, fy, cx, cy, k1, k2, p1, p2, k3]
  std::vector<double> ToVector() const { return {fx, fy, cx, cy, k1, k2, p1, p2, k3}; }

  void FromVector(const std::vector<double>& v)
  {
    fx = v[0]; fy = v[1]; cx = v[2]; cy = v[3];
    k1 = v[4]; k2 = v[5]; p1 = v[6]; p2 = v[7]; k3 = v[8];
  }
};

// ─── Extrinsics (right w.r.t. left) ─────────────────────────────────────────

struct StereoExtrinsics {
  cv::Mat R;  // 3×3 rotation  : X_right = R * X_left + t
  cv::Mat t;  // 3×1 translation

  StereoExtrinsics()
  {
    R = cv::Mat_<double>::eye(3, 3);
    t = cv::Mat_<double>::zeros(3, 1);
  }

  // [r1, r2, r3, t1, t2, t3]  (Rodrigues)
  std::vector<double> ToVector() const
  {
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    return {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
            t.at<double>(0),    t.at<double>(1),    t.at<double>(2)};
  }

  void FromVector(const std::vector<double>& v)
  {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << v[0], v[1], v[2]);
    cv::Rodrigues(rvec, R);
    t = (cv::Mat_<double>(3, 1) << v[3], v[4], v[5]);
  }
};

// ─── Stereo camera ───────────────────────────────────────────────────────────

struct StereoCamera {
  Intrinsics       left;
  Intrinsics       right;
  StereoExtrinsics extrinsics;
};

// ─── Match data ──────────────────────────────────────────────────────────────

// One stereo correspondence: observed pixel in left and right image
struct StereoMatch {
  cv::Point2f pt_left;
  cv::Point2f pt_right;
};

// All correspondences from one stereo image pair
struct StereoPair {
  std::string               name;
  std::vector<StereoMatch>  matches;
};

}  // namespace stereocalib

#endif  // STEREO_CALIB_SRC_STEREO_TYPES_H
