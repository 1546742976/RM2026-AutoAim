#ifndef AUTO_AIM__CORNER_ORDER_HPP
#define AUTO_AIM__CORNER_ORDER_HPP

#include <algorithm>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace auto_aim
{
// Entries are model-output indices of physical TL, TR, BR, BL, not image-y ranks.
inline std::vector<int> read_corner_order(const YAML::Node & node)
{
  if (!node) return {};
  const auto order = node.as<std::vector<int>>();
  auto sorted = order;
  std::sort(sorted.begin(), sorted.end());
  if (sorted != std::vector<int>{0, 1, 2, 3})
    throw std::invalid_argument("Keypoint order must be a permutation of [0,1,2,3]");
  return order;
}

inline bool order_corners(std::vector<cv::Point2f> & corners, const std::vector<int> & order)
{
  if (corners.size() != 4) return false;
  if (!order.empty()) {
    const auto raw = corners;
    for (std::size_t i = 0; i < 4; ++i) corners[i] = raw.at(order.at(i));
    return true;
  }
  // Unknown export labels support upright compatibility tracking, never certified fire poses.
  std::sort(corners.begin(), corners.end(), [](const auto & a, const auto & b) {
    return a.y < b.y;
  });
  if (corners[0].x > corners[1].x) std::swap(corners[0], corners[1]);
  if (corners[2].x < corners[3].x) std::swap(corners[2], corners[3]);
  return false;
}
}  // namespace auto_aim
#endif
