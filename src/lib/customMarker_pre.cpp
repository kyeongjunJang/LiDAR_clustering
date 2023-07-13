#include "marker/customMarker.h"

using namespace std;
using namespace cv;


CustomMarker::CustomMarker(){}

CustomMarker::~CustomMarker(){}

geometry_msgs::PolygonStamped CustomMarker::get_polygon_msg(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster)
{
  vector<Point2f> points;
  geometry_msgs::PolygonStamped polygonStamped;
  polygonStamped.header.frame_id = "os_sensor";

  float min_z = 100.0;
  float max_z = -100.0;

  for (unsigned int i = 0; i < cluster->points.size(); i++)
  {
    Point2f pt;
    pt.x = cluster->points[i].x;
    pt.y = cluster->points[i].y;
    points.push_back(pt);

    if(min_z > cluster->points[i].z)
            min_z = cluster->points[i].z;
    if(max_z < cluster->points[i].z)
        max_z = cluster->points[i].z;
  }

  vector<Point2f> hull;
  convexHull(points, hull);
  for (size_t i = 0; i < hull.size() + 1; i++)
  {
    geometry_msgs::Point32 point;
    point.x = hull[i % hull.size()].x ;
    point.y = hull[i % hull.size()].y ;
    point.z = min_z;
    polygonStamped.polygon.points.push_back(point);
  }

  for (size_t i = 0; i < hull.size() + 1; i++)
  {
    geometry_msgs::Point32 point;
    point.x = hull[i % hull.size()].x;
    point.y = hull[i % hull.size()].y;
    point.z = max_z;
    polygonStamped.polygon.points.push_back(point);
  }

  return polygonStamped;
}

jsk_recognition_msgs::BoundingBox CustomMarker::get_bbox_msg(Box box, int clusterId)
{
  jsk_recognition_msgs::BoundingBox bbox;
  bbox.header.frame_id = "os_sensor";

  bbox.pose.position.x = (box.x_min + box.x_max) / 2.0;
  bbox.pose.position.y = (box.y_min + box.y_max) / 2.0;
  bbox.pose.position.z = (box.z_min + box.z_max) / 2.0;
  bbox.pose.orientation.x = 0.0;
  bbox.pose.orientation.y = 0.0;
  bbox.pose.orientation.z = 0.0;
  bbox.pose.orientation.w = 1.0;
  bbox.dimensions.x = box.x_max - box.x_min;
  bbox.dimensions.y = box.y_max - box.y_min;
  bbox.dimensions.z = box.z_max - box.z_min;
  bbox.value = clusterId;

  return bbox;
}

jsk_recognition_msgs::BoundingBox CustomMarker::get_bboxq_msg(BoxQ box, int clusterId)
{
  jsk_recognition_msgs::BoundingBox bbox;
  bbox.header.frame_id = "os_sensor";

  bbox.pose.position.x = box.bboxTransform[0];
  bbox.pose.position.y = box.bboxTransform[1];
  bbox.pose.position.z = box.bboxTransform[2];
  bbox.pose.orientation.x = box.bboxQuaternion.vec()[0];
  bbox.pose.orientation.y = box.bboxQuaternion.vec()[1];
  bbox.pose.orientation.z = box.bboxQuaternion.vec()[2];
  bbox.pose.orientation.w = box.bboxQuaternion.w();
  bbox.dimensions.x = box.cube_length;
  bbox.dimensions.y = box.cube_width;
  bbox.dimensions.z = box.cube_height;
  bbox.value = clusterId;

  return bbox;
}

visualization_msgs::Marker CustomMarker::get_text_msg(struct trackingStruct &track, int i)
{
  visualization_msgs::Marker text;
  text.header.frame_id = "os_sensor";
  text.ns = "text";
  text.id = i;
  text.action = visualization_msgs::Marker::ADD;
  text.type = visualization_msgs::Marker::CUBE;
  text.lifetime = ros::Duration(0.1);
  text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text.color.r = 1.0;
  text.color.g = 1.0;
  text.color.b = 1.0;
  text.color.a = 1.0;
  text.scale.z = 1.0;

  text.pose.position.x = track.cur_bbox.pose.position.x;
  text.pose.position.y = track.cur_bbox.pose.position.y;
  text.pose.position.z = 0.5;
  text.pose.orientation.w = 1.0;

  char buf[100];
  sprintf(buf, "ID : %d", track.id*10+1);

  text.text = buf;

  return text;
}
