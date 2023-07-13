#ifndef CUSTOMMARKER_H
#define CUSTOMMARKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <visualization_msgs/Marker.h>

#include "process/box.h"
#include "tracking/track.h"


class CustomMarker
{
public:
    //constructor
    CustomMarker();

    //deconstructor
    ~CustomMarker();

    geometry_msgs::PolygonStamped get_polygon_msg(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster);
    
    jsk_recognition_msgs::BoundingBox get_bbox_msg(Box box, int clusterId, int clusterclass);
    
    jsk_recognition_msgs::BoundingBox get_bboxq_msg(BoxQ box, int clusterId);

    visualization_msgs::Marker get_text_msg(struct trackingStruct &track, int i);
};

#endif