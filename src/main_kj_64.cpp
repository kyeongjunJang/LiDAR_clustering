#include "utility64.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "process/processPointClouds.h"
#include "lib/processPointClouds.cpp"
#include "marker/customMarker.h"
#include "tracking/track.h"
#include <pcl/common/transforms.h>
#include <chrono>
#include <thread>
#include <math.h>
#include <typeinfo>

#define PI 3.14159265359

using namespace cv;

class Process
{

private:
    double height_ave = 0;
    double length_ave = 0;
    double width_ave = 0;
    double intens_ave = 0;
    double cluster_height = 0;
    double cluster_width = 0;
    double cluster_length = 0;
    double cluster_term1 = 0;
    double cluster_term2 = 0;
    double car_x = 0;
    double car_y = 0;
    double car_theta = 0;
    int cluster_num = 0;
    int upper_num = 0;
    int bottom_num = 0;

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subCarPose;

    ros::Publisher pubFullCloud;
    ros::Publisher pubTransCloud;
    ros::Publisher pubGroundCloud;
    ros::Publisher pubGroundRemovedCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pub_cluster_poly;
    ros::Publisher pub_cluster_box;
    ros::Publisher pub_track_box;
    ros::Publisher pub_track_text;

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn;
    pcl::PointCloud<pcl::PointXYZI>::Ptr fullCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr transCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr groundCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr groundRemovedCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr segmentedCloud;

    ProcessPointClouds<pcl::PointXYZI> pointProcessor;

    jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array;
    jsk_recognition_msgs::BoundingBoxArray track_bbox_array;
    visualization_msgs::MarkerArray track_text_array;
    geometry_msgs::Pose2D ego_car_pose;

    pcl::PointXYZI nanPoint; // fill in fullCloud at each iteration

    Mat groundMat; // ground matrix for ground cloud marking

    std_msgs::Header cloudHeader;

    CustomMarker customMarker;
    Track tracker;
    Track mylane_tracker;

    ros::Publisher pub_mylane_cluster_box;
    ros::Publisher pub_mylane_track_box;
    jsk_recognition_msgs::BoundingBoxArray mylane_cluster_bbox_array;
    jsk_recognition_msgs::BoundingBoxArray mylane_track_bbox_array;
    double lidarSec;

public:
    Process() : nh("~")
    {
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &Process::cloudHandler, this);
        subCarPose = nh.subscribe<geometry_msgs::Pose2D>("/enu_pose", 1, &Process::carPoseCallback, this);
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projection", 1);
        pubTransCloud = nh.advertise<sensor_msgs::PointCloud2>("/transformed_cloud", 1);
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubGroundRemovedCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_removed_cloud", 1);
        pub_cluster_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/cluster_box", 1);
        pub_track_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_box", 1);
        pub_track_text = nh.advertise<visualization_msgs::MarkerArray>("/lidar/track_text", 1);

        pub_mylane_cluster_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/mylane_cluster_box", 1);
        pub_mylane_track_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/mylane_track_box", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();

        resetParameters();
    }

    ~Process() {}

    void carPoseCallback(const geometry_msgs::Pose2D carPoseMsg)
    {
        car_x = carPoseMsg.x;
        car_y = carPoseMsg.y;
        car_theta = carPoseMsg.theta;
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<pcl::PointXYZI>());
        fullCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        transCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        groundCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        groundRemovedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        segmentedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        fullCloud->points.resize(N_SCAN * Horizon_SCAN);
        transCloud->points.resize(N_SCAN * Horizon_SCAN);
    }

    void cloudHandler(const sensor_msgs::PointCloud2 laserCloudMsg)
    { // laserCloudMsg is pcl2 data
        clock_t astart = clock();
        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // 2. Range image projection
        projectPointCloud();
        // transformPointCloud();
        //fogFiltering();
        // 3. Mark ground points
        groundRemoval();
        // 4. Point cloud segmentation
        cloudSegmentation();
        // 5. Publish all clouds
        cloudPublish();
        // 6. Tracking box
        tracking();
        // mylaneTracking();
        // 7. publish clustering and tracking
        publishResult();
        // 8. Reset parameters for next iteration
        resetParameters();
        
        //printf("%f s-all\n", (float)(clock() - astart) / CLOCKS_PER_SEC);
    }

    void copyPointCloud(const sensor_msgs::PointCloud2 laserCloudMsg)
    {
        cloudHeader = laserCloudMsg.header;
        lidarSec = laserCloudMsg.header.stamp.toSec();
        cloudHeader.stamp = ros::Time::now();
        pcl::fromROSMsg(laserCloudMsg, *laserCloudIn); // pcl2 data -> cloud dst
    }

    void projectPointCloud()
    {
        // fullCloud = laserCloudIn;
        //  range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize;
        pcl::PointXYZI thisPoint;

        cloudSize = laserCloudIn->points.size();
        // 모든 포인트들에 대해서 row, col으로 인덱싱
        for (size_t i = 0; i < cloudSize; ++i)
        {
            
            thisPoint.x = -laserCloudIn->points[i].y;
            thisPoint.y = laserCloudIn->points[i].x;
            thisPoint.z = laserCloudIn->points[i].z;
            /*
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            */
            if (sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y) < 1.5) continue;

            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;

            // Hesai Pandar64
            if (verticalAngle < -19){
                rowIdn = (verticalAngle + ang_bottom) / 6;
            }
            else if (verticalAngle < -14){
                rowIdn = (verticalAngle + ang_bottom - 6) / 5 + 2;
            }
            else if (verticalAngle < -6){
                rowIdn = (verticalAngle + ang_bottom - 11) / 1 + 3;
            }
            else if (verticalAngle < 2){
                rowIdn = (verticalAngle + ang_bottom - 19) / 0.167 + 11;
            }
            else if (verticalAngle < 3){
                rowIdn = (verticalAngle + ang_bottom - 27) / 1 + 59;
            }
            else if (verticalAngle < 5){
                rowIdn = (verticalAngle + ang_bottom - 28) / 2 + 60;
            }
            else if (verticalAngle < 11){
                rowIdn = (verticalAngle + ang_bottom - 30) / 3 + 61;
            }
            else{
                rowIdn = (verticalAngle + ang_bottom - 36) / 4 + 63;
            }
            
            // Ouster OS-1 64Ch
            //rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            thisPoint.intensity = laserCloudIn->points[i].intensity;

            index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;

            // // find the row and column index in the iamge for this point
            // //각 포인트들과 가장 아래 방향 채널의 포인트가 이루는 각을 y방향 resolution으로 나누어 넘버링
            // // 좌측이 y축, 앞이 x축, 위가 z축
            // // LiDAR 정면을 (X축) GRID : Horizon_RESOLUTION / 2로 두겠다.
            // // GRID 안에 값 대입 : 거리 값
        }
    }

    void fogFiltering()
    {
        size_t middleInd, tempInd;
        float midDist, dist;

        for (size_t j = 1; j < N_SCAN; ++j)
        {
            for (size_t i = 0; i < Horizon_SCAN; ++i)
            {
                middleInd = i + j * Horizon_SCAN;
                midDist = sqrt(fullCloud->points[middleInd].x * fullCloud->points[middleInd].x \
                    + fullCloud->points[middleInd].y * fullCloud->points[middleInd].y \
                    + fullCloud->points[middleInd].z * fullCloud->points[middleInd].z);

                if (midDist < 10)
                {
                    for (size_t k = 0; k < 5 ; k++)
                    {
                        tempInd = middleInd - 2 + k;
                        dist = sqrt((fullCloud->points[middleInd].x - fullCloud->points[tempInd].x) * (fullCloud->points[middleInd].x - fullCloud->points[tempInd].x) \
                            + (fullCloud->points[middleInd].y - fullCloud->points[tempInd].y) * (fullCloud->points[middleInd].y - fullCloud->points[tempInd].y) \
                            + (fullCloud->points[middleInd].z - fullCloud->points[tempInd].z) * (fullCloud->points[middleInd].z - fullCloud->points[tempInd].z));
                        if (dist > 2)
                        {
                            fullCloud->points[middleInd] = nanPoint;
                            break;
                        }
                    }
                }
            }
        }   
    }

    void transformPointCloud()
    {
        
        Eigen::Matrix4f trans;
        trans<< cos((car_theta -1) * PI / 180), -sin((car_theta -1) * PI / 180),  0, car_x + 0.5,
                sin((car_theta -1) * PI / 180), cos((car_theta -1) * PI / 180),  0, car_y - 0.7,
                0,   0,  1,     2,
                0,   0,  0,     1;
        pcl::transformPointCloud(*fullCloud, *fullCloud, trans);
    }

    void groundRemoval()
    {
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j)
        {
            for (size_t i = 0; i < groundScanInd; ++i)
            {
                lowerInd = j + (i) * Horizon_SCAN;
                upperInd = j + (i + 1) * Horizon_SCAN;
                if (fullCloud->points[lowerInd].intensity == -1||
                    fullCloud->points[upperInd].intensity == -1)
                {
                    groundMat.at<int8_t>(i, j) = -1;
                    continue;
                }
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
                angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
                if (abs(angle - sensorMountAngle) <= groundThreshold)// && diffX <= 20)
                {
                    groundMat.at<int8_t>(i, j) = 1;
                    groundMat.at<int8_t>(i + 1, j) = 1;
                }

                if (groundMat.at<int8_t>(i, j) == 1)
                {
                    groundCloud->push_back(fullCloud->points[lowerInd]);
                }
                else if (fullCloud->points[lowerInd].x < MAX_X && fullCloud->points[lowerInd].x > MIN_X &&
                          fullCloud->points[lowerInd].y < MAX_Y && fullCloud->points[lowerInd].y > MIN_Y &&
                          fullCloud->points[lowerInd].z < MAX_Z && fullCloud->points[lowerInd].z > MIN_Z &&
                          groundMat.at<int8_t>(i, j) == 0)
                {
                    groundRemovedCloud->push_back(fullCloud->points[lowerInd]);
                }
            }
        }
    }

    void cloudSegmentation()
    {
        // Clustering
        vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = pointProcessor.Clustering(groundRemovedCloud, clusterTolerance, minSize, maxSize);
        // Processing af
        int clusterId = 0;
        for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters)
        {
            pcl::PointXYZI minPoint, maxPoint;
            pcl::getMinMax3D(*cluster, minPoint, maxPoint);

            // new
            BoxQ boxq = pointProcessor.MinimumOrientedBoundingBox(cluster, cloudHeader.stamp.toSec());
            jsk_recognition_msgs::BoundingBox bbox = customMarker.get_bboxq_msg(boxq, clusterId);

            if (boxq.cube_length < 18 && boxq.cube_width < 4)
            {
                cluster_bbox_array.boxes.push_back(bbox);
                if (boxq.bboxTransform[1] < LANE_MAX_Y && boxq.bboxTransform[1] > LANE_MIN_Y){
                    
                    Box box = pointProcessor.BoundingBox(cluster);
                    bbox.pose.position.x = (box.x_max + box.x_min) / 2;
                    bbox.pose.position.y = (box.y_max + box.y_min) / 2;
                    bbox.pose.position.z = (box.z_max + box.z_min) / 2;
                    bbox.dimensions.x = (box.x_max - box.x_min);
                    bbox.dimensions.y = (box.y_max - box.y_min);
                    bbox.dimensions.z = (box.z_max - box.z_min);
                    bbox.pose.orientation.x = 0;
                    bbox.pose.orientation.y = 0;
                    bbox.pose.orientation.z = 0;

                    mylane_cluster_bbox_array.boxes.push_back(bbox);
                }
            }
        }
    }

    void cloudPublish()
    {
        sensor_msgs::PointCloud2 laserCloudTemp;
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = frameID;
            pubFullCloud.publish(laserCloudTemp);
        }
        // transformed cloud
        if (pubTransCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*transCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = frameID;
            pubTransCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = frameID;
            pubGroundCloud.publish(laserCloudTemp);
        }
        // without ground cloud
        if (pubGroundRemovedCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*groundRemovedCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = frameID;
            pubGroundRemovedCloud.publish(laserCloudTemp);
        }
    }

    void tracking(){
        jsk_recognition_msgs::BoundingBoxArray filtered_bbox_array = tracker.filtering(cluster_bbox_array);
        tracker.predictNewLocationOfTracks();
        tracker.assignDetectionsTracks(filtered_bbox_array);
        tracker.assignedTracksUpdate(filtered_bbox_array, lidarSec);
        tracker.unassignedTracksUpdate();
        tracker.deleteLostTracks();
        tracker.createNewTracks(filtered_bbox_array);
        pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_text = tracker.displayTrack();
        track_bbox_array = bbox_text.first;
        track_text_array = bbox_text.second;
    }

    void mylaneTracking(){
        jsk_recognition_msgs::BoundingBoxArray filtered_bbox_array = mylane_tracker.filtering(mylane_cluster_bbox_array);
        mylane_tracker.predictNewLocationOfTracks();
        mylane_tracker.assignDetectionsTracksMylane(filtered_bbox_array);
        mylane_tracker.assignedTracksUpdate(filtered_bbox_array, lidarSec);
        mylane_tracker.unassignedTracksUpdate();
        mylane_tracker.deleteLostTracks();
        mylane_tracker.createNewTracks(filtered_bbox_array);
        pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_text = mylane_tracker.displayTrack();
        mylane_track_bbox_array = bbox_text.first;
    }

    void publishResult()
    {
        // cluster_box array
        cluster_bbox_array.header.stamp = cloudHeader.stamp;
        cluster_bbox_array.header.frame_id = frameID;
        pub_cluster_box.publish(cluster_bbox_array);      

        // track_box array
        track_bbox_array.header.stamp = cloudHeader.stamp;
        track_bbox_array.header.frame_id = frameID;
        pub_track_box.publish(track_bbox_array);

        // mylane cluster_box array
        mylane_cluster_bbox_array.header.stamp = cloudHeader.stamp;
        mylane_cluster_bbox_array.header.frame_id = frameID;
        pub_mylane_cluster_box.publish(mylane_cluster_bbox_array);

        // mylane track_box array
        mylane_track_bbox_array.header.stamp = cloudHeader.stamp;
        mylane_track_bbox_array.header.frame_id = frameID;
        pub_mylane_track_box.publish(mylane_track_bbox_array);
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        groundCloud->clear();
        groundRemovedCloud->clear();
        segmentedCloud->clear();
        cluster_bbox_array.boxes.clear();
        track_bbox_array.boxes.clear();
        track_text_array.markers.clear();

        mylane_cluster_bbox_array.boxes.clear();
        mylane_track_bbox_array.boxes.clear();

        groundMat = Mat(N_SCAN, Horizon_SCAN, CV_8S, Scalar::all(0));

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(transCloud->points.begin(), transCloud->points.end(), nanPoint);
    }
};
int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar");

    while(ros::ok())
    {
        Process P;
        
        ros::spin();
        
    }
    
    return 0;
}
