#include "utility128.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "process/processPointClouds.h"
#include "lib/processPointClouds.cpp"
#include "marker/customMarker.h"
#include "tracking/track.h"

#include <chrono>
#include <thread>

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
    int cluster_num = 0;
    int upper_num = 0;
    int bottom_num = 0;

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;

    ros::Publisher pubFullCloud;
    ros::Publisher pubCropCloud;
    ros::Publisher pubGroundCloud;
    ros::Publisher pubGroundRemovedCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pub_cluster_poly;
    ros::Publisher pub_cluster_box;
    ros::Publisher pub_track_box;
    ros::Publisher pub_track_text;

    

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn;
    pcl::PointCloud<pcl::PointXYZI>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix

    pcl::PointCloud<pcl::PointXYZI>::Ptr cropCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr groundCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr groundRemovedCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr segmentedCloud;

    ProcessPointClouds<pcl::PointXYZI> pointProcessor;

    jsk_recognition_msgs::PolygonArray polygon_array;
    jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array;
    jsk_recognition_msgs::BoundingBoxArray track_bbox_array;
    visualization_msgs::MarkerArray track_text_array;

    pcl::PointXYZI nanPoint; // fill in fullCloud at each iteration

    Mat rangeMat;  // range matrix for range image
    Mat labelMat;  // label matrix for segmentaiton marking
    Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    std_msgs::Header cloudHeader;

    CustomMarker customMarker;
    Track tracker;
    Track tracker_fusion;
    Track tracker_pillar;

public:
    Process() : nh("~")
    {
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &Process::cloudHandler, this);
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projection", 1);
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubGroundRemovedCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_removed_cloud", 1);
        pub_cluster_poly = nh.advertise<jsk_recognition_msgs::PolygonArray>("/lidar/cluster_poly", 1);
        pub_cluster_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/lidar/cluster_box", 1);
        pub_track_box = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/lidar/track_box", 1);
        pub_track_text = nh.advertise<visualization_msgs::MarkerArray>("/lidar/track_text", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();

        resetParameters();
    }

    ~Process() {}

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<pcl::PointXYZI>());
        fullCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        cropCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        groundCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        groundRemovedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        segmentedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        fullCloud->points.resize(N_SCAN * Horizon_SCAN);
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        cropCloud->clear();
        groundCloud->clear();
        groundRemovedCloud->clear();
        segmentedCloud->clear();
        polygon_array.polygons.clear();
        cluster_bbox_array.boxes.clear();
        // track_bbox_array.boxes.clear();
        // track_text_array.markers.clear();

        rangeMat = Mat(N_SCAN, Horizon_SCAN, CV_32F, Scalar::all(FLT_MAX));
        groundMat = Mat(N_SCAN, Horizon_SCAN, CV_8S, Scalar::all(0));
        labelMat = Mat(N_SCAN, Horizon_SCAN, CV_32S, Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    }

    void copyPointCloud(const sensor_msgs::PointCloud2 laserCloudMsg)
    {
        cloudHeader = laserCloudMsg.header;
        cloudHeader.stamp = ros::Time::now();
        pcl::fromROSMsg(laserCloudMsg, *laserCloudIn); // pcl2 data -> cloud dst
    }

    void cloudHandler(const sensor_msgs::PointCloud2 laserCloudMsg)
    { // laserCloudMsg is pcl2 data
        clock_t astart = clock();
        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // 2. Range image projection
        projectPointCloud();
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // 3. Mark ground points
        groundRemoval();
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        printf("%f s-removal\n", (float)(clock() - astart) / CLOCKS_PER_SEC);
        // 4. Point cloud segmentation
        //cloudSegmentation();
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // 5. Publish all clouds
        cloudPublish();
        // 6. Tracking box or polygon
        traking();
        
        // 7. publish clustering and tracking
        publishResult();
        // 8. Reset parameters for next iteration
        resetParameters();
        
        printf("%f s-all\n", (float)(clock() - astart) / CLOCKS_PER_SEC);
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

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;

            rangeMat.at<float>(rowIdn, columnIdn) = range;

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
                lowerInd = j + (i)*Horizon_SCAN;
                upperInd = j + (i + 1) * Horizon_SCAN;
                if (fullCloud->points[lowerInd].intensity == -1||
                    fullCloud->points[upperInd].intensity == -1)
                {
                    groundMat.at<int8_t>(i, j) = -1;
                    continue;
                }
                // else if (fullCloud->points[upperInd].intensity == -1)
                // {
                //     groundMat.at<int8_t>(i, j) = 1;
                //     continue;
                // }
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
                angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
                if (abs(angle - sensorMountAngle) <= groundThreshold)// && sqrt(diffX * diffX + diffY * diffY) <= 20) //||\
                 //(isnan(angle)&& fullCloud->points[lowerInd].z < -2.2))
                {
                    groundMat.at<int8_t>(i, j) = 1;
                    groundMat.at<int8_t>(i + 1, j) = 1;
                }
                // else if (abs(angle - sensorMountAngle) <= groundThreshold && diffZ < 0.3)
                //     {
                //         groundMat.at<int8_t>(i, j) = 1;
                //         groundMat.at<int8_t>(i + 1, j) = 1;
                //     }

            }
        }
        for (size_t j = 0; j < Horizon_SCAN; ++j)
        {
            for (size_t i = 0; i < N_SCAN; ++i)
            {
                lowerInd = j + (i)*Horizon_SCAN;
                if (groundMat.at<int8_t>(i, j) == 1 )
                {
                    groundCloud->push_back(fullCloud->points[lowerInd]);
                }
                else if (fullCloud->points[lowerInd].x < MAX_X && fullCloud->points[lowerInd].x > MIN_X &&
                         fullCloud->points[lowerInd].y < MAX_Y && fullCloud->points[lowerInd].y > MIN_Y &&
                         fullCloud->points[lowerInd].z < 3  && groundMat.at<int8_t>(i, j) == 0) //
                {
                    groundRemovedCloud->push_back(fullCloud->points[lowerInd]);
                }
            }
        }
    }
    void cloudSegmentation()
    {
        // Clustering
        vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = pointProcessor.Clustering(groundRemovedCloud, 0.8, 3, 1000);
        // Processing af
        int clusterId = 0;
        for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters)
        {
            // Polygon
            geometry_msgs::PolygonStamped polygon = customMarker.get_polygon_msg(cluster);

            // Exclude high area
            pcl::PointXYZI minPoint, maxPoint;
            pcl::getMinMax3D(*cluster, minPoint, maxPoint);
            if (maxPoint.z < 0.5 && \
            (maxPoint.x - minPoint.x > 0.05 || maxPoint.y - minPoint.y > 0.2) && \
            maxPoint.x - minPoint.x < 5 && maxPoint.y - minPoint.y < 9 && \
            (maxPoint.x - minPoint.x) * (maxPoint.y - minPoint.y) < 20)// && \
            //sqrt(pow(maxPoint.x, 2) + pow(maxPoint.y, 2)) < 35)
            {
                BoxQ box = pointProcessor.MinimumOrientedBoundingBox(cluster, cloudHeader.stamp.toSec());
                jsk_recognition_msgs::BoundingBox bbox = customMarker.get_bboxq_msg(box, clusterId);
                polygon_array.polygons.push_back(polygon);
                cluster_bbox_array.boxes.push_back(bbox);
                ++clusterId;
                if (box.cube_length > 0.4 && box.cube_width > 0.4 && box.cube_length < 9 && box.cube_width < 9 && box.cube_length * box.cube_width < 25)
                {
                    if ((box.cube_length > 5 && box.cube_width < 2) || (box.cube_width > 5 && box.cube_length < 2) || (box.cube_width > 2.5 && box.cube_length > 2.5))
                        continue;
                    else
                    {
                        height_ave = 0;
                        length_ave = 0;
                        width_ave = 0;
                        cluster_length = 0;
                        cluster_width = 0;
                        cluster_height = 0;
                        upper_num = 0;
                        bottom_num = 0;
                        height_ave = (maxPoint.z + minPoint.z) / 2;
                        length_ave = (maxPoint.x + minPoint.x) / 2;
                        width_ave = (maxPoint.y + minPoint.y) / 2;
                        // for (size_t i = 0; i < cluster->points.size(); ++i)
                        // {
                        //     if (cluster->points[i].z > height_ave && cluster->points[i].z < maxPoint.z - (maxPoint.z - minPoint.z) / 8)
                        //     {
                        //         upper_num += 1;
                        //     }
                        //     else if (cluster->points[i].z > minPoint.z + (maxPoint.z - minPoint.z) / 8 && cluster->points[i].z < height_ave)
                        //     {
                        //         bottom_num += 1;
                        //     }
                        // }
                        for (size_t i = 0; i < cluster->points.size(); ++i)
                        {
                            cluster_length += cluster->points[i].x;
                            cluster_width += cluster->points[i].y;
                            cluster_height += cluster->points[i].z;
                            cluster_num = i;
                        }
                        cluster_length = cluster_length / cluster_num;
                        cluster_width = cluster_width / cluster_num;
                        cluster_height = cluster_height / cluster_num;
                        cluster_term1 = sqrt(pow(height_ave, 2) + pow(width_ave, 2) + pow(length_ave, 2));
                        cluster_term2 = sqrt(pow(cluster_height, 2) + pow(cluster_width, 2) + pow(cluster_length, 2));
                        // cout << "upper_num =" << upper_num << endl;
                        // cout << "bottom_num =" << bottom_num << endl;
                        // if (upper_num < bottom_num) //&& abs(cluster_term1-cluster_term2)>0.2) // && intens_ave <80)
                        // {
                        // jsk_recognition_msgs::BoundingBox bbox = customMarker.get_bboxq_msg(box, clusterId);
                        // polygon_array.polygons.push_back(polygon);
                        // cluster_bbox_array.boxes.push_back(bbox);
                        // ++clusterId;
                        // }
                    }
                }
                // polygon_array.polygons.push_back(polygon);
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

    void traking(){
        jsk_recognition_msgs::BoundingBoxArray filtered_bbox_array = tracker.filtering(cluster_bbox_array);
        tracker.predictNewLocationOfTracks();
        tracker.assignDetectionsTracks(filtered_bbox_array);
        tracker.assignedTracksUpdate(filtered_bbox_array);
        tracker.unassignedTracksUpdate();
        tracker.deleteLostTracks();
        tracker.createNewTracks(filtered_bbox_array);
        pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_text = tracker.displayTrack();
        track_bbox_array = bbox_text.first;
        //track_text_array = bbox_text.second;

        // jsk_recognition_msgs::BoundingBoxArray filtered_pillarbox_array = tracker_pillar.filtering(pillarhandler_msg);
        // tracker_pillar.predictNewLocationOfTracks();
        // tracker_pillar.assignDetectionsTracks(filtered_pillarbox_array);
        // tracker_pillar.assignedTracksUpdate(filtered_pillarbox_array);
        // tracker_pillar.unassignedTracksUpdate();
        // tracker_pillar.deleteLostTracks();
        // tracker_pillar.createNewTracks(filtered_pillarbox_array);
        // pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> pillarbox_text = tracker_pillar.displayTrack();
        // track_pillarbox_array = pillarbox_text.first;
        // track_pillartext_array = pillarbox_text.second;

    }

    void publishResult()
    {
        // polygon
        polygon_array.header.stamp = cloudHeader.stamp;
        polygon_array.header.frame_id = frameID;
        pub_cluster_poly.publish(polygon_array);

        // bbox
        cluster_bbox_array.header.stamp = cloudHeader.stamp;
        cluster_bbox_array.header.frame_id = frameID;
        pub_cluster_box.publish(cluster_bbox_array);

        // tracking
        track_bbox_array.header.stamp = ros::Time::now();
        track_bbox_array.header.frame_id = frameID;
        pub_track_box.publish(track_bbox_array);
        // tacking id

        // Publish clouds

        // // clster_box array


        // // track_box array


        // // track_text array
        //     pub_track_text.publish(track_text_array);

        // // far object marker array
        // pub_far_object_marker.publish(far_object_marker_array);
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
