// PCL lib Functions for processing point clouds 

#include "process/processPointClouds.h"


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    // auto startTime = std::chrono::steady_clock::now();

    // Voxel grid point reduction and region based filtering
    pcl::VoxelGrid<PointT> vg;
    typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(filterRes, filterRes, filterRes);
    vg.filter(*cloudFiltered);

    typename pcl::PointCloud<PointT>::Ptr cloudRegion (new pcl::PointCloud<PointT>);

    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(cloudFiltered);
    region.filter(*cloudRegion);

    std::vector<int> indices;

    // remove
    pcl::CropBox<PointT> roof(true);
    roof.setMin(Eigen::Vector4f (-1.5, -1.7, -1, 1));
    roof.setMax(Eigen::Vector4f (2.6, 1.7, -.4, 1));
    roof.setInputCloud(cloudRegion);
    roof.filter(indices);

    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    for(int point : indices)
        inliers->indices.push_back(point);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloudRegion);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*cloudRegion);

    // auto endTime = std::chrono::steady_clock::now();
    // auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    // std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
    typename pcl::PointCloud<PointT>::Ptr obstCloud (new pcl::PointCloud<PointT> ());
    typename pcl::PointCloud<PointT>::Ptr planeCloud (new pcl::PointCloud<PointT> ());

    for(int index : inliers->indices)
        planeCloud->points.push_back(cloud->points[index]);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (*obstCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);

    // std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloud, cloud);

    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold, double epsAngle)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // RANSAC for removing ground
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    pcl::ModelCoefficients::Ptr coefficients {new pcl::ModelCoefficients};

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    //because we want a specific plane (X-Y Plane)
    Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0); //z axis
    seg.setAxis(axis);
    seg.setEpsAngle(epsAngle*(PI/180.0)); // plane can be within 30 degrees of X-Z plane

    // Segment the largest planar component from the input cloud
    seg.setInputCloud(cloud);
    seg.segment (*inliers, *coefficients);
    
    if(inliers->indices.size() == 0)
    {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }
    
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    // std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::ComplementRANSAC(std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentCloud, float complement_z, float min_z, float max_z)
{   
    // Remove point under min_z to complement RANSAC
    pcl::PassThrough<PointT> pass;
    std::vector<int> indices;
    pass.setInputCloud(segmentCloud.first);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_z, complement_z);
    pass.filter(indices);

    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    for(int point : indices)
    {
        segmentCloud.second->push_back(segmentCloud.first->points[point]);
    }

    pass.setInputCloud(segmentCloud.first);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(complement_z, max_z);
    pass.filter(*segmentCloud.first);

    return segmentCloud;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    //std::cout << "PointCloud before filtering has: " << cloud->points.size ()  << " data points." << std::endl;
    
    // Data containers used
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
    vg.setInputCloud (cloud->makeShared());
    vg.setLeafSize (0.1f, 0.1f, 0.1f);
    vg.filter (*cloud);
    //std::cout << "PointCloud after filtering has: " << cloud->points.size ()  << " data points." << std::endl;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    for(pcl::PointIndices getIndices: clusterIndices)
    {
        typename pcl::PointCloud<PointT>::Ptr cloudCluster (new pcl::PointCloud<PointT>);

        for(int index : getIndices.indices)
            cloudCluster->points.push_back (cloud->points[index]);

        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        clusters.push_back(cloudCluster);
        
    } 


    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    // std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
BoxQ ProcessPointClouds<PointT>::MinimumOrientedBoundingBox(typename pcl::PointCloud<PointT>::Ptr cloudSegmented, double time)
{
    //declare parameters
    int random_points_ = 120;  //first is 80
    float slope_dist_thres_ = 0.5;
    int num_points_thres_ = 10;
    float sensor_height_ = 2.35;
    float roi_m_ = 120;
    float pic_scale_ = 15;
    // get a cloud
    pcl::PointCloud<pcl::PointXYZI> cloud = *cloudSegmented;
    
    // calculating offset so that projecting pointcloud into cv::mat
    cv::Mat m(pic_scale_ * roi_m_, pic_scale_ * roi_m_, CV_8UC1, cv::Scalar(0));
    cv::Point2f tmp_pointcloud_point(cloud[0].x, cloud[0].y);
    cv::Point2f tmp_pointcloud_offset(roi_m_ / 2, roi_m_ / 2);
    cv::Point2f tmp_offset_pointcloud_point = tmp_pointcloud_point + tmp_pointcloud_offset;
    cv::Point tmp_pic_point = tmp_offset_pointcloud_point * pic_scale_;

    int tmp_init_pic_x = tmp_pic_point.x;
    int tmp_init_pic_y = pic_scale_ * roi_m_ - tmp_pic_point.y;

    cv::Point tmp_init_pic_point(tmp_init_pic_x, tmp_init_pic_y);
    cv::Point tmp_init_offset_vec(roi_m_ * pic_scale_ / 2, roi_m_ * pic_scale_ / 2);
    cv::Point offset_init_pic_point = tmp_init_offset_vec - tmp_init_pic_point;

    int num_points = cloud.size();
    std::vector<cv::Point> point_vec(num_points);
    std::vector<cv::Point2f> pointcloud_frame_points(4);

    // init variables
    cv::Point2f min_m_p(0, 0);
    cv::Point2f max_m_p(0, 0);
    float min_m = std::numeric_limits<float>::max();
    float max_m = std::numeric_limits<float>::lowest();

    for (int i_point = 0; i_point < num_points; i_point++)
    {
      const float p_x = cloud[i_point].x;
      const float p_y = cloud[i_point].y;

      // cast (roi _m_/2 < x,y < roi_m_/2) into (0 < x,y < roi_m_)
      cv::Point2f pointcloud_point(p_x, p_y);
      cv::Point2f pointcloud_offset_vec(roi_m_ / 2, roi_m_ / 2);
      cv::Point2f offset_pointcloud_point = pointcloud_point + pointcloud_offset_vec;
      // cast (roi_m_)m*(roi_m_)m into  pic_scale_
      cv::Point scaled_point = offset_pointcloud_point * pic_scale_;
      // cast into image coordinate
      int pic_x = scaled_point.x;
      int pic_y = pic_scale_ * roi_m_ - scaled_point.y;
      // offset so that the object would be locate at the center
      cv::Point pic_point(pic_x, pic_y);
      cv::Point offset_point = pic_point + offset_init_pic_point;

      // Make sure points are inside the image size
      if (offset_point.x > (pic_scale_ * roi_m_) || offset_point.x < 0 || offset_point.y < 0 || offset_point.y > (pic_scale_ * roi_m_))
      {
        continue;
      }
      // cast the pointcloud into cv::mat
      m.at<uchar>(offset_point.y, offset_point.x) = 255;
      point_vec[i_point] = offset_point;
      // calculate min and max slope for x1, x3(edge points)
      float delta_m = p_y / p_x;
      if (delta_m < min_m)
      {
        min_m = delta_m;
        min_m_p.x = p_x;
        min_m_p.y = p_y;
      }

      if (delta_m > max_m)
      {
        max_m = delta_m;
        max_m_p.x = p_x;
        max_m_p.y = p_y;
      }
    // }
    
    if (max_m == std::numeric_limits<float>::lowest() || min_m == std::numeric_limits<float>::max())
    {
      continue;
    }

    }  //without }

    // L shape fitting parameters
    cv::Point2f dist_vec = max_m_p - min_m_p;
    float slope_dist = sqrt(dist_vec.x * dist_vec.x + dist_vec.y * dist_vec.y);
    float slope = (max_m_p.y - min_m_p.y) / (max_m_p.x - min_m_p.x);

    // random variable
    std::mt19937_64 mt;
    mt.seed(time);
    // mt.seed(0);
    std::uniform_int_distribution<> rand_points(0, num_points - 1);

    // start l shape fitting for car like object
    if (slope_dist > slope_dist_thres_ && num_points > num_points_thres_)
    {
      float max_dist = 0;
      cv::Point2f max_p(0, 0);

      // get max distance from random sampling points
      for (int i = 0; i < random_points_; i++)
      {
        int p_ind = rand_points(mt);
        // std::cout << "p_ind : " << p_ind << "   cloud size : " << cloud.size() << std::endl; 
        assert(p_ind >= 0 && p_ind <= (cloud.size() - 1));
        cv::Point2f p_i(cloud[p_ind].x, cloud[p_ind].y);

        // from equation of distance between line and point
        float dist = std::abs(slope * p_i.x - 1 * p_i.y + max_m_p.y - slope * max_m_p.x) / std::sqrt(slope * slope + 1);
        if (dist > max_dist)
        {
          max_dist = dist;
          max_p = p_i;
        }
      }
      // vector adding
      cv::Point2f max_m_vec = max_m_p - max_p;
      cv::Point2f min_m_vec = min_m_p - max_p;
      cv::Point2f last_p = max_p + max_m_vec + min_m_vec;

      pointcloud_frame_points[0] = min_m_p;
      pointcloud_frame_points[1] = max_p;
      pointcloud_frame_points[2] = max_m_p;
      pointcloud_frame_points[3] = last_p;
    }
    else
    {
      // MinAreaRect fitting
      cv::RotatedRect rect_info = cv::minAreaRect(point_vec);
      cv::Point2f rect_points[4];
      rect_info.points(rect_points);
      for (int point_i = 0; point_i < 4; point_i++)
      {
        cv::Point2f offset_point_float(offset_init_pic_point.x, offset_init_pic_point.y);

        cv::Point2f reverse_offset_point = rect_points[point_i] - offset_point_float;
        // reverse from image coordinate to eucledian coordinate
        float r_x = reverse_offset_point.x;
        float r_y = pic_scale_ * roi_m_ - reverse_offset_point.y;
        cv::Point2f eucledian_coordinate_pic_point(r_x, r_y);
        // reverse to roi_m_*roi_m_ scale
        cv::Point2f offset_pointcloud_point = eucledian_coordinate_pic_point * float(1/pic_scale_);
        // reverse from (0 < x,y < roi_m_) to (roi_m_/2 < x,y < roi_m_/2)
        cv::Point2f offset_vec_(roi_m_ / 2, roi_m_ / 2);
        cv::Point2f pointcloud_point = offset_pointcloud_point - offset_vec_;
        pointcloud_frame_points[point_i] = pointcloud_point;
      }
    }

    // updateCpFromPoints
    
    cv::Point2f p1 = pointcloud_frame_points[0];
    cv::Point2f p2 = pointcloud_frame_points[1];
    cv::Point2f p3 = pointcloud_frame_points[2];
    cv::Point2f p4 = pointcloud_frame_points[3];

    PointT minPoint, maxPoint;
    pcl::getMinMax3D(cloud, minPoint, maxPoint);
    /*
    double s1 = ((p4.x - p2.x) * (p1.y - p2.y) - (p4.y - p2.y) * (p1.x - p2.x)) / 2;
    double s2 = ((p4.x - p2.x) * (p2.y - p3.y) - (p4.y - p2.y) * (p2.x - p3.x)) / 2;
    double cx = p1.x + (p3.x - p1.x) * s1 / (s1 + s2);
    double cy = p1.y + (p3.y - p1.y) * s1 / (s1 + s2);
    */
    double cx = (maxPoint.x+minPoint.x)/2.0;
    double cy = (maxPoint.y+minPoint.y)/2.0;
    double cz = (maxPoint.z+minPoint.z)/2.0;
    
    BoxQ box;
    box.bboxTransform = {cx, cy, cz};

    //toRightAngleBBox
    cv::Point2f pp1 = pointcloud_frame_points[0];
    cv::Point2f pp2 = pointcloud_frame_points[1];
    cv::Point2f pp3 = pointcloud_frame_points[2];

    cv::Point2f vec1(pp2.x - pp1.x, pp2.y - pp1.y);
    cv::Point2f vec2(pp3.x - pp2.x, pp3.y - pp2.y);

    // from the equation of inner product
    double cos_theta = vec1.dot(vec2) / (norm(vec1) + norm(vec2));
    double theta = acos(cos_theta);
    double diff_theta = theta - M_PI / 2;

    if (std::abs(diff_theta) > 0.1)
    {
      double m1 = vec1.y / vec1.x;
      double b1 = pp3.y - m1 * pp3.x;
      double m2 = -1.0 / m1;
      double b2 = pp2.y - (m2 * pp2.x);

      double x = (b2 - b1) / (m1 - m2);
      double y = (b2 * m1 - b1 * m2) / (m1 - m2);

      double delta_x = x - pp2.x;
      double delta_y = y - pp2.y;

      pointcloud_frame_points[2].x = x;
      pointcloud_frame_points[2].y = y;
      pointcloud_frame_points[3].x = pointcloud_frame_points[0].x + delta_x;
      pointcloud_frame_points[3].y = pointcloud_frame_points[0].y + delta_y;
    }

    //updateDimentionAndEstimatedAngle
    cv::Point2f ppp1 = pointcloud_frame_points[0];
    cv::Point2f ppp2 = pointcloud_frame_points[1];
    cv::Point2f ppp3 = pointcloud_frame_points[2];

    cv::Point2f vecc1 = ppp1 - ppp2;
    cv::Point2f vecc2 = ppp3 - ppp2;
    double dist1 = norm(vecc1);
    double dist2 = norm(vecc2);
    double bb_yaw;
    // dist1 is length, dist2 is width
    if (dist1 > dist2)
    {
        bb_yaw = atan2(ppp1.y - ppp2.y, ppp1.x - ppp2.x);
        box.cube_length = dist1;
        box.cube_width = dist2;
        box.cube_height = maxPoint.z-minPoint.z;
    }
    // dist1 is width, dist2 is length
    else
    {
        bb_yaw = atan2(ppp3.y - ppp2.y, ppp3.x - ppp2.x);
        box.cube_length = dist2;
        box.cube_width = dist1;
        box.cube_height = maxPoint.z-minPoint.z;
    }
    // convert yaw to quartenion
    tf::Matrix3x3 obs_mat;
    obs_mat.setEulerYPR(bb_yaw, 0, 0);

    tf::Quaternion q_tf;
    obs_mat.getRotation(q_tf);
    box.bboxQuaternion = {q_tf.getW(), q_tf.getX(), q_tf.getY(), q_tf.getZ()};
    
    return box;
    
    // previous
    /*
        // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloudSegmented, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*cloudSegmented, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));  /// This line is necessary for proper orientation in some cases. The numbers come out the same without it, but
                                                                                    ///    the signs are different and the box doesn't get correctly oriented in some cases.
    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*cloudSegmented, *cloudPointsProjected, projectionTransform);
    
    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZI minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    // Final transform
    const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA); //Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
    const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

    //BoxQ box;
    //box.bboxTransform = bboxTransform;
    //box.bboxQuaternion = bboxQuaternion;
    //box.cube_length = maxPoint.x - minPoint.x;
    //box.cube_width = maxPoint.y - minPoint.y;
    //box.cube_height = maxPoint.z - minPoint.z;

    return box;

    */
}



template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}