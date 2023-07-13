#ifndef TRACK_H
#define TRACK_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <numeric>

#include <vector>
#include <cmath>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <visualization_msgs/MarkerArray.h>

#include "tracking/HungarianAlg.h"
#include "marker/customMarker.h"


struct trackingStruct
{
	unsigned int id;
	unsigned int age;
	unsigned cntTotalVisible;
	unsigned cntConsecutiveInvisible;

	jsk_recognition_msgs::BoundingBox cur_bbox;
	jsk_recognition_msgs::BoundingBox pre_bbox;

	pcl::PointXYZI pre_minDistPoint;
	pcl::PointXYZI cur_minDistPoint;

	float vx;
	float vy;
	float v;
	vector<float> v_list;
	vector<float> angle_list;
	double sec;

	cv::KalmanFilter kf;
};

class Track
{
private:
	// Setting parameters
	int stateVariableDim;
	int stateMeasureDim;
	unsigned int nextID;
	unsigned int m_thres_invisibleCnt;

	cv::Mat m_matTransition;
	cv::Mat m_matMeasurement;

	cv::Mat m_matProcessNoiseCov;
	cv::Mat m_matMeasureNoiseCov;

	float m_thres_associationCost;
	
	// Global variables
	vector<trackingStruct> vecTracks;
	vector<pair<int, int>> vecAssignments;
	vector<int> vecUnssignedTracks;
	vector<int> vecUnssignedDetections;

public:
	//constructor
	Track();
	//deconstructor
	~Track();
	float getVectorScale(float v1, float v2);
	double getBBoxIOU(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2);
	double getBBoxDistance(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2);
	jsk_recognition_msgs::BoundingBoxArray filtering(jsk_recognition_msgs::BoundingBoxArray &clusterBboxArray);
	void predictNewLocationOfTracks();
	void assignDetectionsTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray);
	void assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray, vector<pcl::PointXYZI> minDistPoint_vector, const double lidarSec);
	void unassignedTracksUpdate();
	void deleteLostTracks();
	void createNewTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray, vector<pcl::PointXYZI> minDistPoint_vector);
	void deleteOverlappedTracks();
	pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> displayTrack();
};

#endif