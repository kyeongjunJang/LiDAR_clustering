#include "tracking/track.h"

using namespace cv;

float gaussianFilter(const std::vector<float> &data, float sigma)
{
    int kernelSize = 2 * static_cast<int>(std::ceil(2 * sigma)) + 1; // Recommended kernel size for Gaussian filter
    int halfKernelSize = kernelSize / 2;
    std::vector<float> kernel(kernelSize);
    float sum = 0.0f;

    for (int i = 0; i < kernelSize; ++i)
    {
        int x = i - halfKernelSize;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    for (int i = 0; i < kernelSize; ++i)
    {
        kernel[i] /= sum;
    }

    float filteredValue = 0.0f;
    for (int i = 0; i < kernelSize; ++i)
    {
        int dataIndex = i - halfKernelSize;
        if (dataIndex >= 0 && dataIndex < static_cast<int>(data.size()))
        {
            filteredValue += kernel[i] * data[dataIndex];
        }
    }

    return filteredValue;
}
std::pair<float, float> getIQRThreshold(const std::vector<float> &data, float multiplier)
{
    std::vector<float> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());

    int dataSize = sortedData.size();
    int q1Index = dataSize / 4;
    int q3Index = 3 * dataSize / 4;

    float q1 = sortedData[q1Index];
    float q3 = sortedData[q3Index];

    float iqr = q3 - q1;
    float lowerThreshold = q1 - multiplier * iqr;
    float upperThreshold = q3 + multiplier * iqr;

    // return std::max(lowerThreshold, 0.0f); // Ensure the threshold is non-negative
	return std::make_pair(lowerThreshold, upperThreshold);
}

//constructor
Track::Track()
{
	stateVariableDim = 4; // cx, cy, dx, dy
	stateMeasureDim = 2;  // cx, cy
	nextID = 0;
	m_thres_invisibleCnt = 3;

	//A & Q ==> Predict process
	//H & R ==> Estimation process
	
	float dt = 0.1;
	// A
	m_matTransition = (Mat_<float>(stateVariableDim, stateVariableDim) << 1, 0, dt, 0,
																		  0, 1, 0, dt,
																		  0, 0, 0, 0,
																		  0, 0, 0, 0);

	// H
	m_matMeasurement = (Mat_<float>(stateMeasureDim, stateVariableDim) << 1, 0, 0, 0,
																		  0, 1, 0, 0);

	// Q size small -> smooth
	// float Q[] = {1e-5f, 1e-3f, 1e-3f, 1e-3f};
	float Q[] = {1e-2f, 1e-2f, 1e-2f, 1e-2f};
	Mat tempQ(stateVariableDim, 1, CV_32FC1, Q);
	m_matProcessNoiseCov = Mat::diag(tempQ);

	// R
	float R[] = {1e-3f, 1e-3f};
	Mat tempR(stateMeasureDim, 1, CV_32FC1, R);
	m_matMeasureNoiseCov = Mat::diag(tempR);

	m_thres_associationCost = 5.0f;
}

//deconstructor
Track::~Track(){}

float Track::getVectorScale(float v1, float v2)
{
	float distance = sqrt(pow(v1, 2) + pow(v2, 2));
	if (v1 < 0) return -distance;
	else return distance;
}

double Track::getBBoxIOU(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
	double xA = max(boxA[0], boxB[0]);
	double yA = max(boxA[1], boxB[1]);
	double xB = min(boxA[2], boxB[2]);
	double yB = min(boxA[3], boxB[3]);

	double interArea = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1);
 	
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
	double boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

	double iou = interArea / double(boxAArea + boxBArea - interArea);

	return iou;
}

double Track::getBBoxOverlap(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
	double xA = max(boxA[0], boxB[0]);
	double yA = max(boxA[1], boxB[1]);
	double xB = min(boxA[2], boxB[2]);
	double yB = min(boxA[3], boxB[3]);

	double interArea = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1);
 	
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
	double boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

	double overlap = interArea / boxAArea;

	return overlap;
}

double Track::getBBoxGap(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
 	
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
	double boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

	double gap;
	if(boxAArea > boxBArea) gap = boxAArea / boxBArea;
	else gap = boxBArea / boxAArea;

	return gap;
}

double Track::getBBoxDistance(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{	
	float distance = sqrt(pow(bbox2.pose.position.x - bbox1.pose.position.x, 2) + pow(bbox2.pose.position.y - bbox1.pose.position.y, 2));
	return distance;
}

jsk_recognition_msgs::BoundingBoxArray Track::filtering(jsk_recognition_msgs::BoundingBoxArray &clusterBboxArray)
{
	for (int i = 0; i < (int)clusterBboxArray.boxes.size(); i++)
	{
		if (clusterBboxArray.boxes[i].pose.position.z > 1.5 ||
			clusterBboxArray.boxes[i].dimensions.x > 16.0 ||
			clusterBboxArray.boxes[i].dimensions.y > 16.0)
		{
			clusterBboxArray.boxes.erase(clusterBboxArray.boxes.begin() + i);
			i--;
		}
	}

	return clusterBboxArray;
}

void Track::predictNewLocationOfTracks()
{
	for (int i = 0; i < vecTracks.size(); i++)
	{
		// Predict current state
		vecTracks[i].kf.predict();

		vecTracks[i].cur_bbox.pose.position.x = vecTracks[i].kf.statePre.at<float>(0);
		vecTracks[i].cur_bbox.pose.position.y = vecTracks[i].kf.statePre.at<float>(1);
		vecTracks[i].vx = vecTracks[i].kf.statePre.at<float>(2);
		vecTracks[i].vy = vecTracks[i].kf.statePre.at<float>(3);
	}
}

void Track::assignDetectionsTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray)
{
	int N = (int)vecTracks.size();             //  N = number of tracking
	int M = (int)bboxMarkerArray.boxes.size(); //  M = number of detection

	vector<vector<double>> Cost(N, vector<double>(M)); //2 array

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			// Box Over Lap
			// Cost[i][j] = 1 - getBBoxIOU(vecTracks[i].cur_bbox, bboxMarkerArray.boxes[j]);
			// Distance
			Cost[i][j] = getBBoxDistance(vecTracks[i].cur_bbox, bboxMarkerArray.boxes[j]);
		}
	}

	vector<int> assignment;

	if (N != 0)
	{
		AssignmentProblemSolver APS;
		APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);
	}

	vecAssignments.clear();
	vecUnssignedTracks.clear();
	vecUnssignedDetections.clear();

	for (int i = 0; i < N; i++)
	{
		if (assignment[i] == -1)
		{
			vecUnssignedTracks.push_back(i);
		}
		else
		{
			
			if ((Cost[i][assignment[i]] < m_thres_associationCost) && (3 > getBBoxGap(vecTracks[i].cur_bbox, bboxMarkerArray.boxes[assignment[i]])))
			{
				vecAssignments.push_back(pair<int, int>(i, assignment[i]));
			}
			else
			{
				vecUnssignedTracks.push_back(i);
				assignment[i] = -1;
			}
		}
	}

	for (int j = 0; j < M; j++)
	{
		auto it = find(assignment.begin(), assignment.end(), j);
		if (it == assignment.end())
			vecUnssignedDetections.push_back(j);
	}
}

void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray, vector<pcl::PointXYZI> minDistPoint_vector, const double lidarSec)
{
	for (int i = 0; i < (int)vecAssignments.size(); i++)
	{
		int idT = vecAssignments[i].first;
		int idD = vecAssignments[i].second;

		Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
		measure.at<float>(0) = bboxMarkerArray.boxes[idD].pose.position.x;
		measure.at<float>(1) = bboxMarkerArray.boxes[idD].pose.position.y;

		vecTracks[idT].kf.correct(measure);

		//double curSec = ros::Time::now().toSec();
		double curSec = lidarSec;
		double dt = curSec - vecTracks[idT].sec;

		if (sqrt(pow(vecTracks[idT].pre_bbox.pose.position.x, 2) + pow(vecTracks[idT].pre_bbox.pose.position.y, 2)) < 15)
		{
			vecTracks[idT].vx = (bboxMarkerArray.boxes[idD].pose.position.x - vecTracks[idT].pre_bbox.pose.position.x) / dt;
			vecTracks[idT].vy = (bboxMarkerArray.boxes[idD].pose.position.y - vecTracks[idT].pre_bbox.pose.position.y) / dt;
		}
		else
		{
			vecTracks[idT].vx = (minDistPoint_vector[idD].x - vecTracks[idT].pre_minDistPoint.x) / dt;
			vecTracks[idT].vy = (minDistPoint_vector[idD].y - vecTracks[idT].pre_minDistPoint.y) / dt;
		}

		float v = getVectorScale(vecTracks[idT].vx, vecTracks[idT].vy); // m/s

		//cout << vecTracks[idT].v << endl;

		if (vecTracks[idT].v_list.size() < 6)
		{
			vecTracks[idT].v_list.push_back(v);
		}
		else
		{
			if (abs(vecTracks[idT].v - v) < 3){
				for(int i = 0; i < 6; i++)
				{
					vecTracks[idT].v_list[i] = vecTracks[idT].v_list[i + 1];
					vecTracks[idT].v_list.pop_back();
					vecTracks[idT].v_list.push_back(v);
				}
			}
		}
		vecTracks[idT].v = std::accumulate(vecTracks[idT].v_list.begin(), vecTracks[idT].v_list.end(), 0.0) / vecTracks[idT].v_list.size();

		//cout << vecTracks[idT].v << endl;
		
		vecTracks[idT].cur_bbox.pose.orientation.z = std::accumulate(vecTracks[idT].angle_list.begin(), vecTracks[idT].angle_list.end(), 0.0) / vecTracks[idT].angle_list.size();

		vecTracks[idT].cur_bbox.pose.position.x = bboxMarkerArray.boxes[idD].pose.position.x;
		vecTracks[idT].cur_bbox.pose.position.y = bboxMarkerArray.boxes[idD].pose.position.y;
		vecTracks[idT].cur_bbox.pose.position.z = bboxMarkerArray.boxes[idD].pose.position.z;
		vecTracks[idT].cur_bbox.dimensions.x = bboxMarkerArray.boxes[idD].dimensions.x;
		vecTracks[idT].cur_bbox.dimensions.y = bboxMarkerArray.boxes[idD].dimensions.y;
		vecTracks[idT].cur_bbox.dimensions.z = bboxMarkerArray.boxes[idD].dimensions.z;
		vecTracks[idT].cur_bbox.pose.orientation.x = bboxMarkerArray.boxes[idD].pose.orientation.x;
		vecTracks[idT].cur_bbox.pose.orientation.y = bboxMarkerArray.boxes[idD].pose.orientation.y;
		vecTracks[idT].cur_bbox.pose.orientation.z = bboxMarkerArray.boxes[idD].pose.orientation.z;
		vecTracks[idT].cur_bbox.pose.orientation.w = bboxMarkerArray.boxes[idD].pose.orientation.w;

		vecTracks[idT].pre_bbox = vecTracks[idT].cur_bbox;
		vecTracks[idT].pre_minDistPoint = minDistPoint_vector[idD];
		vecTracks[idT].sec = curSec;

		vecTracks[idT].age++;
		vecTracks[idT].cntTotalVisible++;
		vecTracks[idT].cntConsecutiveInvisible = 0;
	}
}

void Track::unassignedTracksUpdate()
{
	for (int i = 0; i < (int)vecUnssignedTracks.size(); i++)
	{
		int id = vecUnssignedTracks[i];
		vecTracks[id].age++;
		vecTracks[id].cntConsecutiveInvisible++;
	}
}

void Track::deleteLostTracks()
{
	if ((int)vecTracks.size() == 0)
	{
		return;
	}
	for (int i = 0; i < (int)vecTracks.size(); i++)
	{
		if (vecTracks[i].cntConsecutiveInvisible >= m_thres_invisibleCnt)
		{
			vecTracks.erase(vecTracks.begin() + i);
			i--;
		}
	}

}

void Track::createNewTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray, vector<pcl::PointXYZI> minDistPoint_vector)
{
	for (int i = 0; i < (int)vecUnssignedDetections.size(); i++)
	{
		int id = vecUnssignedDetections[i];

		trackingStruct ts;
		ts.id = nextID++;
		ts.age = 1;
		ts.cntTotalVisible = 1;
		ts.cntConsecutiveInvisible = 0;

		ts.cur_bbox = bboxMarkerArray.boxes[id];
		ts.pre_bbox = bboxMarkerArray.boxes[id];

		ts.pre_minDistPoint = minDistPoint_vector[id];
		ts.cur_minDistPoint = minDistPoint_vector[id];

		ts.vx = 0.0;
		ts.vy = 0.0;
		ts.v = 0.0;
		ts.sec = 0.0;

		ts.kf.init(stateVariableDim, stateMeasureDim);

		m_matTransition.copyTo(ts.kf.transitionMatrix);         //A
		m_matMeasurement.copyTo(ts.kf.measurementMatrix);       //H

		m_matProcessNoiseCov.copyTo(ts.kf.processNoiseCov);     //Q
		m_matMeasureNoiseCov.copyTo(ts.kf.measurementNoiseCov); //R

		Mat tempCov(stateVariableDim, 1, CV_32FC1, 1);
		ts.kf.errorCovPost = Mat::diag(tempCov);

		ts.kf.statePost.at<float>(0) = ts.cur_bbox.pose.position.x;
		ts.kf.statePost.at<float>(1) = ts.cur_bbox.pose.position.y;
		ts.kf.statePost.at<float>(2) = ts.vx;
		ts.kf.statePost.at<float>(3) = ts.vy;

		vecTracks.push_back(ts);
	}
}

void Track::deleteOverlappedTracks()
{
    for (auto it = vecTracks.begin(); it != vecTracks.end(); ++it) {
        auto next_it = std::next(it);
        while (next_it != vecTracks.end()) {
            double IoU = getBBoxOverlap(it->cur_bbox, next_it->cur_bbox);
            if (IoU > 0.3){
                if (it->age < next_it->age) {
                    it = vecTracks.erase(it);
                } else {
                    next_it = vecTracks.erase(next_it);
                }
            } else {
                ++next_it;
            }
        }
    }
}

pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> Track::displayTrack()
{   
	CustomMarker customMarker;
	jsk_recognition_msgs::BoundingBoxArray bboxArray;
	visualization_msgs::MarkerArray textArray;

	
	for (int i = 0; i < vecTracks.size(); i++)
	{
		vecTracks[i].cur_bbox.label = vecTracks[i].age;
		// if (vecTracks[i].age >= 3)
		if (vecTracks[i].age >= 2)
		{
			vecTracks[i].cur_bbox.value = vecTracks[i].v;
			bboxArray.boxes.push_back(vecTracks[i].cur_bbox);
			textArray.markers.push_back(customMarker.get_text_msg(vecTracks[i], i));
		}
	}
	
	pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_text(bboxArray, textArray);
	return bbox_text;
}