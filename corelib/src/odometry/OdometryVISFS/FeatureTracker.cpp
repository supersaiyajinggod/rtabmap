/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdarg>

#include "rtabmap/core/odometry/OdometryVISFS/FeatureTracker.h"
#include "rtabmap/core/util3d_features.h"
#include "rtabmap/core/util3d_motion_estimation.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/Optimizer.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/utilite/UStl.h"

namespace rtabmap {

FeatureTracker::FeatureTracker(const ParametersMap & _parameters, FeatureManager * _featureManager) : 
	featureManager_(_featureManager),
	force3DoF_(Parameters::defaultRegForce3DoF()),
	displayTracker_(Parameters::defaultOdomVISFSDisplayTracker()),
	maxFeatures_(Parameters::defaultOdomVISFSMaxFeatures()),
	qualityLevel_(Parameters::defaultGFTTQualityLevel()),
	minDistance_(Parameters::defaultGFTTMinDistance()),
	flowBack_(Parameters::defaultOdomVISFSFlowBack()),
	maxDepth_(Parameters::defaultOdomVISFSMaxDepth()),
	minDepth_(Parameters::defaultOdomVISFSMinDepth()),
	flowWinSize_(Parameters::defaultOdomVISFSFlowWinSize()),
	flowIterations_(Parameters::defaultOdomVISFSFlowIterations()),
	flowEps_(Parameters::defaultOdomVISFSFlowEps()),
	flowMaxLevel_(Parameters::defaultOdomVISFSFlowMaxLevel()),
	cullByFundationMatrix_(Parameters::defaultOdomVISFSCullByFundationMatrix()),
	fundationPixelError_(Parameters::defaultOdomVISFSFundationPixelError()),
	minInliers_(Parameters::defaultOdomVISFSMinInliers()),
	pnpIterations_(Parameters::defaultOdomVISFSPnPIterations()),
	pnpReprojError_(Parameters::defaultOdomVISFSPnPReprojError()),
	pnpFlags_(Parameters::defaultOdomVISFSPnPFlags()),
	refineIterations_(Parameters::defaultOdomVISFSRefineIterations()),
	bundleAdjustment_(Parameters::defaultOdomVISFSBundleAdjustment()) {
	parameters_ = _parameters;

	Parameters::parse(parameters_, Parameters::kRegForce3DoF(), force3DoF_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSDisplayTracker(), displayTracker_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMaxFeatures(), maxFeatures_);
	Parameters::parse(parameters_, Parameters::kGFTTQualityLevel(), qualityLevel_);
	Parameters::parse(parameters_, Parameters::kGFTTMinDistance(), minDistance_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowBack(), flowBack_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMaxDepth(), maxDepth_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMinDepth(), minDepth_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowWinSize(), flowWinSize_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowMaxLevel(), flowMaxLevel_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSCullByFundationMatrix(), cullByFundationMatrix_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFundationPixelError(), fundationPixelError_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowIterations(), flowIterations_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowEps(), flowEps_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMinInliers(), minInliers_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSPnPIterations(), pnpIterations_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSPnPReprojError(), pnpReprojError_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSPnPFlags(), pnpFlags_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSRefineIterations(), refineIterations_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSBundleAdjustment(), bundleAdjustment_);
	uInsert(bundleParameters_, parameters_);
}

FeatureTracker::~FeatureTracker() {}

std::vector<cv::Point3f> FeatureTracker::generateKeyPoints3D(const SensorData & _data, const std::vector<cv::KeyPoint> & _keyPoints) const {
	std::vector<cv::Point3f> keyPoints3D;
	if (_keyPoints.size()) {
		if (!_data.depthRaw().empty() && _data.cameraModels().size()) {
			keyPoints3D = util3d::generateKeypoints3DDepth(_keyPoints, _data.depthRaw(), _data.cameraModels(), minDepth_, maxDepth_);
		}
	}
	return keyPoints3D;
}

std::vector<cv::Point3f> FeatureTracker::points2NormalizedPlane(const std::vector<cv::Point2f> & _points, const CameraModel & _cameraModel) const {
	const float invK11 = 1 / static_cast<float>(_cameraModel.fx());
	const float invK13 = -1.f * static_cast<float>(_cameraModel.cx()) / static_cast<float>(_cameraModel.fx());
	const float invK22 = 1 / static_cast<float>(_cameraModel.fy());
	const float invK23 = -1.f * static_cast<float>(_cameraModel.cy()) / static_cast<float>(_cameraModel.fy());

	std::vector<cv::Point3f> pointsInNormalizedPlane;
	for (auto point : _points) {
		cv::Point3f pt;
		pt.x = invK11 * point.x + invK13;
		pt.y = invK22 * point.y + invK23;
		pt.z = 1.f;
		pointsInNormalizedPlane.push_back(pt);
	}

	return pointsInNormalizedPlane;
}

std::vector<cv::Point2f> FeatureTracker::points2VirtualImage(const std::vector<cv::Point3f> & _points) const {
	const float focalLength = 532.f;
	const float halfCol = 320.f;
	const float halfRow = 240.f;
	std::vector<cv::Point2f> virtualCorners(_points.size());

	for (std::size_t i = 0; i < _points.size(); ++i) {
		cv::Point2f pt;
		pt.x = focalLength * virtualCorners[i].x + halfCol;
		pt.y = focalLength * virtualCorners[i].y + halfRow;
		virtualCorners[i] = pt;	
	}
	return virtualCorners;
}

void FeatureTracker::rejectOutlierWithFundationMatrix(const std::vector<cv::Point2f> & _cornersFrom, const std::vector<cv::Point2f> & _cornersTo, std::vector<unsigned char> & _status) const {
	UASSERT(_cornersFrom.size() == _cornersTo.size());

	std::vector<unsigned char> statusFundationMatrix;
	cv::findFundamentalMat(_cornersFrom, _cornersTo, cv::FM_RANSAC, static_cast<double>(fundationPixelError_), 0.99, statusFundationMatrix);

	UASSERT(_status.size() == statusFundationMatrix.size());
	for (std::size_t i = 0; i < _status.size(); ++i) {
		if (_status[i] && statusFundationMatrix[i]) {
			_status[i] = 1;
		} else {
			_status[i] = 0;
		}
	}		

}

inline float FeatureTracker::distanceL2(const cv::Point2f & pt1, const cv::Point2f & pt2) const {
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void FeatureTracker::displayTracker(int _n, ...) const {
	UASSERT(_n == 8);

	std::va_list vl;
	va_start(vl, _n);

	if (_n == 8) {	// Subsequence: ColorImageTo GrayImageFrom, GrayImageTo, cornersFrom, cornersTo, newCornersInTo, status, TrackerInfo
		// cv::Mat imageToOri;
		// cv::Mat imageFrom;
		// cv::Mat imageTo;
		std::vector<cv::Mat> imagesContainer;
		// std::vector<cv::Point2f> cornersFrom;
		// std::vector<cv::Point2f> cornersTo;
		// std::vector<cv::Point2f> newCornersInTo;
		std::vector<std::vector<cv::Point2f>> cornersContainer;
		std::vector<unsigned char> status;
		int isKeyFrame = 0;
		for (auto i = 0; i < _n; ++i) {
			if (i <= 2) {
				cv::Mat tempImage = va_arg(vl, cv::Mat);
				if (tempImage.channels() !=3 )
					cv::cvtColor(tempImage, tempImage, CV_GRAY2BGR);
				imagesContainer.push_back(tempImage);
			} else if (i > 2 && i < _n - 2) {
				cornersContainer.push_back(va_arg(vl, std::vector<cv::Point2f>));
			} else if (i == _n - 2) {
				status = va_arg(vl, std::vector<unsigned char>);
			} else if (i == _n - 1) {
				isKeyFrame = va_arg(vl, int);
			}
		}
		va_end(vl);
		UASSERT(imagesContainer.size() == 3);
		UASSERT(cornersContainer.size() == 3);

		const float cols = static_cast<float>(imagesContainer[2].cols);
		cv::Mat top, bottom, fullImage;
		cv::hconcat(imagesContainer[1], imagesContainer[2], top);
		cv::hconcat(imagesContainer[1], imagesContainer[2], bottom);
		if (1 && (isKeyFrame != 0)) {
			std::string text = "KeyFrame";
			int baseLine;
			cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_COMPLEX, 2, 2, &baseLine);
			cv::Point textOrigin;
			textOrigin.x = imagesContainer[2].cols/2 - textSize.width/2 + cols;
			textOrigin.y = imagesContainer[2].cols/2 - textSize.height/2;
			cv::putText(top, text, textOrigin, cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 255), 2);
		}

		for (std::size_t i = 0; i < status.size(); ++i) {
			cv::circle(top, cornersContainer[0][i], 2, cv::Scalar(0, 0, 255), 1);
			cv::circle(bottom, cornersContainer[0][i], 2, cv::Scalar(0, 0, 255), 1);
			if (status[i]) {
				cornersContainer[1][i].x += cols;
				cv::circle(bottom, cornersContainer[1][i], 2, cv::Scalar(0, 0, 255), 1);
				cv::line(bottom, cornersContainer[0][i], cornersContainer[1][i], cv::Scalar(0, 201, 80), 1);
			} else {
				cornersContainer[1][i].x += cols;
				cv::circle(bottom, cornersContainer[1][i], 2, cv::Scalar(176, 48, 96), 1);
			}
		}
		for (auto & corner : cornersContainer[2]) {
			corner.x += cols;
			cv::circle(top, corner, 2, cv::Scalar(255, 255, 0), 1);
			// cv::circle(bottom, corner, 2, cv::Scalar(255, 255, 0), 1);
		}

		cv::vconcat(top, bottom, fullImage);
		cv::namedWindow("Tracker", CV_WINDOW_NORMAL);
		cv::imshow("Tracker", fullImage);
		cv::waitKey(5);
	}
}

Transform FeatureTracker::computeTransformation(const Signature & _fromSignature, const Signature & _toSignature, Transform _guess, TrackerInfo * _infoOut) const {
	Signature fromCopy(_fromSignature);
	Signature toCopy(_toSignature);
	return computeTransformationMod(fromCopy, toCopy, _guess, _infoOut);
}

Transform FeatureTracker::computeTransformation(const SensorData & _fromSignature, const SensorData & _toSignature, Transform _guess, TrackerInfo * _infoOut) const {
	Signature fromCopy(_fromSignature);
	Signature toCopy(_toSignature);
	return computeTransformationMod(fromCopy, toCopy, _guess, _infoOut);
}

Transform FeatureTracker::computeTransformationMod(Signature & _fromSignature, Signature & _toSignature, Transform _guess, TrackerInfo * _infoOut) const {
	UTimer time;
	TrackerInfo info;
	if(_infoOut) {
		info = *_infoOut;
	}

	if(!_guess.isNull() && force3DoF_) {
		_guess = _guess.to3DoF();
	}

	// just some checks to make sure that input data are ok
	UASSERT(_fromSignature.getWords().empty() ||
			_fromSignature.getWords3().empty() ||
			(_fromSignature.getWords().size() == _fromSignature.getWords3().size()));
	UASSERT((int)_fromSignature.sensorData().keypoints().size() == _fromSignature.sensorData().descriptors().rows ||
			_fromSignature.getWords().size() == _fromSignature.getWordsDescriptors().size() ||
			_fromSignature.sensorData().descriptors().rows == 0 ||
			_fromSignature.getWordsDescriptors().size() == 0);
	UASSERT((_toSignature.getWords().empty() && _toSignature.getWords3().empty())||
			(_toSignature.getWords().size() && _toSignature.getWords3().empty())||
			(_toSignature.getWords().size() == _toSignature.getWords3().size()));
	UASSERT((int)_toSignature.sensorData().keypoints().size() == _toSignature.sensorData().descriptors().rows ||
			_toSignature.getWords().size() == _toSignature.getWordsDescriptors().size() ||
			_toSignature.sensorData().descriptors().rows == 0 ||
			_toSignature.getWordsDescriptors().size() == 0);
	UASSERT(_fromSignature.sensorData().imageRaw().empty() ||
			_fromSignature.sensorData().imageRaw().type() == CV_8UC1 ||
			_fromSignature.sensorData().imageRaw().type() == CV_8UC3);
	UASSERT(_toSignature.sensorData().imageRaw().empty() ||
			_toSignature.sensorData().imageRaw().type() == CV_8UC1 ||
			_toSignature.sensorData().imageRaw().type() == CV_8UC3);
	
	if (_fromSignature.sensorData().imageRaw().empty() || _toSignature.sensorData().imageRaw().empty()) {
		UERROR("The from signature or the to signature is empty.");
		return Transform::getIdentity();
	}

	_infoOut->projectedIDs.clear();

	std::vector<cv::KeyPoint> kptsFrom;
	std::vector<int> orignalWordsFromIds;
	cv::Mat imageFrom = _fromSignature.sensorData().imageRaw();
	cv::Mat imageTo = _toSignature.sensorData().imageRaw();
	cv::Mat imageToOri = _toSignature.sensorData().imageRaw();

	if (imageFrom.channels() > 1) {
		cv::Mat tmp;
		cv::cvtColor(imageFrom, tmp, cv::COLOR_BGR2GRAY);
		imageFrom = tmp;
	}
	if (imageTo.channels() > 1) {
		cv::Mat tmp;
		cv::cvtColor(imageTo, tmp, cv::COLOR_BGR2GRAY);
		imageTo = tmp;
	}

	if (_fromSignature.getWords().empty()) {
		if (_fromSignature.sensorData().keypoints().empty()) {
			std::vector<cv::Point2f> corners;
			cv::goodFeaturesToTrack(imageFrom, corners, maxFeatures_, qualityLevel_, minDistance_);
			std::for_each(corners.begin(), corners.end(), [&kptsFrom](cv::Point2f point){
				kptsFrom.push_back(cv::KeyPoint(point, 1.f));
			});
			UDEBUG("New extract feature: %d.", static_cast<int>(kptsFrom.size()));
		} else {
			kptsFrom = _fromSignature.sensorData().keypoints();
			UDEBUG("Get key points from the former signature's keypoints: %d.", static_cast<int>(kptsFrom.size()));
		}
	} else {		// Process the former extracted keypoints. //////
		kptsFrom.resize(_fromSignature.getWords().size());
		orignalWordsFromIds.resize(_fromSignature.getWords().size());
		int index = 0;
		bool allWordsUniques = true;
		for (std::multimap<int, cv::KeyPoint>::const_iterator iter = _fromSignature.getWords().begin(); iter != _fromSignature.getWords().end(); ++iter) {
			kptsFrom[index] = iter->second;
			orignalWordsFromIds[index] = iter->first;
			if (index > 0 && iter->first == orignalWordsFromIds[index - 1]) {
				allWordsUniques = false;
			}
			++index;
		}
		if (!allWordsUniques) {
			UINFO("IDs are not unique, IDs will be regenerated!");
			// orignalWordsFromIds.clear();			
		}
		UDEBUG("Get key points from the former signature's words2d: %d.", static_cast<int>(kptsFrom.size()));
	}

	// Generate the from signature key points in 3d.
	std::vector<cv::Point3f> kptsFrom3D;
	if (kptsFrom.size() == _fromSignature.getWords3().size()) {
		kptsFrom3D = uValues(_fromSignature.getWords3());
	} else if (kptsFrom.size() == _fromSignature.sensorData().keypoints3D().size()) {
		kptsFrom3D = _fromSignature.sensorData().keypoints3D();
	} else {
		kptsFrom3D = generateKeyPoints3D(_fromSignature.sensorData(), kptsFrom);
	}
	UDEBUG("Size of kptsFrom3D = %d", static_cast<int>(kptsFrom3D.size()));

	std::multimap<int, cv::KeyPoint> wordsFrom;
	std::multimap<int, cv::KeyPoint> wordsTo;
	std::multimap<int, cv::Point3f> words3From;
	std::multimap<int, cv::Point3f> words3To;
	std::multimap<int, cv::Mat> wordsDescFrom;
	std::multimap<int, cv::Mat> wordsDescTo;

	// Do a initial estimate of the to signature key points's pixel positon.
	std::vector<cv::Point2f> cornersFrom;
	cv::KeyPoint::convert(kptsFrom, cornersFrom);
	std::vector<cv::Point2f> cornersTo;
	bool guessSet = !_guess.isIdentity() && !_guess.isNull();
	if (guessSet) {
		Transform localTransform = _fromSignature.sensorData().cameraModels().size()?_fromSignature.sensorData().cameraModels()[0].localTransform():_fromSignature.sensorData().stereoCameraModel().left().localTransform();
		Transform guessCameraRef = (_guess * localTransform).inverse();
		cv::Mat R = (cv::Mat_<double>(3,3) <<
				(double)guessCameraRef.r11(), (double)guessCameraRef.r12(), (double)guessCameraRef.r13(),
				(double)guessCameraRef.r21(), (double)guessCameraRef.r22(), (double)guessCameraRef.r23(),
				(double)guessCameraRef.r31(), (double)guessCameraRef.r32(), (double)guessCameraRef.r33());
		cv::Mat	rvec(1, 3, CV_64FC1);
		cv::Rodrigues(R, rvec);
		cv::Mat tvec = (cv::Mat_<double>(1,3) << (double)guessCameraRef.x(), (double)guessCameraRef.y(), (double)guessCameraRef.z());
		cv::Mat K = _fromSignature.sensorData().cameraModels().size()?_fromSignature.sensorData().cameraModels()[0].K():_fromSignature.sensorData().stereoCameraModel().left().K();
		cv::projectPoints(kptsFrom3D, rvec, tvec, K, cv::Mat(), cornersTo);
	}

	// Find features in the new left image
	UDEBUG("guessSet = %d", guessSet?1:0);
	std::vector<unsigned char> status;
	std::vector<float> err;
	UDEBUG("cv::calcOpticalFlowPyrLK() begin");
	cv::calcOpticalFlowPyrLK(imageFrom, imageTo, cornersFrom, cornersTo, status, err, cv::Size(flowWinSize_, flowWinSize_), flowMaxLevel_,
								cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, flowIterations_, flowEps_),
								cv::OPTFLOW_LK_GET_MIN_EIGENVALS | (guessSet?cv::OPTFLOW_USE_INITIAL_FLOW:0), 1e-4);
	if (flowBack_) {
		std::vector<unsigned char> reverseStatus;
		std::vector<cv::Point2f> cornersReverse = cornersFrom;
		cv::calcOpticalFlowPyrLK(imageTo, imageFrom, cornersTo, cornersReverse, reverseStatus, err, cv::Size(flowWinSize_, flowWinSize_), flowMaxLevel_,
								cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, flowIterations_, flowEps_),
								cv::OPTFLOW_LK_GET_MIN_EIGENVALS | cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
		for (std::size_t i = 0; i < status.size(); ++i) {
			if (status[i] && reverseStatus[i] && distanceL2(cornersReverse[i], cornersFrom[i]) <= 1.5) {
				status[i] = 1;
			} else {
				status[i] = 0;
			}
		}
	}
	if (!flowBack_ && cullByFundationMatrix_) {
		// std::vector<cv::Point2f> virtualCornersFrom = points2VirtualImage(points2NormalizedPlane(cornersFrom, _fromSignature.sensorData().cameraModels()[0]));
		// std::vector<cv::Point2f> virtualCornersTo = points2VirtualImage(points2NormalizedPlane(cornersTo, _toSignature.sensorData().cameraModels()[0]));
		rejectOutlierWithFundationMatrix(cornersFrom, cornersTo, status);
	}
	UDEBUG("cv::calcOpticalFlowPyrLK() end");

	// Reduce feature vector
	UASSERT(kptsFrom.size() == kptsFrom3D.size());
	std::vector<cv::KeyPoint> kptsTo(kptsFrom.size());
	std::vector<cv::Point3f> kptsFrom3DKept(kptsFrom3D.size());
	std::vector<int> orignalWordsFromIdsCpy = orignalWordsFromIds;
	int index = 0;
	for (std::size_t i = 0; i < status.size(); ++i) {
		if (status[i] && uIsInBounds(cornersTo[i].x, 0.f, static_cast<float>(imageTo.cols)) && uIsInBounds(cornersTo[i].y, 0.f, static_cast<float>(imageTo.rows))) {
			if (orignalWordsFromIdsCpy.size()) {
				orignalWordsFromIds[index] = orignalWordsFromIdsCpy[i];
			}
			kptsFrom[index] = cv::KeyPoint(cornersFrom[i], 1);
			kptsFrom3DKept[index] = kptsFrom3D[i];
			kptsTo[index++] = cv::KeyPoint(cornersTo[i], 1);
		}
	}
	if (orignalWordsFromIds.size())
		orignalWordsFromIds.resize(index);
	kptsFrom.resize(index);
	kptsTo.resize(index);
	kptsFrom3DKept.resize(index);
	kptsFrom3D = kptsFrom3DKept;

	std::vector<cv::Point3f> kptsTo3D;
	kptsTo3D = generateKeyPoints3D(_toSignature.sensorData(), kptsTo);

	// Words management, create a temporary id for each key point.
	UASSERT(kptsFrom.size() == kptsFrom3DKept.size());
	UASSERT(kptsFrom.size() == kptsTo.size());
	UASSERT(kptsTo3D.size() == 0 || kptsTo.size() == kptsTo3D.size());

	if (_fromSignature.getWords().empty() && _fromSignature.sensorData().keypoints3D().empty()) {
		auto wordsId = featureManager_->addFeature(kptsFrom, kptsFrom3D, wordsFrom, words3From);
		for (std::size_t i = 0; i < wordsId.size(); ++i) {
			wordsTo.insert(std::pair<int, cv::KeyPoint>(wordsId[i], kptsTo[i]));
			words3To.insert(std::pair<int, cv::Point3f>(wordsId[i], kptsTo3D[i]));
		}
	} else {
		// update words still tracking.
		for (std::size_t i = 0; i < kptsFrom3DKept.size(); ++i) {
			int id = orignalWordsFromIds.size()?orignalWordsFromIds[i] : static_cast<int>(i);
			wordsFrom.insert(std::pair<int, cv::KeyPoint>(id, kptsFrom[i]));
			words3From.insert(std::pair<int, cv::Point3f>(id, kptsFrom3DKept[i]));
			wordsTo.insert(std::pair<int, cv::KeyPoint>(id, kptsTo[i]));
			words3To.insert(std::pair<int, cv::Point3f>(id, kptsTo3D[i]));
		}
	}

	_fromSignature.setWords(wordsFrom);
	_fromSignature.setWords3(words3From);
	_toSignature.setWords(wordsTo);
	_toSignature.setWords3(words3To);
	_infoOut->matchesInImageIDs = orignalWordsFromIds;

	// Motion estimation, 3D->2D PnP.
	Transform transform;
	cv::Mat covariance = cv::Mat::eye(6, 6, CV_64FC1);
	_infoOut->inliersIDs.clear();
	_infoOut->matchesIDs.clear();
	std::string msg;
	std::vector<int> allInliers;
	std::vector<int> allMatches;

	if (_toSignature.getWords().size()) {
		UDEBUG("Motion estimation start.");
		if (!_toSignature.sensorData().stereoCameraModel().isValidForProjection() && (_toSignature.sensorData().cameraModels().size() != 1 || !_toSignature.sensorData().cameraModels()[0].isValidForProjection())) {
			UERROR("Calibrated camera required (multi-cameras not supported). Id=%d Models=%d StereoModel=%d weight=%d",
				_toSignature.id(), (int)_toSignature.sensorData().cameraModels().size(), _toSignature.sensorData().stereoCameraModel().isValidForProjection()?1:0, _toSignature.getWeight());
		} else {
			UDEBUG("words from3D=%d to2D=%d", (int)_fromSignature.getWords3().size(), (int)_toSignature.getWords().size());
			if (static_cast<int>(_fromSignature.getWords3().size()) >= minInliers_ && static_cast<int>(_toSignature.getWords().size()) >= minInliers_) {
				UASSERT(_toSignature.sensorData().stereoCameraModel().isValidForProjection() || (_toSignature.sensorData().cameraModels().size() == 1 && _toSignature.sensorData().cameraModels()[0].isValidForProjection()));
				const CameraModel & cameraModel = _toSignature.sensorData().stereoCameraModel().isValidForProjection()?_toSignature.sensorData().stereoCameraModel().left():_toSignature.sensorData().cameraModels()[0];
				transform = util3d::estimateMotion3DTo2D(uMultimapToMapUnique(_fromSignature.getWords3()), uMultimapToMapUnique(_toSignature.getWords()),
								cameraModel, minInliers_, pnpIterations_, pnpReprojError_, pnpFlags_, refineIterations_, !_guess.isNull()?_guess:Transform::getIdentity(),
								uMultimapToMapUnique(_toSignature.getWords3()), &covariance, &allMatches, &allInliers);
				UDEBUG("Inliers: %d/%d", (int)allInliers.size(), (int)allMatches.size());
				if (transform.isNull()) {
					msg = uFormat("Not enough inliers %d/%d (matches=%d) between %d and %d",
							static_cast<int>(allInliers.size()), minInliers_, static_cast<int>(allMatches.size()), _fromSignature.id(), _toSignature.id());
					UINFO(msg.c_str());
				} else if(force3DoF_) {
					transform = transform.to3DoF();
				}
			} else {
				msg = uFormat("Not enough features in images (old=%d, new=%d, min=%d)",
						static_cast<int>(_fromSignature.getWords3().size()), static_cast<int>(_toSignature.getWords().size()), minInliers_);
				UINFO(msg.c_str());
			}
		}
		UDEBUG("Motion estimation end.");

		// Bundle adjustment
		if (bundleAdjustment_ > 0 && !transform.isNull() && allInliers.size() && _fromSignature.getWords3().size() && _toSignature.getWords().size()
			&& _fromSignature.sensorData().cameraModels().size() <= 1 && _toSignature.sensorData().cameraModels().size() <= 1) {
			UDEBUG("Refine with bundle adjustment");
			Optimizer * sba = Optimizer::create(bundleAdjustment_ == 3 ? Optimizer::kTypeCeres : bundleAdjustment_ == 2 ? Optimizer::kTypeCVSBA : Optimizer::kTypeG2O, bundleParameters_);

			std::map<int, Transform> poses;
			std::multimap<int, Link> links;
			std::map<int, cv::Point3f> points3DMap;

			poses.insert(std::make_pair(1, Transform::getIdentity()));
			poses.insert(std::make_pair(2, transform));

			UASSERT(covariance.cols == 6 && covariance.rows == 6 && covariance.type() == CV_64FC1);
			if (covariance.at<double>(0, 0) <= COVARIANCE_EPSILON)
				covariance.at<double>(0, 0) = COVARIANCE_EPSILON;
			if (covariance.at<double>(1, 1) <= COVARIANCE_EPSILON)
				covariance.at<double>(1, 1) = COVARIANCE_EPSILON;
			if (covariance.at<double>(2, 2) <= COVARIANCE_EPSILON)
				covariance.at<double>(2, 2) = COVARIANCE_EPSILON;
			if (covariance.at<double>(3, 3) <= COVARIANCE_EPSILON)
				covariance.at<double>(3, 3) = COVARIANCE_EPSILON;
			if (covariance.at<double>(4, 4) <= COVARIANCE_EPSILON)
				covariance.at<double>(4, 4) = COVARIANCE_EPSILON;
			if (covariance.at<double>(5, 5) <= COVARIANCE_EPSILON)
				covariance.at<double>(5, 5) = COVARIANCE_EPSILON; 

			links.insert(std::make_pair(1, Link(1, 2, Link::kNeighbor, transform, covariance.inv())));

			UASSERT(_toSignature.sensorData().stereoCameraModel().isValidForProjection() ||
					(_toSignature.sensorData().cameraModels().size() == 1 && _toSignature.sensorData().cameraModels()[0].isValidForProjection()));
			std::map<int, CameraModel> models;
			Transform invLocalTransformFrom;
			CameraModel cameraModelFrom;
			if (_fromSignature.sensorData().stereoCameraModel().isValidForProjection()) {
				cameraModelFrom = _fromSignature.sensorData().stereoCameraModel().left();
				// Set Tx = -baseline*fx for stereo BA
				cameraModelFrom = CameraModel(cameraModelFrom.fx(), cameraModelFrom.fy(), cameraModelFrom.cx(), cameraModelFrom.cy(),
									cameraModelFrom.localTransform(), -_fromSignature.sensorData().stereoCameraModel().baseline()*cameraModelFrom.fy());
				invLocalTransformFrom = _toSignature.sensorData().stereoCameraModel().localTransform().inverse();
			} else if (_fromSignature.sensorData().cameraModels().size() == 1) {
				cameraModelFrom = _fromSignature.sensorData().cameraModels()[0];
				invLocalTransformFrom = _toSignature.sensorData().cameraModels()[0].localTransform().inverse();
			}

			Transform invLocalTransformTo = Transform::getIdentity();
			CameraModel cameraModelTo;
			if (_toSignature.sensorData().stereoCameraModel().isValidForProjection()) {
				cameraModelTo = _toSignature.sensorData().stereoCameraModel().left();
				cameraModelTo = CameraModel(cameraModelTo.fx(), cameraModelTo.fy(), cameraModelTo.cx(), cameraModelTo.cy(),
									cameraModelTo.localTransform(), -_toSignature.sensorData().stereoCameraModel().baseline()*cameraModelTo.fy());
				invLocalTransformTo = _toSignature.sensorData().stereoCameraModel().localTransform().inverse();
			} else if (_toSignature.sensorData().cameraModels().size() == 1) {
				cameraModelTo = _toSignature.sensorData().cameraModels()[0];
				invLocalTransformTo = _toSignature.sensorData().cameraModels()[0].localTransform().inverse();
			}
			if (invLocalTransformFrom.isNull()) {
				invLocalTransformFrom = invLocalTransformTo;
			}

			models.insert(std::make_pair(1, cameraModelFrom.isValidForProjection()?cameraModelFrom:cameraModelTo));
			models.insert(std::make_pair(2, cameraModelTo));

			std::map<int, std::map<int, FeatureBA>> wordReferences;
			for (size_t i = 0; i < allInliers.size(); ++i) {
				int wordId = allInliers[i];
				const cv::Point3f & pt3d = _fromSignature.getWords3().find(wordId)->second;
				points3DMap.insert(std::make_pair(wordId, pt3d));

				std::map<int, FeatureBA> ptMap;
				if (_fromSignature.getWords().size() && cameraModelFrom.isValidForProjection()) {
					float depthFrom = util3d::transformPoint(pt3d, invLocalTransformFrom).z;
					const cv::KeyPoint & kpt = _fromSignature.getWords().find(wordId)->second;
					ptMap.insert(std::make_pair(1, FeatureBA(kpt, depthFrom)));
				}
				if (_toSignature.getWords().size() && cameraModelTo.isValidForProjection()) {
					float depthTo = 0.f;
					if (_toSignature.getWords3().find(wordId) != _toSignature.getWords3().end()) {
						depthTo = util3d::transformPoint(_toSignature.getWords3().find(wordId)->second, invLocalTransformTo).z;
					}
					const cv::KeyPoint & kpt = _toSignature.getWords().find(wordId)->second;
					ptMap.insert(std::make_pair(2, FeatureBA(kpt, depthTo)));
				}

				wordReferences.insert(std::make_pair(wordId, ptMap));
			}

			std::map<int, Transform> optimizedPoses;
			std::set<int> sbaOutliers;
			optimizedPoses = sba->optimizeBA(1, poses, links, models, points3DMap, wordReferences, &sbaOutliers);
			delete sba;

			// Update BA result
			if (optimizedPoses.size() == 2 && !optimizedPoses.begin()->second.isNull() && !optimizedPoses.rbegin()->second.isNull()) {
				UDEBUG("Pose optimization: %s -> %s", transform.prettyPrint().c_str(), optimizedPoses.rbegin()->second.prettyPrint().c_str());
				if (sbaOutliers.size()) {
					std::vector<int> newInliers(allInliers.size());
					int oi = 0;
					for (std::size_t i = 0; i < allInliers.size(); ++i) {
						if (sbaOutliers.find(allInliers[i]) == sbaOutliers.end())
							newInliers[oi++] = allInliers[i]; 
					}
					newInliers.resize(oi);
					UDEBUG("BA outliers ratio %f", float(sbaOutliers.size())/float(allInliers.size()));
					allInliers = newInliers;				
				}
				if (static_cast<int>(allInliers.size()) < minInliers_) {
					msg = uFormat("Not enough inliers after bundle adjustment %d/%d (matches=%d) between %d and %d",
							(int)allInliers.size(), minInliers_, (int)allInliers.size()+sbaOutliers.size(), _fromSignature.id(), _toSignature.id());
					transform.setNull();			
				} else {
					transform = optimizedPoses.rbegin()->second;
				}

				// Update 3d points in to Signatures.
				std::multimap<int, cv::Point3f> cpyWordsTo3 = _toSignature.getWords3();
				Transform invT = transform.inverse();
				for (std::map<int, cv::Point3f>::iterator iter = points3DMap.begin(); iter != points3DMap.end(); ++iter) {
					if (cpyWordsTo3.find(iter->first) != cpyWordsTo3.end())
						cpyWordsTo3.find(iter->first)->second = util3d::transformPoint(iter->second, invT);
				}
				_toSignature.setWords3(cpyWordsTo3);
			} else {
				transform.setNull();
			}
		}
		//update Feature, update _toSignature.getWords3() _toSignature.getWords()
		_infoOut->inliersIDs = allInliers;
		_infoOut->matchesIDs = allMatches;
	} else if(_toSignature.sensorData().isValid()) {
		msg = uFormat("Missing correspondences for registration (%d->%d). fromWords = %d fromImageEmpty=%d toWords = %d toImageEmpty=%d",
				_fromSignature.id(), _toSignature.id(),
				(int)_fromSignature.getWords().size(), _fromSignature.sensorData().imageRaw().empty()?1:0,
				(int)_toSignature.getWords().size(), _toSignature.sensorData().imageRaw().empty()?1:0);		
	}

	_infoOut->inliers = static_cast<int>(allInliers.size());
	_infoOut->matches = static_cast<int>(allMatches.size());
	_infoOut->matchesInImage = static_cast<int>(kptsTo.size());
	_infoOut->rejectedMsg = msg;
	_infoOut->covariance = covariance;

	// Check keyframe, research new feature, set feature, set words.
	std::vector<cv::Point2f> newCornersInTo;
	int backUpPointsCount = maxFeatures_ - _infoOut->matchesInImage; 
	if (backUpPointsCount > 0 && !_toSignature.sensorData().imageRaw().empty()) {
		// Set mask
		cv::Mat mask = cv::Mat(imageTo.rows, imageTo.cols, CV_8UC1, cv::Scalar(255));
		for (auto kpt : kptsTo) {
			if (mask.at<unsigned char>(kpt.pt) == 255)
				cv::circle(mask, kpt.pt, minDistance_, 0, -1);
		}
		cv::goodFeaturesToTrack(imageTo, newCornersInTo, backUpPointsCount, qualityLevel_, minDistance_, mask);

		std::vector<cv::KeyPoint> newKpt;
		std::vector<cv::Point3f> newkpt3D;
		for (auto kpt2f : newCornersInTo) {
			newKpt.push_back(cv::KeyPoint(kpt2f, 1.f));
		}
		newkpt3D = generateKeyPoints3D(_toSignature.sensorData(), newKpt);

		//Add new feature
		UASSERT(newKpt.size() == newkpt3D.size());
		for (std::size_t i = 0; i < newKpt.size(); ++i) {
			kptsTo.push_back(newKpt[i]);
			kptsTo3D.push_back(newkpt3D[i]);
		}
		std::multimap<int, cv::KeyPoint> newWordsTo;
		std::multimap<int, cv::Point3f> newWords3To;
		featureManager_->addFeature(newKpt, newkpt3D, newWordsTo, newWords3To);
	
		wordsTo.clear();
		words3To.clear();
		wordsTo = _toSignature.getWords();
		words3To = _toSignature.getWords3();
		// Copy newWordsTo, newWords3To to wordsTo and words3To.
		UASSERT(wordsTo.size() == words3To.size());
		for (auto pos = newWordsTo.begin(); pos != newWordsTo.end(); ++pos) {
			wordsTo.insert(*pos);
		}
		for (auto pos = newWords3To.begin(); pos != newWords3To.end(); ++pos) {
			words3To.insert(*pos);
		}
		_toSignature.setWords(wordsTo);
		_toSignature.setWords3(words3To);

	} else {
		newCornersInTo.clear();
	}

	_toSignature.sensorData().setFeatures(kptsTo, kptsTo3D, cv::Mat());

	if (displayTracker_) 
		displayTracker(8, imageToOri, imageFrom, imageTo, cornersFrom, cornersTo, newCornersInTo, status, 0);

	return transform;

}

}
