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

#include "rtabmap/core/featureTracker/FeatureTracker.h"
#include "rtabmap/core/util3d_features.h"
#include "rtabmap/core/util3d_motion_estimation.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/Optimizer.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/utilite/UStl.h"

namespace rtabmap {

FeatureTracker::FeatureTracker(const ParametersMap & _parameters) : 
	force3DoF_(Parameters::defaultRegForce3DoF()),
	maxFeatures_(Parameters::defaultOdomVISFSMaxFeatures()),
	qualityLevel_(Parameters::defaultGFTTQualityLevel()),
	minDistance_(Parameters::defaultGFTTMinDistance()),
	flowBack_(Parameters::defaultOdomVISFSFlowBack()),
	minParallax_(Parameters::defaultOdomVISFSMinParallax()),
	maxDepth_(Parameters::defaultOdomVISFSMaxDepth()),
	minDepth_(Parameters::defaultOdomVISFSMinDepth()),
	flowWinSize_(Parameters::defaultOdomVISFSFlowWinSize()),
	flowIterations_(Parameters::defaultOdomVISFSFlowIterations()),
	flowEps_(Parameters::defaultOdomVISFSFlowEps()),
	flowMaxLevel_(Parameters::defaultOdomVISFSFlowMaxLevel()),
	minInliers_(Parameters::defaultOdomVISFSMinInliers()),
	pnpIterations_(Parameters::defaultOdomVISFSPnPIterations()),
	pnpReprojError_(Parameters::defaultOdomVISFSPnPReprojError()),
	pnpFlags_(Parameters::defaultOdomVISFSPnPFlags()),
	refineIterations_(Parameters::defaultOdomVISFSRefineIterations()),
	bundleAdjustment_(Parameters::defaultOdomVISFSBundleAdjustment()) {
	parameters_ = _parameters;

	Parameters::parse(parameters_, Parameters::kRegForce3DoF(), force3DoF_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMaxFeatures(), maxFeatures_);
	Parameters::parse(parameters_, Parameters::kGFTTQualityLevel(), qualityLevel_);
	Parameters::parse(parameters_, Parameters::kGFTTMinDistance(), minDistance_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowBack(), flowBack_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMinParallax(), minParallax_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMaxDepth(), maxDepth_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSMinDepth(), minDepth_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowWinSize(), flowWinSize_);
	Parameters::parse(parameters_, Parameters::kOdomVISFSFlowMaxLevel(), flowMaxLevel_);
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

std::vector<cv::Point3f> FeatureTracker::generateKeyPoints3D(const SensorData & _data, const std::vector<cv::KeyPoint> & _keyPoints) const {
	std::vector<cv::Point3f> keyPoints3D;
	if (_keyPoints.size()) {
		if (!_data.depthRaw().empty() && _data.cameraModels().size()) {
			keyPoints3D = util3d::generateKeypoints3DDepth(_keyPoints, _data.depthRaw(), _data.cameraModels(), minDepth_, maxDepth_);
		}
	}
	return keyPoints3D;
}

inline float FeatureTracker::distanceL2(const cv::Point2f & pt1, const cv::Point2f & pt2) const {
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
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
	} else {		// Process the former extracted keypoints.
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
			UDEBUG("IDs are not unique, IDs will be regenerated!");
			orignalWordsFromIds.clear();			
		}
		UDEBUG("Get key points from the former signature's words2d: %d.", static_cast<int>(kptsFrom.size()));
	}

	std::multimap<int, cv::KeyPoint> wordsFrom;
	std::multimap<int, cv::KeyPoint> wordsTo;
	std::multimap<int, cv::Point3f> words3From;
	std::multimap<int, cv::Point3f> words3To;
	std::multimap<int, cv::Mat> wordsDescFrom;
	std::multimap<int, cv::Mat> wordsDescTo;

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
			if (status[i] && reverseStatus[i] && distanceL2(cornersReverse[i], cornersFrom[i]) <= 0.5) {
				status[i] = 1;
			} else {
				status[i] = 0;
			}
		}
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
	for (size_t i = 0; i < kptsFrom3DKept.size(); ++i) {
		int id = orignalWordsFromIds.size()?orignalWordsFromIds[i] : static_cast<int>(i);
		wordsFrom.insert(std::pair<int, cv::KeyPoint>(id, kptsFrom[i]));
		words3From.insert(std::pair<int, cv::Point3f>(id, kptsFrom3DKept[i]));
		wordsTo.insert(std::pair<int, cv::KeyPoint>(id, kptsTo[i]));
		words3To.insert(std::pair<int, cv::Point3f>(id, kptsTo3D[i]));
	}
	_fromSignature.setWords(wordsFrom);
	_fromSignature.setWords3(words3From);
	_toSignature.setWords(wordsTo);
	_toSignature.setWords3(words3To);

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
	_infoOut->rejectedMsg = msg;
	_infoOut->covariance = covariance;

	// Check keyframe, research new feature, set feature, set words.
	std::vector<cv::Point2f> newCornersInTo;
	int backUpPointsCount = maxFeatures_ - static_cast<int>(kptsTo.size()); 

	if (allInliers.size() < (0.1 * maxFeatures_) || backUpPointsCount > 0.5 * allInliers.size())
		_infoOut->keyFrame = true;
	// Compute parallax
	int parallaxNum = static_cast<int>(kptsFrom.size());
	float parallaxSum = 0.f;
	for (std::size_t i = 0; i < kptsFrom.size(); ++i) {
		const float du = kptsFrom[i].pt.x - kptsTo[i].pt.y;
		const float dv = kptsFrom[i].pt.y - kptsTo[i].pt.y;
		parallaxSum += std::max(0.f, sqrt(du * du + dv * dv));
	}
	if (parallaxSum / static_cast<float>(parallaxNum) >= minParallax_)
		_infoOut->keyFrame = true;

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

		UASSERT(newKpt.size() == newkpt3D.size());
		for (std::size_t i = 0; i < newKpt.size(); ++i) {
			kptsTo.push_back(newKpt[i]);
			kptsTo3D.push_back(newkpt3D[i]);
		}

		UASSERT(kptsTo3D.size() == 0 || kptsTo.size() == kptsTo3D.size());
		wordsTo.clear();
		words3To.clear();
		for (std::size_t i = 0; i < kptsTo.size(); ++i) {
			int id = static_cast<int>(i);
			wordsTo.insert(std::pair<int, cv::KeyPoint>(id, kptsTo[i]));
			words3To.insert(std::pair<int, cv::Point3f>(id, kptsTo3D[i]));
		}
		_toSignature.setWords(wordsTo);
		_toSignature.setWords3(words3To);

	} else {
		newCornersInTo.clear();
	}

	_toSignature.sensorData().setFeatures(kptsTo, kptsTo3D, cv::Mat());

	UDEBUG("inliers=%d/%d", _infoOut->inliers, _infoOut->matches);
	UDEBUG("transform=%s", transform.prettyPrint().c_str());
	return transform;

}

}