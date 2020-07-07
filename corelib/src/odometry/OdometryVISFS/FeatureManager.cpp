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

#include "rtabmap/core/odometry/OdometryVISFS/FeatureManager.h"

#include <Eigen/Eigenvalues>

namespace rtabmap {

FeatureManager::FeatureManager(const ParametersMap & _parameters) : 
	optimizationWindowSize_(Parameters::defaultOdomVISFSOptimizationWindowSize()),
	maxFeature_(Parameters::defaultOdomVISFSMaxFeatures()),
	minParallax_(Parameters::defaultOdomVISFSMinParallax()),
	featureId_(0),
	signatureId_(0) {
	Parameters::parse(_parameters, Parameters::kOdomVISFSOptimizationWindowSize(), optimizationWindowSize_);
	Parameters::parse(_parameters, Parameters::kOdomVISFSMaxFeatures(), maxFeature_);
	Parameters::parse(_parameters, Parameters::kOdomVISFSMinParallax(), minParallax_);
    parameters_ = _parameters;
}

std::size_t FeatureManager::getSignatureId() {
	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	return signatureId_;
}

// Caution, after call this function, the member variable, signatureId_, of this class have been increased. 
std::size_t FeatureManager::addSignatrue(const Signature & _signature) {
    boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
    signatures_.push_back(_signature);
	++signatureId_;
	return signatureId_ - 1;
}

std::vector<std::size_t> FeatureManager::addFeature(const std::vector<cv::KeyPoint> & _kpt, const std::vector<cv::Point3f> & _kpt3d, std::multimap<int, cv::KeyPoint> & _words, std::multimap<int, cv::Point3f> & _words3d) {
    UASSERT(_kpt.size() == _kpt3d.size());
    if (!_words.empty() && !_words3d.empty()) {
		_words.clear();
		_words3d.clear();
    }
	std::vector<std::size_t> featureIndex;

    boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
    for (std::size_t i = 0; i < _kpt.size(); ++i) {
        Featrue newFeture(featureId_, signatureId_, 0);
        std::isfinite(_kpt3d[i].z) ? newFeture.setSolveState(Featrue::FROM_DEPTH) : newFeture.setSolveState(Featrue::NOT_SOLVE);
        newFeture.featrueStatusInFrames_.insert(std::pair<std::size_t, FeatureStatusOfEachFrame>(signatureId_, FeatureStatusOfEachFrame(_kpt[i].pt, _kpt3d[i])));
        featrues_.push_back(newFeture);
		featureIndex.push_back(featureId_);
		_words.insert(std::pair<int, cv::KeyPoint>(static_cast<int>(featureId_), _kpt[i]));
		_words3d.insert(std::pair<int, cv::Point3f>(static_cast<int>(featureId_), _kpt3d[i]));
        ++featureId_;
    }

	return featureIndex;
}

std::vector<std::size_t> FeatureManager::updateFeature(const std::multimap<int, cv::KeyPoint> & _words, const std::multimap<int, cv::Point3f> & _words3d) {
	UASSERT(_words.size() == _words3d.size());
	std::vector<std::size_t> updatedFeatureIndex;

	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	for (auto word : _words) {
		auto it1 = _words3d.find(word.first);
		auto it2 = std::find_if(featrues_.begin(), featrues_.end(), [word] (Featrue & feature) {
			return word.first == static_cast<int>(feature.getId());
		});
		
		if (it1 != _words3d.end() && it2 == featrues_.end()) {	// Add new feature
			UINFO("Error? There is a feature not in the list.");
		} else if (word.first == it1->first && word.first == static_cast<int>(it2->getId())) {	// Update
			it2->featrueStatusInFrames_.insert(std::pair<std::size_t, FeatureStatusOfEachFrame>(signatureId_, FeatureStatusOfEachFrame(word.second.pt, it1->second)));
			updatedFeatureIndex.push_back(word.first);
			if (it2->featrueStatusInFrames_.size() > static_cast<std::size_t>(optimizationWindowSize_)) {
				it2->featrueStatusInFrames_.erase(it2->featrueStatusInFrames_.begin());
			}
		}
	}

	return updatedFeatureIndex;
}

bool FeatureManager::checkParallax(const TrackerInfo & _trackInfo) {
	if (signatures_.size() < 0.2 * optimizationWindowSize_ || static_cast<int>(featrues_.size()) < maxFeature_) { return true; }
	int backUpPointsCount = maxFeature_ - _trackInfo.matchesInImage;
	if (_trackInfo.inliers < (0.2 * maxFeature_) || backUpPointsCount > (0.5 * _trackInfo.inliers)) {
		UDEBUG("KeyFrame condition satisfied!, inliers = %d, maxFeatures_ = %d, 0.1*maxFeatures_ = %f, backUpPointsCount = %d.", _trackInfo.inliers, maxFeature_, 0.1 * maxFeature_, backUpPointsCount);
		return true;
	} else {	// Compute parallax
		std::multimap<int, cv::KeyPoint> wordsFrom;
		std::multimap<int, cv::KeyPoint> wordsTo;
		{
			const int latestSignatureId = static_cast<int>(_trackInfo.signatureId);
			const int secondLatestSignatureId = static_cast<int>(latestSignatureId - 1);
			int breakFlag = 0;
			boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
			for (std::list<Signature>::reverse_iterator rIt = signatures_.rbegin(); rIt != signatures_.rend(); ++rIt) {
				if (rIt->id() == latestSignatureId) {
					wordsTo = rIt->getWords();
					++breakFlag;
				} else if (rIt->id() == secondLatestSignatureId) {
					wordsFrom = rIt->getWords();
					++breakFlag;
				}
				if (breakFlag == 2)
					break;
			}
		}
		const auto matchesId = _trackInfo.matchesInImageIDs;
		const int parallaxNum = static_cast<int>(matchesId.size());
		float parallaxSum = 0.f;
		for (auto featureId : matchesId) {
			auto itFrom = wordsFrom.find(featureId);
			auto itTo = wordsTo.find(featureId);
			if (itFrom != wordsFrom.end() && itTo != wordsTo.end()) {
				const float du = itFrom->second.pt.x - itTo->second.pt.x;
				const float dv = itFrom->second.pt.y - itTo->second.pt.y;
				parallaxSum += std::max(0.f, sqrt(du * du + dv * dv));
			}
		}
		if ((parallaxSum / static_cast<float>(parallaxNum)) >= minParallax_) {
			UDEBUG("Keyframe condition satisfied!. parallax!, compute result=%f, minParallax_=%f.", (parallaxSum / static_cast<float>(parallaxNum)),minParallax_);
			return true;
		}
	}

	return false;
}

void FeatureManager::cleanFeatureAndSignature(bool _keyFrame) {
	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	// Remove signature.
	if (_keyFrame && (signatures_.size() > static_cast<std::size_t>(optimizationWindowSize_))) {
		signatures_.pop_front();
	} else {
		auto it = signatures_.end();
		--it; --it;
		signatures_.erase(it);
	}
	// Remove features.
	const size_t latestSignatureId = static_cast<std::size_t>(signatures_.rbegin()->id());
	
	for (auto it = featrues_.begin(); it != featrues_.end();) {
		auto stateInFrames = it->featrueStatusInFrames_.end();
		--stateInFrames;
		if (stateInFrames->first < latestSignatureId - static_cast<std::size_t>(optimizationWindowSize_)) {
			it = featrues_.erase(it);
		}
		++it;
	}
}

void FeatureManager::depthRecovery(const std::map<std::size_t, Transform> & _framePoseInWorld) {
	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	const auto lastSignatureIt = signatures_.rbegin();
	std::multimap<int, cv::Point3f> word3d = lastSignatureIt->getWords3();
	for (auto feature : featrues_) {
		if ((feature.getSolveState() == Featrue::NOT_SOLVE || feature.getSolveState() == Featrue::FROM_CALCULATE) && (feature.getTrackedCnt() > 1)) {
			auto latestIt = feature.featrueStatusInFrames_.rbegin();
			auto tmp = latestIt;
			auto secondLatestIt = ++tmp; 
			const std::size_t latestSignatureId = static_cast<std::size_t>(latestIt->first);
			const std::size_t secondLatestsignatureId = static_cast<std::size_t>(secondLatestIt->first);
			const auto lastPoseIt = _framePoseInWorld.find(latestSignatureId);
			const auto secondLatestPoseIt = _framePoseInWorld.find(secondLatestsignatureId);
			if (lastPoseIt != _framePoseInWorld.end() && secondLatestPoseIt != _framePoseInWorld.end()) {
				cv::Point3f pointInWorld;
				cv::Point3f pointInSecondLatest;
				cv::Point3f pointInLatest;
				triangulateByTwoFrames(secondLatestPoseIt->second, lastPoseIt->second, secondLatestIt->second.uv_, secondLatestIt->second.uv_,
										pointInWorld, pointInSecondLatest, pointInLatest);
				feature.setEstimatePose(pointInWorld);
				// Update features_
				secondLatestIt->second.point3d_ = pointInSecondLatest;
				latestIt->second.point3d_ = pointInLatest;
				// Update sigantures_
				if (latestSignatureId == static_cast<std::size_t>(lastSignatureIt->id())) {
					auto wordIt = word3d.find(static_cast<int>(feature.getId()));
					if (wordIt != word3d.end())
						wordIt->second = pointInLatest;
				}
				feature.setSolveState(Featrue::FROM_CALCULATE);
			}
		}

		if ((feature.getSolveState() == Featrue::FROM_CALCULATE) && (feature.getTrackedCnt() > 4)) {
			Eigen::MatrixXf svdA(static_cast<int>(2*feature.getTrackedCnt()), 4);
			int avaliableCnt = 0;
			int svdIndex = 0;
			for (auto it = feature.featrueStatusInFrames_.begin(); it != feature.featrueStatusInFrames_.end(); ++it) {
				auto findPose = _framePoseInWorld.find(it->first);
				if (findPose != _framePoseInWorld.end()) {
					const Eigen::Matrix4f pose = findPose->second.toEigen4f();
					Eigen::Matrix<float, 3, 4> Tcw;
					Tcw.leftCols<3>() = pose.topLeftCorner(3, 3).transpose();
					Tcw.rightCols<1>() = -Tcw.leftCols<3>()*pose.topRightCorner(3, 1);
					
					Eigen::Vector3f f = Eigen::Vector3f(it->second.uv_.x, it->second.uv_.y, 1).normalized();
					svdA.row(svdIndex++) = f[0] * Tcw.row(2) - f[2] * Tcw.row(0);
					svdA.row(svdIndex++) = f[1] * Tcw.row(2) - f[2] * Tcw.row(1);
					++avaliableCnt;
				}
			}
			Eigen::MatrixXf svdAavalible(2*avaliableCnt, 4);
			svdAavalible = svdA.topRows(2*avaliableCnt);
			Eigen::Vector4f svdV = Eigen::JacobiSVD<Eigen::MatrixXf>(svdAavalible, Eigen::ComputeThinV).matrixV().rightCols<1>();
			Eigen::Vector3f pointInWorld = Eigen::Vector3f(svdV[0]/svdV[3], svdV[1]/svdV[3], svdV[2]/svdV[3]);
			// Update signatures_ and features_
			auto latestIt = feature.featrueStatusInFrames_.rbegin();
			const std::size_t latestSignatureId = static_cast<std::size_t>(latestIt->first);
			const auto lastPoseIt = _framePoseInWorld.find(latestSignatureId);
			if (lastPoseIt != _framePoseInWorld.end()) {
				Eigen::Matrix<float, 3, 4> Tcw;
				Eigen::Matrix4f pose = lastPoseIt->second.toEigen4f();
				Tcw.leftCols<3>() = pose.topLeftCorner(3, 3).transpose();
				Tcw.rightCols<1>() = -Tcw.leftCols<3>()*pose.topRightCorner(3, 1);
				Eigen::Vector3f pointInCamera = Tcw.leftCols<3>() * pointInWorld + Tcw.rightCols<1>();
				auto signatureId = feature.featrueStatusInFrames_.rbegin()->first;
				auto lastSignatureIt = signatures_.rbegin();
				if (signatureId == static_cast<std::size_t>(lastSignatureIt->id())) {
					auto wordIt = word3d.find(static_cast<int>(feature.getId()));
					if (wordIt != word3d.end())
						wordIt->second = cv::Point3f(pointInCamera[0], pointInCamera[1], pointInCamera[2]);
				}
			
				latestIt->second.point3d_ = cv::Point3f(pointInCamera[0], pointInCamera[1], pointInCamera[2]);
				feature.setEstimatePose(cv::Point3f(pointInWorld[0], pointInWorld[1], pointInWorld[2]));
				feature.setSolveState(Featrue::OPTIMIZED);
			}
		}

		if ((feature.getSolveState() == Featrue::OPTIMIZED) && !std::isfinite(feature.featrueStatusInFrames_.rbegin()->second.point3d_.z)) {
			auto latestIt = feature.featrueStatusInFrames_.rbegin();
			const std::size_t latestSignatureId = static_cast<std::size_t>(latestIt->first);
			const auto lastPoseIt = _framePoseInWorld.find(latestSignatureId);
			cv::Point3f pointInWorld = feature.getEstimatedPose();
			Eigen::Vector3f eigenPointInWorld(pointInWorld.x, pointInWorld.y, pointInWorld.z);
			if (lastPoseIt != _framePoseInWorld.end()) {
				Eigen::Matrix<float, 3, 4> Tcw;
				Eigen::Matrix4f pose = lastPoseIt->second.toEigen4f();
				Tcw.leftCols<3>() = pose.topLeftCorner(3, 3).transpose();
				Tcw.rightCols<1>() = -Tcw.leftCols<3>()*pose.topRightCorner(3, 1);
				Eigen::Vector3f pointInCamera = Tcw.leftCols<3>() * eigenPointInWorld + Tcw.rightCols<1>();
				
				auto lastSignatureIt = signatures_.rbegin();
				if (latestSignatureId == static_cast<std::size_t>(lastSignatureIt->id())) {
					auto wordIt = word3d.find(static_cast<int>(feature.getId()));
					if (wordIt != word3d.end())
						wordIt->second = cv::Point3f(pointInCamera[0], pointInCamera[1], pointInCamera[2]);					
				}
				latestIt->second.point3d_ = cv::Point3f(pointInCamera[0], pointInCamera[1], pointInCamera[2]);
			}

		}
	}
	lastSignatureIt->setWords3(word3d);
}

bool FeatureManager::hasSignature(const std::size_t _id) {
	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	auto it = std::find_if(signatures_.begin(), signatures_.end(), [_id] (Signature & signature) {
		return signature.id() == static_cast<int>(_id);
	});
	if (it != signatures_.end()) {
		return true;
	} else {
		return false;
	}
}

bool FeatureManager::hasFeature(const std::size_t _id) {
	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	auto it = std::find_if(featrues_.begin(), featrues_.end(), [_id] (Featrue & feature) {
		return feature.getId() == _id;
	});
	if (it != featrues_.end()) {
		return true;
	} else {
		return false;
	}
}

bool FeatureManager::triangulateByTwoFrames(const Transform & _pose0, const Transform & _pose1, const cv::Point2f & _uv0, const cv::Point2f & _uv1,
												cv::Point3f & _pointInWord, cv::Point3f & _pointInPose0, cv::Point3f & _pointInPose1) {
	const Eigen::Matrix4f pose0 = _pose0.toEigen4f();
	const Eigen::Matrix4f pose1 = _pose1.toEigen4f();
	const Eigen::Vector2f uv0 = Eigen::Vector2f(_uv0.x, _uv0.y);
	const Eigen::Vector2f uv1 = Eigen::Vector2f(_uv1.x, _uv1.y);

	Eigen::Matrix<float, 3, 4> Tcw0;
	Tcw0.leftCols<3>() = pose0.topLeftCorner(3, 3).transpose();
	Tcw0.rightCols<1>() = -Tcw0.leftCols<3>()*pose0.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 4> Tcw1;
	Tcw1.leftCols<3>() = pose1.topLeftCorner(3, 3).transpose();
	Tcw1.rightCols<1>() = -Tcw1.leftCols<3>()*pose1.topRightCorner(3, 1);

    Eigen::Matrix4f designMatrix = Eigen::Matrix4f::Zero();
    designMatrix.row(0) = uv0[0] * Tcw0.row(2) - Tcw0.row(0);
    designMatrix.row(1) = uv0[1] * Tcw0.row(2) - Tcw0.row(1);
    designMatrix.row(2) = uv1[0] * Tcw1.row(2) - Tcw1.row(0);
    designMatrix.row(3) = uv1[1] * Tcw1.row(2) - Tcw1.row(1);
    Eigen::Vector4f triangulatedPoint;
    triangulatedPoint = designMatrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

	Eigen::Vector3f pointInWord(triangulatedPoint(0) / triangulatedPoint(3), triangulatedPoint(1) / triangulatedPoint(3), triangulatedPoint(2) / triangulatedPoint(3));	
	Eigen::Vector3f pointInPose0 = Tcw0.leftCols<3>() * pointInWord + Tcw0.rightCols<1>();
	Eigen::Vector3f pointInPose1 = Tcw1.leftCols<3>() * pointInWord + Tcw1.rightCols<1>();

	_pointInWord = cv::Point3f(pointInWord[0], pointInWord[1], pointInWord[2]);
	_pointInPose0 = cv::Point3f(pointInPose0[0], pointInPose0[1], pointInPose0[2]);
	_pointInPose1 = cv::Point3f(pointInPose1[0], pointInPose1[1], pointInPose1[2]);

	return true;
}

Signature FeatureManager::getLastSignature() {
	boost::lock_guard<boost::mutex> lock(mutexFeatureProcess_);
	return * signatures_.rbegin();
}

}
