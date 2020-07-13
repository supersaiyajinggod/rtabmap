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

#ifndef RTABMAP_FEATURE_MANAGER_H_
#define RTABMAP_FEATURE_MANAFER_H_

#include "rtabmap/core/RtabmapExp.h"
#include "rtabmap/core/Parameters.h"
#include "rtabmap/core/Signature.h"
#include "rtabmap/core/Transform.h"
#include "rtabmap/core/RegistrationInfo.h"

#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

namespace rtabmap {

class Featrue;
class FeatureStatusOfEachFrame;

class RTABMAP_EXP TrackerInfo {
public:
    TrackerInfo() :
	totalTime(0.),
	inliers(0),
	inliersMeanDistance(0.f),
	inliersDistribution(0.f),
	matches(0),
    matchesInImage(0),
	signatureId(0) {
		deltaT = Transform::getIdentity();
		globalPose = Transform::getIdentity();
	}

	RegistrationInfo copyWithoutData() const {
		RegistrationInfo output;
		output.totalTime = totalTime;
		output.covariance = covariance.clone();
		output.rejectedMsg = rejectedMsg;
		output.inliers = inliers;
		output.inliersMeanDistance = inliersMeanDistance;
		output.inliersDistribution = inliersDistribution;
		output.matches = matches;
		output.icpInliersRatio = 0.f;
		output.icpTranslation = 0.f;
		output.icpRotation = 0.f;
		output.icpStructuralComplexity = 0.f;
		output.icpStructuralDistribution = 0.f;
		output.icpCorrespondences = 0;
		return output;
	}

	cv::Mat covariance;
	std::string rejectedMsg;
	double totalTime;
	int inliers;
	float inliersMeanDistance;
	float inliersDistribution;
	std::vector<int> inliersIDs;
	int matches;
    int matchesInImage;
	std::size_t signatureId;
	Transform deltaT;
	Transform globalPose;
	std::vector<int> matchesIDs;
    std::vector<int> matchesInImageIDs;
	std::vector<int> projectedIDs;
    Signature signature;
    std::queue<Featrue> newFeatures;
    std::queue<std::pair<std::multimap<int, cv::KeyPoint>, std::multimap<int, cv::Point3f>>> featuresState;
};

class RTABMAP_EXP FeatureStatusOfEachFrame {
public:
    FeatureStatusOfEachFrame(const cv::Point2f & _uv, const cv::Point3f & _point3d) {
        uv_ = _uv;
        point3d_ = _point3d;
        velocity_ = cv::Point2f(-1.f, -1.f);
    };

    FeatureStatusOfEachFrame(const cv::Point2f & _uv, const cv::Point3f & _point3d, const cv::Point2f & _velocity) {
        uv_ = _uv;
        point3d_ = _point3d;
        velocity_ = _velocity;
    };

    void rightObservation(const cv::Point2f & _uv, const cv::Point3f & _point3d) {
        uvRight_ = _uv;
        point3dRight_ = _point3d;
        velocityRight_ = cv::Point2f(-1.f, -1.f);
        isStereo = true;
    }

    void rightObservation(const cv::Point2f & _uv, const cv::Point3f & _point3d, const cv::Point2f & _velocity) {
        uvRight_ = _uv;
        point3dRight_ = _point3d;
        velocityRight_ = _velocity;
        isStereo = true;
    }

    cv::Point2f uv_, uvRight_;
    cv::Point3f point3d_, point3dRight_;
    cv::Point2f velocity_, velocityRight_;
    bool isStereo;
};

class RTABMAP_EXP Featrue {
public:
    Featrue(std::size_t _featrueId, std::size_t _startFrameId, int _solveState) : featrueId_(_featrueId), startFrameId_(_startFrameId), estimatedDepth_(-1.0), solveState_(_solveState) {}
    
    enum eSolveState {
        NOT_SOLVE       = 0,
        FROM_DEPTH      = 1,
        FROM_CALCULATE  = 2,
        OPTIMIZED       = 3
    };

    std::size_t getTrackedCnt() {return featrueStatusInFrames_.size();}
    std::size_t getId() {return featrueId_;}
    std::size_t getStartFrame() {return startFrameId_;}
    double getEstimatedDepth() {return estimatedDepth_;}
    cv::Point3f getEstimatedPose() {return PoseTwc_;}
    void setEstimatePose(const cv::Point3f & _pose) {PoseTwc_ = _pose;}
    int getSolveState() {return solveState_;}
    void setSolveState(const enum eSolveState & _solveState) {solveState_ = _solveState;}

    std::map<std::size_t, FeatureStatusOfEachFrame> featrueStatusInFrames_;     // <signatrueId, FeatureStatusOfEachFrame> 

private:
    std::size_t featrueId_;
    std::size_t startFrameId_;
    double estimatedDepth_;
    cv::Point3f PoseTwc_;
    int solveState_;    // 0 haven't solve, 1 get depth from depth image, 2 calculate depth with former frame, 3 optical depth with several frames.
};

class RTABMAP_EXP FeatureManager {
public:
    FeatureManager(const ParametersMap & _parameters);
    std::size_t getSignatureId();
    Signature getLastSignature();
    std::size_t publishSignatrue(const Signature & _signature, TrackerInfo & _trackInfo);
    std::vector<std::size_t> publishFeatures(const std::vector<cv::KeyPoint> & _kpt, const std::vector<cv::Point3f> & _kpt3d, std::multimap<int, cv::KeyPoint> & _words, std::multimap<int, cv::Point3f> & _words3d, TrackerInfo * _trackInfo);
    std::vector<std::size_t> publishFeatureStates(const std::multimap<int, cv::KeyPoint> & _words, const std::multimap<int, cv::Point3f> & _words3d, TrackerInfo * _trackInfo);
    void process(const TrackerInfo & _trackInfo, std::map<std::size_t, Transform> & _framePoseInWorld);

    std::queue<Signature> signaturesBuf_;
    std::queue<Featrue> featuresBuf_;
    std::queue<std::pair<std::multimap<int, cv::KeyPoint>, std::multimap<int, cv::Point3f>>> featureStateBuf_;

private:
    bool hasSignature(const std::size_t _id);
    bool hasFeature(const std::size_t _id);
    bool checkParallax(const TrackerInfo & _trackInfo); //True: keyframe, False: Not keyframe.
    void cleanFeatureAndSignature(bool _keyFrame);
    void depthRecovery(const std::map<std::size_t, Transform> & _framePoseInWorld);
    void update(const TrackerInfo & _trackInfo);
    bool triangulateByTwoFrames(const Transform & _pose0, const Transform & _pose1, const cv::Point2f & _uv0, const cv::Point2f & _uv1, cv::Point3f & _pointInWord, cv::Point3f & _pointInPose0, cv::Point3f & _pointInPose1) const;

    int optimizationWindowSize_;
    int maxFeature_;
    float minParallax_;
    ParametersMap parameters_;
    boost::mutex mutexFeatureProcess_;
    // boost::mutex mutexDataReadWrite_;

    std::list<Featrue> features_;
    std::list<Signature> signatures_;

    std::size_t featureId_;
    std::size_t signatureId_;
   
};

}

#endif  /* RTABMAP_FEATURE_MANAGER_H_ */