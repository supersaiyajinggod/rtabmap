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

#ifndef RTABMAP_FEATURE_TRACKER_H_
#define RTABMAP_FEATURE_TRACKER_H_

#include "rtabmap/core/RtabmapExp.h"
#include "rtabmap/core/Parameters.h"
#include "rtabmap/core/Signature.h"
#include "rtabmap/core/Transform.h"
#include "rtabmap/core/RegistrationInfo.h"

#include <opencv2/opencv.hpp>

namespace rtabmap {

class RTABMAP_EXP TrackerInfo {
public:
    TrackerInfo() :
	totalTime(0.),
	keyFrame(false),
	inliers(0),
	inliersMeanDistance(0.f),
	inliersDistribution(0.f),
	matches(0) {}

	// TrackerInfo copyWithoutData() const {
	// 	TrackerInfo output;
	// 	output.totalTime = totalTime;
	// 	output.covariance = covariance.clone();
	// 	output.rejectedMsg = rejectedMsg;
	// 	output.keyFrame = keyFrame;
	// 	output.inliers = inliers;
	// 	output.inliersMeanDistance = inliersMeanDistance;
	// 	output.inliersDistribution = inliersDistribution;
	// 	output.matches = matches;
	// 	return output;
	// }

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
	bool keyFrame;
	int inliers;
	float inliersMeanDistance;
	float inliersDistribution;
	std::vector<int> inliersIDs;
	int matches;
	std::vector<int> matchesIDs;
	std::vector<int> projectedIDs;
};

class RTABMAP_EXP FeatureTracker {

public:
    FeatureTracker(const ParametersMap & _parameters);
    virtual ~FeatureTracker();

	Transform computeTransformation(const Signature & _fromSignature, const Signature & _toSignature, Transform _guess = Transform::getIdentity(), TrackerInfo * _info = nullptr) const;
  	Transform computeTransformation(const SensorData & _fromSignature, const SensorData & _toSignature, Transform _guess = Transform::getIdentity(), TrackerInfo * _info = nullptr) const;
    Transform computeTransformationMod(Signature & _fromSignature, Signature & _toSignature, Transform _guess = Transform::getIdentity(), TrackerInfo * _info = nullptr) const;

private:
	std::vector<cv::Point3f> generateKeyPoints3D(const SensorData & _data, const std::vector<cv::KeyPoint> & _keyPoints) const;
	inline float distanceL2(const cv::Point2f & pt1, const cv::Point2f & pt2) const;
	void displayTracker(int _n, ...) const;

private:
	const double COVARIANCE_EPSILON = 0.000000001;

	ParametersMap parameters_;
	bool force3DoF_;
	bool displayTracker_;
	int maxFeatures_;
	double qualityLevel_;
	int minDistance_;
	bool flowBack_;
	float minParallax_;
	float maxDepth_;
	float minDepth_;
	int flowWinSize_;
	int flowIterations_;
	float flowEps_;
	int flowMaxLevel_;
	int minInliers_;
	int pnpIterations_;
	float pnpReprojError_;
	int pnpFlags_;
	int refineIterations_;
	int bundleAdjustment_;
	ParametersMap bundleParameters_;

};

}

#endif /* RTABMAP_FEATURE_TRACKER_H_ */