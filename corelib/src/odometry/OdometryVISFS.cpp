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

#include "rtabmap/core/odometry/OdometryVISFS.h"
#include "rtabmap/core/OdometryInfo.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/utilite/UTimer.h"

namespace rtabmap {

OdometryVISFS::OdometryVISFS(const ParametersMap & _parameters) : Odometry(_parameters) {
     parameters_ = _parameters;
     featureTracker_ = new FeatureTracker(_parameters);
}

OdometryVISFS::~OdometryVISFS() {
     delete featureTracker_;
}

void OdometryVISFS::reset(const Transform & _initialPose) {
    Odometry::reset(_initialPose);
}

Transform OdometryVISFS::computeTransform(SensorData & _data, const Transform & _guess, OdometryInfo * _info) {
	UTimer timer;
	Transform output;
	if(!_data.rightRaw().empty() && !_data.stereoCameraModel().isValidForProjection()) {
		UERROR("Calibrated stereo camera required");
		return output;
	}
	if(!_data.depthRaw().empty() &&
		(_data.cameraModels().size() != 1 || !_data.cameraModels()[0].isValidForProjection())) {
		UERROR("Calibrated camera required (multi-cameras not supported).");
		return output;
	}

     TrackerInfo trackInfo;

     UASSERT(!this->getPose().isNull());
     if (lastFramePose_.isNull()) {
          lastFramePose_ = this->getPose();  // reset to current pose
     }

     Transform motionSinceLastFrame = lastFramePose_.inverse()*this->getPose();

     Signature newFrame(_data);
     if (lastFrame_.sensorData().isValid()) {
          // TODO: If set use icp for fake laser scan, do some prepare works.
          // OdometryVISFS
          Signature tmpRefFrame = lastFrame_;
          output = featureTracker_->computeTransformationMod(tmpRefFrame, newFrame, _guess.isNull()?motionSinceLastFrame*_guess:Transform(), &trackInfo);
     
          // If OdometryVISFS failed, use F2M strategy.

          if (output.isNull()) 
               UWARN("Trial failed.");

          if (_info) {
               Transform t = this->getPose()*motionSinceLastFrame.inverse();
               for(std::multimap<int, cv::Point3f>::const_iterator iter=tmpRefFrame.getWords3().begin(); iter!=tmpRefFrame.getWords3().end(); ++iter) {
                    _info->localMap.insert(std::make_pair(iter->first, util3d::transformPoint(iter->second, t)));
               }
               _info->localMapSize = tmpRefFrame.getWords3().size();
               _info->words = newFrame.getWords();
               _info->localScanMapSize = tmpRefFrame.sensorData().laserScanRaw().size();
               _info->localScanMap = util3d::transformLaserScan(tmpRefFrame.sensorData().laserScanRaw(), tmpRefFrame.sensorData().laserScanRaw().localTransform().inverse()*t*tmpRefFrame.sensorData().laserScanRaw().localTransform());
          }

     } else {
          //return Identity
          output = Transform::getIdentity();
          // a very high variance tells that the new pose is not linked with the previous one
          trackInfo.covariance = cv::Mat::eye(6,6,CV_64FC1)*9999.0;
     }

     if (!output.isNull()) {   // Set words and reference frame.
          output = motionSinceLastFrame.inverse() * output;
          if (trackInfo.keyFrame) {  // Keyframe check
               lastFramePose_.setNull();
          } 
     } else if (!trackInfo.rejectedMsg.empty()) {
          UWARN("Registration failed: \"%s\"", trackInfo.rejectedMsg.c_str());
     }
     lastFrame_ = newFrame;
     _data.setFeatures(newFrame.sensorData().keypoints(), newFrame.sensorData().keypoints3D(), newFrame.sensorData().descriptors());

     if (_info) {
          _info->type = kTypeVISFS;
          output.isNull() ? _info->lost = true : _info->lost = false; 
          _info->features = newFrame.sensorData().keypoints().size();
          _info->keyFrameAdded = trackInfo.keyFrame;
          _info->reg = trackInfo.copyWithoutData();
     }

	UINFO("Odom update time = %fs lost=%s inliers=%d, ref frame corners=%d, transform accepted=%s",
		timer.elapsed(), output.isNull() ? "true" : "false", static_cast<int>(trackInfo.inliers),
		static_cast<int>(newFrame.sensorData().keypoints().size()), !output.isNull() ? "true" : "false");

	return output;
}


}