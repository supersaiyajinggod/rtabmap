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

#include "rtabmap/core/odometry/OdometryVISFS/StateEstimator.h"

namespace rtabmap {
StateEstimator::StateEstimator(const ParametersMap & _parameters) {
    parameters_ = _parameters;
    featureManager_ = new FeatureManager(parameters_);
    featureTracker_ = new FeatureTracker(parameters_, featureManager_);
    initialize();
}

StateEstimator::~StateEstimator() {
    if (featureTracker_ != nullptr)
        delete featureTracker_;
    if (featureManager_ != nullptr)
        delete featureManager_;
}

void StateEstimator::initialize() {
    boost::lock_guard<boost::mutex> lock(mutexProcess_);
    threadEstimateState_ =  boost::thread(&StateEstimator::estimateState, this);
}

void StateEstimator::estimateState() {
    while(1) {
        UINFO("should running here!");
        TrackerInfo trackInfo;
        if (!trackInfoBuf_.empty()) {
            {
                boost::lock_guard<boost::mutex> lock(mutexDataReadWrite_);
                trackInfo = trackInfoBuf_.front();
                trackInfoBuf_.pop();
            }
            trackInfoState_.insert(std::pair<std::size_t, TrackerInfo>(trackInfo.signatureId, trackInfo));
            framePoseInWorld_.insert(std::pair<std::size_t, Transform>(trackInfo.signatureId, trackInfo.globalPose));            

            // Process, checkParallax, triangular map point, update the result to feature manager buf and
            //  remember the feature tracker should modify to get the last frame from the feature manager buf.

            bool keyFrame = featureManager_->checkParallax(trackInfo);

            featureManager_->cleanFeatureAndSignature(keyFrame);
            for (auto it = trackInfoState_.begin(); it != trackInfoState_.end();) {
                std::size_t id = it->first;
                if (!featureManager_->hasSignature(id)) {
                    it = trackInfoState_.erase(it);
                    framePoseInWorld_.erase(framePoseInWorld_.find(id));
                } else {
                    ++it;
                }
            }

            featureManager_->depthRecovery(framePoseInWorld_);
            UINFO("estimateState running!");
        }
        UINFO("estimateThread running!");

        boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(5));
    }
}

void StateEstimator::updateTrackState(const TrackerInfo & _trackInfo) {
    boost::lock_guard<boost::mutex> lock(mutexDataReadWrite_);
    trackInfoBuf_.push(_trackInfo);
}

}