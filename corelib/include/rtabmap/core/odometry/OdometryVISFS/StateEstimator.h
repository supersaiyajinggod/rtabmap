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

#ifndef RTABMAP_STATE_ESTIMATOR_H_
#define RTABMAP_STATE_ESTIMATOR_H_

#include <boost/thread.hpp>

#include "rtabmap/core/odometry/OdometryVISFS/FeatureTracker.h"

namespace rtabmap {

class RTABMAP_EXP StateEstimator {
public:
    StateEstimator(const ParametersMap & _parameters);
    void initialize();
    virtual ~StateEstimator();

	void publishTrackState(const TrackerInfo & _trackInfo);

    void estimateState();

    FeatureManager * featureManager_;
    FeatureTracker * featureTracker_;

private:
    // void updateTrackState

	int optimizationWindowSize_;
    ParametersMap parameters_;

	std::queue<TrackerInfo> trackInfoBuf_;
    std::map<std::size_t, Transform> framePoseInWorld_;     // sigatureId, pose
	std::map<std::size_t, TrackerInfo> trackInfoState_;		// signatureId, trackInfo

    boost::thread threadEstimateState_;
    boost::mutex mutexProcess_;
	boost::mutex mutexDataReadWrite_;

};

}

#endif  /* RTABMAP_STATE_ESTIMATOR_H_ */