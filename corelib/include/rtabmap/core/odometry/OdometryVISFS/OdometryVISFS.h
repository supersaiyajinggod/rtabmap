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

#ifndef ODOMETRYVISFS_H_
#define ODOMETRYVISFS_H_

#include "rtabmap/core/Odometry.h"
#include "rtabmap/core/Signature.h"
#include "rtabmap/core/odometry/OdometryVISFS/StateEstimator.h"

namespace rtabmap {

class RTABMAP_EXP OdometryVISFS : public Odometry {
public:
    OdometryVISFS(const rtabmap::ParametersMap & _parameters = rtabmap::ParametersMap());
    virtual ~OdometryVISFS();

    virtual void reset(const Transform & _initialPose = Transform::getIdentity());
    virtual Odometry::Type getType() {return Odometry::kTypeVISFS;}
    virtual bool canProcessRawImage() const {return true;}
    virtual bool canProcessIMU() const {return true;}

private:
    virtual Transform computeTransform(SensorData & _data, const Transform & _guess = Transform(), OdometryInfo * _info = 0);

private:
    //private variable
    Transform lastFramePose_;
    Signature lastFrame_;
	ParametersMap parameters_;

    StateEstimator * stateEstimator_;
	#define featureTracker stateEstimator_->featureTracker_
	#define featureManager stateEstimator_->featureManager_
};


}

#endif /*ODOMETRYVISFS_H_ */