#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

struct Pose {
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};

void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

int main() {

    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();
        std::cout << "Pc of frame " << i << " : " << x << " " << y << " " << z << std::endl;
        camera_pose[i].uv = Eigen::Vector2d(x/z,y/z);
    }
    
    // 遍历所有的观测数据，并三角化。
    // 测试思路：两两三角化，比较深度值。然后最小二乘优化比较深度值。
    Eigen::MatrixXd svd_A(2 * ((end_frame_id - 1) - start_frame_id), 4);
    int svd_idx = 0;
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();

    for (int i = start_frame_id; i < end_frame_id - 1; ++i) {
        // Between two frame
        Eigen::Matrix<double, 3, 4> leftPose;
        leftPose.leftCols<3>() = camera_pose[i].Rwc.transpose();
        leftPose.rightCols<1>() = -leftPose.leftCols<3>()*camera_pose[i].twc;
        Eigen::Matrix<double, 3, 4> rightPose;
        rightPose.leftCols<3>() = camera_pose[i+1].Rwc.transpose();
        rightPose.rightCols<1>() = -rightPose.leftCols<3>()*camera_pose[i+1].twc;
        Eigen::Vector2d point0, point1;
        Eigen::Vector3d point3d;
        point0 = camera_pose[i].uv;
        point1 = camera_pose[i+1].uv;
        triangulatePoint(leftPose, rightPose, point0, point1, point3d);
        std::cout << "Landmark between frame: " << i << " and " << i+1 << " is: " <<  point3d.transpose() << std::endl;
        Eigen::Vector3d localPoint;
        localPoint = rightPose.leftCols<3>() * point3d + rightPose.rightCols<1>();        
        // double depth = localPoint.z();
        std::cout << "frame " << i << "'s localPoint: " << localPoint.transpose() << std::endl;

        // Add optimization
        Eigen::Vector3d f = Eigen::Vector3d(camera_pose[i].uv[0], camera_pose[i].uv[1], 1).normalized();
        svd_A.row(svd_idx++) = f[0] * leftPose.row(2) - f[2] * leftPose.row(0);
        svd_A.row(svd_idx++) = f[1] * leftPose.row(2) - f[2] * leftPose.row(1);        
    }

    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    P_est = Eigen::Vector3d(svd_V[0]/svd_V[3], svd_V[1]/svd_V[3], svd_V[2]/svd_V[3]);
    
    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"my result: \n"<< P_est.transpose() <<std::endl;
    return 0;
}

