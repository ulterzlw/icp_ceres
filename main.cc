//
// Created by alterlimbo on 10/23/19.
//

#include <vector>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

struct AnalyticalErrorTerm : public ceres::SizedCostFunction<3, 6> {
 public:
  AnalyticalErrorTerm(Eigen::Vector3d s, Eigen::Vector3d t) : s(s), t(t) {};
  virtual ~AnalyticalErrorTerm() {};
  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> param(parameters[0]);
    Sophus::SE3d tf = Sophus::SE3d::exp(param);

    Eigen::Vector3d s_trans = tf * s;

    residuals[0] = (t - s_trans).x();
    residuals[1] = (t - s_trans).y();
    residuals[2] = (t - s_trans).z();

    Eigen::Matrix<double, 3, 6> jaco;
    jaco.block(0, 0, 3, 3) = -Eigen::Matrix3d::Identity();
    jaco.block(0, 3, 3, 3) = Sophus::SO3d::hat(s_trans);
    Eigen::Matrix<double, 6, 3> jaco_transpose = jaco.transpose();
    if (jacobians != NULL && jacobians[0] != NULL) {
      for (int i = 0; i < 18; ++i) {
        jacobians[0][i] = jaco_transpose(i);
      }
    }
    return true;
  }

 public:
  Eigen::Vector3d s;
  Eigen::Vector3d t;
};

int main(int argc, char **argv) {
  std::vector<Eigen::Vector3d> source;
  source.emplace_back(Eigen::Vector3d(0, 0, 0));
  source.emplace_back(Eigen::Vector3d(1, 0, 0));
  source.emplace_back(Eigen::Vector3d(0, 1, 0));

  Sophus::SE3d
      tf(Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 1)), Eigen::Vector3d(1, 0, 0));
  std::cout << "truth:\n" << tf.matrix() << std::endl;
  std::vector<Eigen::Vector3d> target;
  for (int i = 0; i < source.size(); ++i) {
    target.emplace_back(tf * source.at(i));
  }

  Eigen::Matrix<double, 6, 1> tf_calculated_parameter(Eigen::Matrix<double,
                                                                    6,
                                                                    1>::Random());
  std::cout << "initial:\n"
            << Sophus::SE3d::exp(tf_calculated_parameter).matrix() << std::endl;

  ceres::Problem problem;

  for (int i = 0; i < source.size(); ++i) {
    ceres::CostFunction
        *cost_function = new AnalyticalErrorTerm(source.at(i), target.at(i));
    problem.AddResidualBlock(cost_function,
                             NULL,
                             tf_calculated_parameter.data());
  }
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;
  std::cout << "result:\n"
            << Sophus::SE3d::exp(tf_calculated_parameter).matrix();

  return 0;

}