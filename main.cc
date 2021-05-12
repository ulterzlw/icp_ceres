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

struct AutoDiffErrorTerm {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AutoDiffErrorTerm(Eigen::Vector3d s, Eigen::Vector3d t) : s(s), t(t) {};

  template<typename T>
  bool operator()(const T* const parameters,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> param(parameters);
    Sophus::SE3<T> tf = Sophus::SE3<T>::exp(param);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> r(residuals);

    r = t - tf * s;
    return true;
  }
  static ceres::CostFunction* Create(const Eigen::Vector3d s,
                                     Eigen::Vector3d t) {
    return new ceres::AutoDiffCostFunction<AutoDiffErrorTerm, 3, 6>(
        new AutoDiffErrorTerm(s, t));
  }

 protected:
  Eigen::Vector3d s;
  Eigen::Vector3d t;
};


template<typename T>
inline void buildDenseMatrix(const ceres::CRSMatrix Ain, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &A)
{
    A.resize(Ain.num_rows, Ain.num_cols);
    A.setZero();
    for (int r = 0; r < Ain.num_rows; ++r) {
        for (int i = Ain.rows[r]; i < Ain.rows[r + 1]; ++i) {
            A(r, Ain.cols[i]) = T(Ain.values[i]);
        }
    }
}

int main(int argc, char **argv) {
  std::vector<Eigen::Vector3d> target;
  target.emplace_back(Eigen::Vector3d(-1, -1, 1));
  target.emplace_back(Eigen::Vector3d(1, -1, 1));
  target.emplace_back(Eigen::Vector3d(-1, 1, 1));
  target.emplace_back(Eigen::Vector3d(1, 1, 1));
  target.emplace_back(Eigen::Vector3d(-1, -1, -1));
  target.emplace_back(Eigen::Vector3d(1, -1, -1));
  target.emplace_back(Eigen::Vector3d(-1, 1, -1));
  target.emplace_back(Eigen::Vector3d(1, 1, -1));

  for(auto i: target) std::cout<<"-t: "<<i.transpose()<<std::endl;
  Sophus::SE3d
      tf(Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0)), Eigen::Vector3d(0, 0, 0));
  std::vector<Eigen::Vector3d> source;
  for (int i = 0; i < target.size(); ++i) {
    auto p = tf*target.at(i);
    source.emplace_back(p);
  }

  //noise
  // for(auto& i: source) i+= 0.1 * Eigen::Vector3d::Random();
  for(auto& i: source) i+= Eigen::Vector3d(0.5,0,0);
  for(auto i: source) std::cout<<"-s: "<<i.transpose()<<std::endl;

  std::cout << "truth:\n" << tf.matrix().inverse() << std::endl;
  // Eigen::Matrix<double, 6, 1> tf_calculated_parameter(Eigen::Matrix<double, 6, 1>::Random());
  Eigen::Matrix<double, 6, 1> tf_calculated_parameter(Eigen::Matrix<double, 6, 1>::Zero());
  std::cout << "initial:\n"
            << Sophus::SE3d::exp(tf_calculated_parameter).matrix() << std::endl;

  ceres::Problem problem;
  std::vector<double *> para_ids;
  std::vector<ceres::internal::ResidualBlock *> res_ids;


  for (int i = 0; i < source.size(); ++i) {
    ceres::CostFunction *cost_function = new AnalyticalErrorTerm(source.at(i), target.at(i));
    // ceres::CostFunction *cost_function = AutoDiffErrorTerm::Create(source.at(i), target.at(i));
    res_ids.push_back(problem.AddResidualBlock(cost_function, NULL, tf_calculated_parameter.data()));
  }
  para_ids.push_back(tf_calculated_parameter.data());
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  // ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;
  std::cout << "result:\n"
            << Sophus::SE3d::exp(tf_calculated_parameter).matrix()<<std::endl;

  double cost;
  ceres::CRSMatrix jaco_crs;
  ceres::Problem::EvaluateOptions e_option;
  e_option.parameter_blocks = para_ids;
  e_option.residual_blocks = res_ids;
  std::vector<double> residuals;

  problem.Evaluate(e_option, &cost, &residuals, nullptr, &jaco_crs);
  std::cout<<"cost: " << cost<<std::endl;
  for(auto i: residuals) std::cout<<i<<" ";
  std::cout<<std::endl;
  Eigen::MatrixXd jaco;
  buildDenseMatrix(jaco_crs, jaco);
  Eigen::MatrixXd H = jaco.transpose() * jaco;
  std::cout << "H: \n" << H << std::endl;
  
  std::cout<<jaco_crs.num_rows<<std::endl;
  double s2 = cost / (jaco_crs.num_rows - 6);
  Eigen::MatrixXd I = 1/(2*s2)*H;
  std::cout<<"I: \n" << I << std::endl;
  Eigen::MatrixXd cov = I.inverse();
  std::cout<<"cov: \n" << cov << std::endl;

  return 0;

}