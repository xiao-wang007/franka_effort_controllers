#pragma once
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <algorithm>

class MultiCubicHermiteSpline {
public:
  void fit(const Eigen::VectorXd& x,
           const Eigen::MatrixXd& Y,
           const Eigen::MatrixXd& dY) {
    if (x.size() < 2) throw std::invalid_argument("need >=2 knots");
    if (Y.rows() != x.size() || dY.rows() != x.size())
      throw std::invalid_argument("Y and dY rows must match x size");
    if (Y.cols() != dY.cols())
      throw std::invalid_argument("Y and dY must have same number of cols");
    for (int i = 1; i < x.size(); ++i)
      if (!(x[i] > x[i-1])) throw std::invalid_argument("x must be strictly increasing");
    x_ = x; Y_ = Y; dY_ = dY;
  }

  // Zero-order hold outside [x0, xN]
  Eigen::VectorXd eval(double t) const {
    const int N = x_.size() - 1;
    if (t <= x_[0]) return Y_.row(0).transpose();
    if (t >= x_[N]) return Y_.row(N).transpose();

    int i = intervalIndex(t);
    double h = x_[i+1] - x_[i];
    double u = (t - x_[i]) / h;
    double u2 = u*u, u3 = u2*u;

    double h00 =  2*u3 - 3*u2 + 1;
    double h10 =      u3 - 2*u2 + u;
    double h01 = -2*u3 + 3*u2;
    double h11 =      u3 -   u2;

    return h00 * Y_.row(i).transpose()
         + h10 * h * dY_.row(i).transpose()
         + h01 * Y_.row(i+1).transpose()
         + h11 * h * dY_.row(i+1).transpose();
  }

  // Derivative is zero outside [x0, xN] when clamped
  Eigen::VectorXd evald(double t) const {
    const int N = x_.size() - 1;
    if (t <= x_[0] || t >= x_[N]) return Eigen::VectorXd::Zero(Y_.cols());

    int i = intervalIndex(t);
    double h = x_[i+1] - x_[i];
    double u = (t - x_[i]) / h;
    double u2 = u*u;

    double h00d = (6*u2 - 6*u) / h;
    double h10d = (3*u2 - 4*u + 1) / h;
    double h01d = (-6*u2 + 6*u) / h;
    double h11d = (3*u2 - 2*u) / h;

    return h00d * Y_.row(i).transpose()
         + h10d * h * dY_.row(i).transpose()
         + h01d * Y_.row(i+1).transpose()
         + h11d * h * dY_.row(i+1).transpose();
  }

private:
  int intervalIndex(double t) const {
    const int N = x_.size()-1;
    // t guaranteed to be strictly inside here (eval/evald clamp first)
    const double* xb = x_.data();
    const double* xe = xb + (N+1);
    const double* it = std::upper_bound(xb, xe, t);
    int idx = static_cast<int>((it - xb) - 1);
    if (idx < 0) idx = 0;
    if (idx > N-1) idx = N-1;
    return idx;
  }

  Eigen::VectorXd x_;
  Eigen::MatrixXd Y_, dY_;
};
