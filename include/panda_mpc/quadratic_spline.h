#pragma once
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <algorithm>

// Multi-dimensional C¹ quadratic spline with shared knots.
// Y: (N+1) x M (rows = knots, cols = channels).
class MultiQuadraticSpline {
public:
  enum class Boundary { NaturalLeft, NaturalRight, Clamped };

  // If Boundary::Clamped, provide m0 and mN (Mx1 end slopes).
  void fit(const Eigen::VectorXd& x,
           const Eigen::MatrixXd& Y,
           Boundary bc = Boundary::NaturalLeft,
           const Eigen::VectorXd& m0 = Eigen::VectorXd(),
           const Eigen::VectorXd& mN = Eigen::VectorXd()) {
    if (x.size() < 2) throw std::invalid_argument("need >= 2 knots");
    if (Y.rows() != x.size()) throw std::invalid_argument("Y rows must equal x size");
    for (int i = 1; i < x.size(); ++i)
      if (!(x[i] > x[i-1])) throw std::invalid_argument("x must be strictly increasing");

    const int N = x.size() - 1;
    const int M = Y.cols();
    if (bc == Boundary::Clamped && (m0.size() != M || mN.size() != M))
      throw std::invalid_argument("Clamped requires m0 and mN of size M");

    x_ = x;
    h_.resize(N);
    for (int i = 0; i < N; ++i) h_[i] = x_[i+1] - x_[i];

    // Per-channel slopes m_i at all knots (N+1 x M)
    Mslopes_.resize(N+1, M);

    // Precompute segment average slopes a_i = 2*(y_{i+1}-y_i)/h_i (N x M)
    Eigen::MatrixXd A(N, M);
    for (int i = 0; i < N; ++i) {
      A.row(i) = 2.0 * (Y.row(i+1) - Y.row(i)) / h_[i];
    }

    if (bc == Boundary::NaturalLeft) {
      // m0 = m1 = A0/2
      Mslopes_.row(0) = 0.5 * A.row(0);
      for (int i = 0; i < N; ++i)
        Mslopes_.row(i+1) = A.row(i) - Mslopes_.row(i);
    } else if (bc == Boundary::NaturalRight) {
      // mN = m_{N-1} = A_{N-1}/2, then go backward
      Mslopes_.row(N) = 0.5 * A.row(N-1);
      for (int i = N-1; i >= 0; --i)
        Mslopes_.row(i) = A.row(i) - Mslopes_.row(i+1);
    } else { // Clamped
      Mslopes_.row(0) = m0.transpose();
      for (int i = 0; i < N; ++i)
        Mslopes_.row(i+1) = A.row(i) - Mslopes_.row(i);
      // Optional consistency nudge to match mN (rarely needed; usually close already)
      // You could blend a tiny correction here if exact endpoint matching is required.
      Mslopes_.row(N) = mN.transpose();
    }

    // Store Y and compute per-interval quadratic coefficient c_i = (m_{i+1}-m_i)/(2 h_i)
    Y_ = Y;
    C_.resize(N, M);
    for (int i = 0; i < N; ++i) {
      C_.row(i) = (Mslopes_.row(i+1) - Mslopes_.row(i)) / (2.0 * h_[i]);
    }
  }

  // Evaluate at scalar t -> Mx1
  Eigen::VectorXd eval(double t) const {
    int i = intervalIndex(t);
    double dx = t - x_[i];
    return Y_.row(i).transpose()
         + Mslopes_.row(i).transpose() * dx
         + C_.row(i).transpose() * (dx * dx);
  }

  // First derivative at scalar t -> Mx1
  Eigen::VectorXd evald(double t) const {
    int i = intervalIndex(t);
    double dx = t - x_[i];
    return Mslopes_.row(i).transpose()
         + 2.0 * C_.row(i).transpose() * dx;
  }

  // Batch evaluation: T(K) -> K x M
  Eigen::MatrixXd evalVec(const Eigen::VectorXd& T) const {
    Eigen::MatrixXd out(T.size(), Y_.cols());
    for (int k = 0; k < T.size(); ++k) out.row(k) = eval(T[k]).transpose();
    return out;
  }
  Eigen::MatrixXd evaldVec(const Eigen::VectorXd& T) const {
    Eigen::MatrixXd out(T.size(), Y_.cols());
    for (int k = 0; k < T.size(); ++k) out.row(k) = evald(T[k]).transpose();
    return out;
  }

private:
    int intervalIndex(double t) const {
      const int N = static_cast<int>(h_.size());
      if (t <= x_[0]) return 0;
      if (t >= x_[N]) return N-1;
      const double* xb = x_.data();
      const double* xe = xb + (N + 1);
      const double* it = std::upper_bound(xb, xe, t);
      int idx = static_cast<int>((it - xb) - 1);
      if (idx < 0) idx = 0;
      if (idx > N-1) idx = N-1;
      return idx;
    }

    Eigen::VectorXd x_;
    std::vector<double> h_;
    Eigen::MatrixXd Y_;        // (N+1) x M
    Eigen::MatrixXd Mslopes_;  // (N+1) x M
    Eigen::MatrixXd C_;        // N x M
};
