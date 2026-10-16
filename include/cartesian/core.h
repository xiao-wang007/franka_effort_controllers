#pragma once
// ROS/Drake-independent mathematics shared by planning and the 1 kHz
// controller.
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <stdexcept>

namespace cartesian {
using V3 = Eigen::Vector3d;
using V7 = Eigen::Matrix<double, 7, 1>;
using M7 = Eigen::Matrix<double, 7, 7>;
using J3 = Eigen::Matrix<double, 3, 7>;

template <int D>
struct State {
  Eigen::Matrix<double, D, 1> p = Eigen::Matrix<double, D, 1>::Zero();
  Eigen::Matrix<double, D, 1> v = Eigen::Matrix<double, D, 1>::Zero();
  Eigen::Matrix<double, D, 1> a = Eigen::Matrix<double, D, 1>::Zero();
};
template <int D>
struct Quintic {
  Eigen::Matrix<double, D, 6> c = Eigen::Matrix<double, D, 6>::Zero();
  double duration = 1.;
  static Quintic Fit(const State<D>& x, const State<D>& y, double T) {
    if (!(T > 0.) || !std::isfinite(T))
      throw std::invalid_argument("invalid segment duration");
    Quintic s;
    s.duration = T;
    s.c.col(0) = x.p;
    s.c.col(1) = T * x.v;
    s.c.col(2) = .5 * T * T * x.a;
    const auto d = (y.p - s.c.col(0) - s.c.col(1) - s.c.col(2)).eval();
    const auto v = (T * y.v - s.c.col(1) - 2 * s.c.col(2)).eval();
    const auto a = (T * T * y.a - 2 * s.c.col(2)).eval();
    s.c.col(3) = 10 * d - 4 * v + .5 * a;
    s.c.col(4) = -15 * d + 7 * v - a;
    s.c.col(5) = 6 * d - 3 * v + .5 * a;
    return s;
  }
  State<D> Eval(double t) const {
    const double s = std::max(0., std::min(1., t / duration));
    State<D> x;
    for (int j = 5; j >= 0; --j) x.p = x.p * s + c.col(j);
    for (int j = 5; j >= 1; --j) x.v = x.v * s + j * c.col(j);
    for (int j = 5; j >= 2; --j) x.a = x.a * s + j * (j - 1) * c.col(j);
    x.v /= duration;
    x.a /= duration * duration;
    return x;
  }
};

inline double Choose(int n, int k) {
  double x = 1.;
  for (int i = 1; i <= k; ++i) x *= double(n - i + 1) / i;
  return x;
}
// Bernstein coefficients of derivative d on the restricted normalized interval
// [lo,hi].
template <int D>
inline Eigen::Matrix<double, D, 6> Bernstein(const Quintic<D>& q, int d = 0,
                                             double lo = 0., double hi = 1.) {
  const int degree = 5 - d;
  Eigen::Matrix<double, D, 6> power = Eigen::Matrix<double, D, 6>::Zero(),
                              out = power;
  for (int j = d; j <= 5; ++j) {
    double f = 1.;
    for (int k = 0; k < d; ++k) f *= double(j - k) / q.duration;
    for (int k = 0; k <= j - d; ++k)
      power.col(k) += q.c.col(j) * f * Choose(j - d, k) *
                      std::pow(lo, j - d - k) * std::pow(hi - lo, k);
  }
  for (int i = 0; i <= degree; ++i)
    for (int k = 0; k <= i; ++k)
      out.col(i) += power.col(k) * Choose(i, k) / Choose(degree, k);
  return out;
}
struct Trajectory {
  std::array<Quintic<3>, 3> segments;
  int count = 0;
  double contact_time = -1., carry_end_time = -1.;
  double Duration() const {
    double t = 0.;
    for (int i = 0; i < count; ++i) t += segments[i].duration;
    return t;
  }
  State<3> Eval(double t) const {
    for (int i = 0; i < count; ++i) {
      if (t <= segments[i].duration || i == count - 1)
        return segments[i].Eval(t);
      t -= segments[i].duration;
    }
    return State<3>();
  }
};
struct Gains {
  double wn = 20., zeta = 1., posture_wn = 2., axis_k = 5., axis_d = 1.;
  double filter_hz = 100., torque_rate = 700., eigen_floor = 1e-4;
};
struct Model {
  V3 p = V3::Zero(), axis = -V3::UnitZ(), bias = V3::Zero();
  J3 J = J3::Zero(), Jw = J3::Zero();
  M7 M = M7::Identity();
  V7 coriolis = V7::Zero();
};
inline J3 ShiftJacobian(const V3& world_offset, const J3& linear,
                        const J3& angular) {
  Eigen::Matrix3d skew;
  skew << 0, -world_offset.z(), world_offset.y(), world_offset.z(), 0,
      -world_offset.x(), -world_offset.y(), world_offset.x(), 0;
  return linear - skew * angular;
}
struct Control {
  V7 torque = V7::Zero();
  double condition = 0.;
  bool valid = false;
};
inline V3 AxisTarget(const V3& initial, double t, double blend) {
  double s = std::max(0., std::min(1., t / std::max(blend, 1e-6)));
  s = s * s * s * (10 + s * (-15 + 6 * s));
  return (Eigen::Quaterniond::Identity().slerp(
              s, Eigen::Quaterniond::FromTwoVectors(initial, -V3::UnitZ())) *
          initial)
      .normalized();
}
inline double FinishWeight(double time, double duration) {
  const double window = std::min(.5, .5 * duration);
  double s = std::max(
      0., std::min(1., (time - (duration - window)) / std::max(window, 1e-6)));
  return s * s * s * (10 + s * (-15 + 6 * s));
}
inline Control Track(const Model& m, const V7& q, const V7& dq,
                     const V7& nominal, const State<3>& ref,
                     const V3& target_axis, const Gains& g,
                     double finish = 0.) {
  Control out;
  if (!m.M.allFinite() || !m.J.allFinite() || !q.allFinite() ||
      !dq.allFinite() || !m.p.allFinite() || !m.bias.allFinite() ||
      !ref.p.allFinite() || !ref.v.allFinite() || !ref.a.allFinite())
    return out;
  Eigen::LDLT<M7> factor(m.M);
  if (factor.info() != Eigen::Success || !factor.isPositive()) return out;
  const Eigen::Matrix<double, 7, 3> MinvJt = factor.solve(m.J.transpose());
  const Eigen::Matrix3d A = m.J * MinvJt;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(A);
  if (eig.info() != Eigen::Success || eig.eigenvalues().maxCoeff() <= 0.)
    return out;
  const double top = eig.eigenvalues().maxCoeff(),
               bottom = eig.eigenvalues().minCoeff();
  out.condition = top / std::max(bottom, 1e-15);
  const V3 inv = eig.eigenvalues().cwiseMax(top * g.eigen_floor).cwiseInverse();
  const Eigen::Matrix3d lambda =
      eig.eigenvectors() * inv.asDiagonal() * eig.eigenvectors().transpose();
  const Eigen::Matrix<double, 7, 3> Jbar = MinvJt * lambda;
  const M7 Nt = M7::Identity() - m.J.transpose() * Jbar.transpose();
  const V3 omega = m.Jw * dq;
  const V3 axis_torque = g.axis_k * m.axis.cross(target_axis) -
                         g.axis_d * (omega - m.axis * m.axis.dot(omega));
  const V7 posture = m.M.diagonal().cwiseProduct(
      g.posture_wn * g.posture_wn * (nominal - q) - 2 * g.posture_wn * dq);
  const V3 accel = ref.a + g.wn * g.wn * (ref.p - m.p) +
                   2 * g.zeta * g.wn * (ref.v - m.J * dq);
  const V7 secondary =
      (1. - finish) * (m.Jw.transpose() * axis_torque + posture) -
      finish * 2 * g.wn * m.M * dq;
  out.torque =
      m.J.transpose() * lambda * (accel - m.bias) + m.coriolis + Nt * secondary;
  out.valid = out.torque.allFinite() && out.condition <= 1. / g.eigen_floor;
  return out;
}
inline V7 TorqueLimits() {
  V7 x;
  x << 87, 87, 87, 87, 12, 12, 12;
  return x;
}
inline V7 Limit(const V7& requested, const V7& previous, double dt, double rate,
                bool* clipped = nullptr) {
  const V7 limits = TorqueLimits();
  const V7 bounded = requested.cwiseMax(-limits).cwiseMin(limits);
  const double step = std::max(0., dt) * rate;
  const V7 applied =
      previous + (bounded - previous).cwiseMax(-step).cwiseMin(step);
  if (clipped) *clipped = (applied - requested).cwiseAbs().maxCoeff() > 1e-9;
  return applied;
}
inline V7 Filter(const V7& measured, const V7& old, double dt, double hz) {
  const double alpha = dt / (dt + 1. / (2. * 3.141592653589793 * hz));
  return old + alpha * (measured - old);
}
inline bool CanAccept(bool holding, bool fault, double position_gap,
                      const V7& velocity, double position_tolerance,
                      double velocity_tolerance) {
  return holding && !fault && std::isfinite(position_gap) &&
         position_gap <= position_tolerance && velocity.allFinite() &&
         velocity.cwiseAbs().maxCoeff() <= velocity_tolerance;
}
}  // namespace cartesian
