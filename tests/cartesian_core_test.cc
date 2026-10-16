#include <iostream>
#include <stdexcept>

#include "cartesian/core.h"
using namespace cartesian;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  State<3> a, b;
  a.p << .1, .2, .3;
  a.v << .2, -.1, 0;
  a.a << .1, .2, -.2;
  b.p << .5, .3, .1;
  b.v << .4, 0, .2;
  b.a << 0, -.1, .2;
  for (double T : {.02, .1, 1., 3.}) {
    auto q = Quintic<3>::Fit(a, b, T);
    auto x = q.Eval(0), y = q.Eval(T);
    Check((x.p - a.p).norm() < 1e-10 && (x.v - a.v).norm() < 1e-10 &&
              (x.a - a.a).norm() < 1e-8,
          "initial boundary");
    Check((y.p - b.p).norm() < 1e-10 && (y.v - b.v).norm() < 1e-9 &&
              (y.a - b.a).norm() < 1e-7,
          "terminal boundary");
    auto bc = Bernstein(q, 0, .7, .95);
    for (int k = 0; k <= 100; ++k) {
      auto p = q.Eval(T * (.7 + .25 * k / 100.)).p;
      for (int j = 0; j < 3; ++j)
        Check(p[j] >= bc.row(j).minCoeff() - 1e-10 &&
                  p[j] <= bc.row(j).maxCoeff() + 1e-10,
              "restricted Bernstein hull");
    }
    double t = .43 * T, h = 1e-5 * T;
    auto z = q.Eval(t);
    Check(((q.Eval(t + h).p - q.Eval(t - h).p) / (2 * h) - z.v).norm() < 1e-5,
          "consistent velocity");
  }
  Model m;
  m.J.leftCols<3>().setIdentity();
  State<3> r;
  r.p << .1, .2, -.1;
  V7 zero = V7::Zero();
  Gains g;
  auto c = Track(m, zero, zero, zero, r, -V3::UnitZ(), g);
  Check(c.valid, "valid task");
  Check((c.torque.head<3>() - g.wn * g.wn * r.p).norm() < 1e-10,
        "operational acceleration feedback");
  V7 nominal = V7::Ones();
  auto n = Track(m, zero, zero, nominal, r, -V3::UnitZ(), g);
  Check((m.J * m.M.inverse() * (n.torque - c.torque)).norm() < 1e-10,
        "null-space task separation");
  m.J.setZero();
  Check(!Track(m, zero, zero, zero, r, -V3::UnitZ(), g).valid,
        "singular rejection");
  bool clipped = false;
  auto limited = Limit(V7::Constant(100.), zero, .002, 700., &clipped);
  Check(clipped && (limited.array() - 1.4).abs().maxCoeff() < 1e-12,
        "period-scaled rate limit");
  Check(Filter(V7::Ones(), zero, .001, 100).maxCoeff() < 1.,
        "velocity filtering");
  Check((AxisTarget(V3::UnitX(), 0, 1) - V3::UnitX()).norm() < 1e-12,
        "axis blend initial");
  Check((AxisTarget(V3::UnitX(), 1, 1) + V3::UnitZ()).norm() < 1e-12,
        "axis blend final");
  J3 jv = J3::Random(), jw = J3::Random();
  V3 tip_offset(.1, .2, .3384);
  V7 dq = V7::Random();
  Check((ShiftJacobian(tip_offset, jv, jw) * dq - jv * dq -
         (jw * dq).cross(tip_offset))
                .norm() < 1e-12,
        "offset tip Jacobian");
  M7 B = M7::Random();
  m.M = B.transpose() * B + M7::Identity();
  m.J = J3::Random();
  auto c0 = Track(m, zero, zero, zero, State<3>(), -V3::UnitZ(), g);
  auto c1 = Track(m, zero, zero, V7::Ones(), State<3>(), -V3::UnitZ(), g);
  Check(c0.valid && c1.valid &&
            (m.J * m.M.inverse() * (c1.torque - c0.torque)).norm() < 1e-9,
        "coupled-inertia null space");
  Check(CanAccept(true, false, .0005, zero, .001, .05),
        "admit matching rest state");
  Check(!CanAccept(false, false, 0., zero, .001, .05),
        "reject moving command replacement");
  Check(!CanAccept(true, true, 0., zero, .001, .05),
        "reject faulted controller");
  Check(!CanAccept(true, false, .002, zero, .001, .05),
        "reject position mismatch");
  Check(!CanAccept(true, false, 0., V7::Constant(.1), .001, .05),
        "reject nonrest robot");
  Check(FinishWeight(1., 2.) == 0. && FinishWeight(2., 2.) == 1.,
        "smooth finishing window");
  auto finish = Track(m, zero, V7::Constant(.1), nominal, State<3>(),
                      -V3::UnitZ(), g, 1.);
  auto moving = Track(m, zero, V7::Constant(.1), nominal, State<3>(),
                      -V3::UnitZ(), g, 0.);
  Check((m.J * m.M.inverse() * (finish.torque - moving.torque)).norm() < 1e-9,
        "finishing preserves translation task");
  std::cout << "PASS quintics, interval hulls, derivatives, control, null "
               "space, singularity, rate limits, filter\n";
}
