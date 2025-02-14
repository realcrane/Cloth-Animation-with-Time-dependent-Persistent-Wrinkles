#include "Constraint.h"

double IneqCon::value(int* sign) {
    if (sign)
        *sign = 1;
    double d{ 0.0 };
    for (int i = 0; i < 4; i++) {
        d = d + w[i] * n.dot(nodes[i]->x);
    }
    d = d - repulsion_thickness;
    return d;
}

std::map<Node*, Eigen::Vector3d> IneqCon::gradient() {
    std::map<Node*, Eigen::Vector3d> grad;
    for (int i = 0; i < 4; i++) {
        grad[nodes[i]] = w[i] * n;
    }
    return grad;
}

double IneqCon::violation(double value) {
    return std::max(-value, 0.0);
}

double IneqCon::energy_grad(double value)
{  
    return -stiff * (violation(value) * violation(value)) / repulsion_thickness / 2;
}

double IneqCon::energy_hess(double value)
{    
    return stiff * violation(value) / repulsion_thickness;
}

std::map<Node*, Eigen::Vector3d> IneqCon::friction(double dt, std::map<std::pair<Node*, Node*>, Eigen::Matrix3d>& jac)
{
    std::map<Node*, Eigen::Vector3d> fric_grad;

    if (mu == 0)
        return fric_grad;

    double fn = abs(energy_grad(value()));

    if (fn == 0)
        return fric_grad;

    Eigen::Vector3d v {0.0, 0.0, 0.0};
    double inv_mass = 0;

    for (int i = 0; i < 4; i++) {
        v = v + w[i] * nodes[i]->v;
        if (free[i])
            inv_mass = inv_mass + (w[i] * w[i]) / nodes[i]->m;
    }
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity() - n * n.transpose();
    double vt = (T * v).norm();
    double f_by_v = std::min(mu * fn / vt, 1.0 / (dt * inv_mass));
    for (int i = 0; i < 4; i++) {
        if (free[i]) {
            fric_grad[nodes[i]] = -w[i] * f_by_v * (T * v);
            for (int j = 0; j < 4; j++) {
                if (free[j]) {
                    jac[std::make_pair(nodes[i], nodes[j])] = -w[i] * w[j] * f_by_v * T;
                }
            }
        }
    }
    return fric_grad;
}

