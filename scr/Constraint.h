#pragma once

#include <map>

#include "Mesh.h"

struct IneqCon {

    Eigen::Vector3d n;
    double a;   // area
    double mu;
    double stiff;
    double repulsion_thickness;

    Node* nodes[4];
    double w[4];
    bool free[4];

    double value(int* sign = nullptr);
    std::map<Node*, Eigen::Vector3d> gradient();

    double violation(double);

    double energy_grad(double value);
    double energy_hess(double value);

    std::map<Node*, Eigen::Vector3d> friction(double dt, std::map<std::pair<Node*, Node* >, Eigen::Matrix3d>& jac);

    // Constructor
    IneqCon() : n{ Eigen::Vector3d::Zero() }, a{ 0.0 }, mu{ 0.0 }, stiff{ 0.0 }, repulsion_thickness{0.0} {
        for (unsigned int i = 0; i < 4; ++i) {
            nodes[i] = nullptr;
            free[i] = false;
            w[i] = 0.0;
        }
    }
       
    // Copy Constructor
    IneqCon(const IneqCon& other) = delete;
    // Copy Assignment
    IneqCon& operator=(const IneqCon& other) = delete;

    // Move Constructor
    IneqCon(IneqCon&& other) = delete;

    // Move Assignment
    IneqCon& operator=(IneqCon&& other) = delete;

    // Destructor
    ~IneqCon() {

        for (unsigned int i = 0; i < 4; ++i) 
            nodes[i] = nullptr;

    }
};