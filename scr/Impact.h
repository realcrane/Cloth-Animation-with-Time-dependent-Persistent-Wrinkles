#pragma once
#include "AccStructure.h"

struct Impact {
    enum Type { VF, EE } type;
    double t;
    double w[4];
    Eigen::Vector3d n;

    Node* nodes[4];

    Impact(Type type, const Node* n0, const Node* n1, const Node* n2,
        const Node* n3) : type{ type }, t{ 0.0 }, w{}, n{Eigen::Vector3d::Zero()} {
        nodes[0] = (Node*)n0;
        nodes[1] = (Node*)n1;
        nodes[2] = (Node*)n2;
        nodes[3] = (Node*)n3;
    }


    Impact() : type{ Type::VF }, t{ 0.0 }, w{}, n{ Eigen::Vector3d::Zero() }{

        nodes[0] = nullptr;
        nodes[1] = nullptr;
        nodes[2] = nullptr;
        nodes[3] = nullptr;

        //std::cout << "Create Impact" << std::endl;
    }

    // Copy Constructor
    Impact(const Impact& other) {

        this->type = other.type;
        this->t = other.t;
        this->n = other.n;

        for (unsigned int i = 0; i < 4; ++i) {
            this->w[i] = other.w[i];
            this->nodes[i] = other.nodes[i];
        }
    }

    // Copy Assignment
    Impact& operator=(const Impact& other) {

        if (this == &other) return *this;

        this->type = other.type;
        this->t = other.t;
        this->n = other.n;

        for (unsigned int i = 0; i < 4; ++i) {
            this->w[i] = other.w[i];
            this->nodes[i] = other.nodes[i];
        }
            
        return *this;
    }

    // Move Constructor
    Impact(Impact&& other) noexcept {

        this->type = other.type;
        this->t = other.t;
        this->n = other.n;

        other.t = 0.0;
        other.n = Eigen::Vector3d::Zero();

        for (unsigned int i = 0; i < 4; ++i) {
            this->w[i] = other.w[i];
            this->nodes[i] = other.nodes[i];

            other.w[i] = 0.0;
            other.nodes[i] = nullptr;
        }
    }

    // Move Assignment
    Impact& operator=(Impact&& other) noexcept {

        if (this == &other) return *this;

        this->type = other.type;
        this->t = other.t;
        this->n = other.n;

        other.t = 0.0;
        other.n = Eigen::Vector3d::Zero();

        for (unsigned int i = 0; i < 4; ++i) {
            this->w[i] = other.w[i];
            this->nodes[i] = other.nodes[i];

            other.w[i] = 0.0;
            other.nodes[i] = nullptr;
        }

        return *this;
    }

    // Destructor
    ~Impact() = default;
};

struct ImpactZone {
    std::vector<Node*> nodes;
    std::vector<Impact> impacts;
    std::vector<double> w, n;
    bool active;

    // Default Constructor
    ImpactZone() : active{ false } {};

    // Copy Constructor
    ImpactZone(const ImpactZone& other) = delete;

    // Copy Assignment
    ImpactZone& operator=(const ImpactZone& other) {
        
        if (this == &other) return *this;

        this->nodes = other.nodes;
        this->impacts = other.impacts;
        this->w = other.w;
        this->n = other.n;
        this->active = other.active;

        return *this;
    }

    // Move Constructor
    ImpactZone(ImpactZone&& other) = delete;

    // Move Assignment
    ImpactZone& operator=(ImpactZone&& other) = delete;

    // Destructor
    ~ImpactZone() = default;
};