#pragma once


struct NLOpt { // nonlinear optimization problem
    // minimize objective
    int nvar;
    virtual void initialize(double* x) const = 0;
    virtual double objective(const double* x) const = 0;
    virtual void precompute(const double* x) const {}
    virtual void gradient(const double* x, double* g) const = 0;
    virtual void finalize(const double* x) const = 0;
};

struct NLConOpt { // nonlinear constrained optimization problem
    // minimize objective s.t. constraints = or <= 0
    int nvar, ncon;
    virtual void initialize(double* x) const = 0;
    virtual void precompute(const double* x) const {}
    virtual double objective(const double* x) const = 0;
    virtual void obj_grad(const double* x, double* grad) const = 0; // set
    virtual double constraint(const double* x, int j, int& sign) const = 0;
    virtual void con_grad(const double* x, int j, double factor,
        double* grad) const = 0; // add factor*gradient
    virtual void finalize(const double* x) const = 0;
};

struct OptOptions {
    int _max_iter;
    double _eps_x, _eps_f, _eps_g;
    OptOptions() : _max_iter(100), _eps_x(1e-6), _eps_f(1e-12), _eps_g(1e-6) {}
    // Named parameter idiom
    // http://www.parashift.com/c++-faq-lite/named-parameter-idiom.html
    OptOptions& max_iter(int n) { _max_iter = n; return *this; }
    OptOptions& eps_x(double e) { _eps_x = e; return *this; }
    OptOptions& eps_f(double e) { _eps_f = e; return *this; }
    OptOptions& eps_g(double e) { _eps_g = e; return *this; }
    int max_iter() { return _max_iter; }
    double eps_x() { return _eps_x; }
    double eps_f() { return _eps_f; }
    double eps_g() { return _eps_g; }
};