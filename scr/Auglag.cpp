#include "Auglag.h"

#include <omp.h>
#include <alglib/optimization.h>

using namespace std;
using namespace alglib;

static const NLConOpt* problem;
static vector<double> lambda;
static double mu;

static void auglag_value_and_grad(const real_1d_array& x, double& value,
    real_1d_array& grad, void* ptr = NULL);

static void multiplier_update(const real_1d_array& x);

void augmented_lagrangian_method(const NLConOpt& problem, OptOptions opt, bool verbose) {
    ::problem = &problem;
    ::lambda = vector<double>(::problem->ncon, 0);
    ::mu = 1e3;
    real_1d_array x;
    x.setlength(::problem->nvar);
    ::problem->initialize(&x[0]);

    mincgstate state;
    mincgreport rep;
    mincgcreate(x, state);
    const int max_total_iter = opt.max_iter(), max_sub_iter = std::floor(sqrt(max_total_iter));
    int iter = 0;
    while (iter < max_total_iter) {
        int max_iter = min(max_sub_iter, max_total_iter - iter);
        mincgsetcond(state, opt.eps_g(), opt.eps_f(), opt.eps_x(), max_iter);
        if (iter > 0)
            mincgrestartfrom(state, x);
        mincgsuggeststep(state, 1e-3 * ::problem->nvar);
        mincgoptimize(state, auglag_value_and_grad);
        mincgresults(state, x, rep);
        multiplier_update(x);

        if (verbose)
            cout << rep.iterationscount << " iterations" << endl;
        if (rep.iterationscount == 0)
            break;
        iter += rep.iterationscount;
    }
    ::problem->finalize(&x[0]);
}


inline double clamp_violation(double x, int sign) {
    return (sign < 0) ? max(x, 0.) : (sign > 0) ? min(x, 0.) : x;
}

static void auglag_value_and_grad(const real_1d_array& x, double& value,
    real_1d_array& grad, void* ptr) {
    ::problem->precompute(&x[0]);
    value = ::problem->objective(&x[0]);
    ::problem->obj_grad(&x[0], &grad[0]);
    static const int nthreads = omp_get_max_threads();
    static double* values = new double[nthreads];
    static vector<double>* grads = new vector<double>[nthreads];
    for (int t = 0; t < nthreads; t++) {
        values[t] = 0;
        grads[t].assign(::problem->nvar, 0);
    }
#pragma omp parallel for
    for (int j = 0; j < ::problem->ncon; j++) {
        int t = omp_get_thread_num();
        int sign;
        double gj = ::problem->constraint(&x[0], j, sign);
        double cj = clamp_violation(gj + ::lambda[j] / ::mu, sign);
        if (cj != 0) {
            values[t] += ::mu / 2 * (cj * cj);
            ::problem->con_grad(&x[0], j, ::mu * cj, &grads[t][0]);
        }
    }
    for (int t = 0; t < nthreads; t++)
        value += values[t];
#pragma omp parallel for
    for (int i = 0; i < ::problem->nvar; i++)
        for (int t = 0; t < nthreads; t++)
            grad[i] += grads[t][i];
}

static void multiplier_update(const real_1d_array& x) {
    ::problem->precompute(&x[0]);
#pragma omp parallel for
    for (int j = 0; j < ::problem->ncon; j++) {
        int sign;
        double gj = ::problem->constraint(&x[0], j, sign);
        ::lambda[j] = clamp_violation(::lambda[j] + ::mu * gj, sign);
    }
}