#pragma once

#include "Optimization.h"

void augmented_lagrangian_method(const NLConOpt& problem, OptOptions opts = OptOptions(), bool verbose = false);
