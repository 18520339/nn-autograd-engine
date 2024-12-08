#pragma once
#include <cmath>
#include <stdexcept>
using namespace std;

class LearningRateScheduler {
protected:
    double initial_learning_rate;
    string name;

public:
    LearningRateScheduler(double initial_lr, string name) : initial_learning_rate(initial_lr), name(name) {
        if (initial_lr <= 0) throw invalid_argument("Initial learning rate must be > 0");
    }
    virtual double operator()(int step) = 0;
};

class WarmUpAndDecayScheduler : public LearningRateScheduler {
private:
    double decay_rate;
    int warmup_steps, decay_steps;

public:
    WarmUpAndDecayScheduler(double initial_lr, int warmup_steps, int decay_steps, double decay_rate, string name = "WarmUpAndDecayScheduler")
        : LearningRateScheduler(initial_lr, name), warmup_steps(warmup_steps), decay_steps(decay_steps), decay_rate(decay_rate) {
        if (warmup_steps < 0) throw invalid_argument("Warmup steps must be >= 0");
        if (decay_steps <= 0) throw invalid_argument("Decay steps must be > 0");
        if (decay_rate <= 0) throw invalid_argument("Decay rate must be > 0");
    }

    double operator()(int step) {
        // Warmup phase: linear increase | Decay phase: exponential decay
        if (step < warmup_steps) return initial_learning_rate * (static_cast<double>(step) / warmup_steps);
        return initial_learning_rate * pow(decay_rate, static_cast<double>(step - warmup_steps) / decay_steps);
    }
};