#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
using namespace std;
using namespace chrono;

class TrainingLogger {
private:
    int bar_width = 30;
    bool display_lr = false;
    const int epochs, steps_per_epoch;
    steady_clock::time_point train_train_time, epoch_start_time, batch_start_time;

    string format_time(int seconds) {
        int hours = seconds / 3600, minutes = (seconds % 3600) / 60;
        stringstream ss;
        if (hours > 0) ss << setfill('0') << setw(2) << hours << "h";
        ss << setfill('0') << setw(2) << minutes << (hours > 0 ? "m" : ":");
        ss << setfill('0') << setw(2) << seconds % 60 << (hours > 0 ? "s" : "");
        return ss.str();
    }

    string get_system_time() {
        system_clock::time_point now = system_clock::now();
        time_t time = system_clock::to_time_t(now);
        stringstream ss;
        ss << put_time(localtime(&time), "%H:%M:%S");
        return ss.str();
    }

    string get_elapsed_time(steady_clock::time_point start_time) {
        steady_clock::time_point now = steady_clock::now();
        int elapsed = duration_cast<seconds>(now - start_time).count();
        return format_time(elapsed);
    }

    string get_eta(int batch_step) {
        steady_clock::time_point now = steady_clock::now();
        int batch_duration = duration_cast<seconds>(now - batch_start_time).count();
        int estimated_remaining = (steps_per_epoch - batch_step) * batch_duration;
        return format_time(estimated_remaining);
    }

    string create_progress_bar(int batch_step) {
        float progress = (float)batch_step / steps_per_epoch;
        int pos = bar_width * progress;

        stringstream ss;
        ss << int(progress * 100.0) << "%|";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) ss << "=";
            else if (i == pos) ss << ">";
            else ss << "-"; // Remaining space
        }
        ss << "|";
        return ss.str();
    }

public:
    TrainingLogger(int epochs, int steps_per_epoch) : epochs(epochs), steps_per_epoch(steps_per_epoch) {
        train_train_time = steady_clock::now();
    }
    void end_training() {
        cout << "Training finished in " << get_elapsed_time(train_train_time) << endl
             << "System time: " << get_system_time() << endl;
    }
    void set_bar_width(int bar_width) { this->bar_width = bar_width; }
    void set_display_lr(bool display_lr) { this->display_lr = display_lr; }
    void start_epoch() { epoch_start_time = steady_clock::now(); }
    void start_batch() { batch_start_time = steady_clock::now(); }

    const void log_progress(int epoch, int batch_step, double loss, const unordered_map<string, double> &metrics = {}, double learning_rate = -1) {
        cout << "\r"; // Clear the current line
        cout << "Epoch " << epoch << "/" << epochs << ": ";
        cout << create_progress_bar(batch_step) << " " << batch_step << "/" << steps_per_epoch;
        cout << " [" << get_elapsed_time(epoch_start_time) << "<" << get_eta(batch_step) << "]";

        cout << ", loss=" << fixed << setprecision(4) << loss;
        for (const auto &[name, value] : metrics)
            cout << ", " << name << "=" << fixed << setprecision(4) << value;

        if (display_lr && learning_rate > 0)
            cout << ", learning_rate=" << scientific << setprecision(3) << learning_rate;

        // Flush without newline except for the last step
        if (batch_step >= steps_per_epoch) cout << endl;
        cout << flush;
    }
};