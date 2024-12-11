#pragma once
#include <any>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

any str_to_any(const string &str) { // Convert string to int, double, or string
    try {
        size_t pos; // Number of characters processed
        int int_value = stoi(str, &pos);
        if (pos == str.size()) return int_value;
    } catch (invalid_argument) {}
    try {
        size_t pos;
        double double_value = stod(str, &pos);
        if (pos == str.size()) return double_value;
    } catch (invalid_argument) {}
    return str;
}

pair<vector<vector<any>>, vector<any>> Xy_from_csv(const string &file_path, bool skip_header = true) {
    vector<vector<any>> X_raw;
    vector<any> y_raw;
    unordered_map<string, int> class_to_index = {};
    int class_index = 0;

    ifstream file(file_path);
    string line, cell_value;
    if (skip_header) getline(file, line); // Skip header

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream line_stream(line);
        vector<any> row;

        // Auto-detect the data type of each cell and store it in the row
        while (getline(line_stream, cell_value, ','))
            row.push_back(str_to_any(cell_value));
        X_raw.emplace_back(row.begin(), row.end() - 1);

        if (row.back().type() == typeid(string)) {
            string class_label = any_cast<string>(row.back());
            if (class_to_index.find(class_label) == class_to_index.end()) // Not found
                class_to_index[class_label] = class_index++;
            y_raw.push_back(class_to_index[class_label]);
        } else y_raw.push_back(row.back());
    }
    return {X_raw, y_raw};
}