#pragma once
#include <any>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

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

        while (getline(line_stream, cell_value, ',')) {
            try {
                row.push_back(stod(cell_value));
            } catch (const invalid_argument &) {
                row.push_back(cell_value);
            }
        }
        X_raw.insert(X_raw.end(), row.begin(), prev(row.end()));
        if (row.back().type() == typeid(string)) {
            string class_label = any_cast<string>(row.back());
            if (class_to_index.find(class_label) == class_to_index.end()) // Not found
                class_to_index[class_label] = class_index++;
            y_raw.push_back(class_to_index[class_label]);
        } else y_raw.push_back(any_cast<int>(row.back()));
    }
    return {X_raw, y_raw};
}