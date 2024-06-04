#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

#define PATH "./results/"

class CSVWriter {
public:
    static void writeCSV(const std::string& file_name, std::vector<vector<int>> data, string header) {
        std::string filename = PATH + file_name;
        std::ofstream file;
        file.open (filename);
        
        if (file.is_open()) {
            file << header << endl;
            for (const auto& row : data) {
                std::stringstream row_stream;
                bool first_value = true;
                for (const auto& value : row) {
                    if (!first_value) {
                        row_stream << ",";
                    }
                    row_stream << value;
                    first_value = false;
                }

                file << row_stream.str() << std::endl;
            }
            file.close();
            } else {
            std::cerr << "Error opening file: " << filename << std::endl;
            }
        }
};
