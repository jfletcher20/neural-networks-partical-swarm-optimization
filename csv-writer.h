#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

#define WRITE_PATH "./results/"

class CSVWriter {
public:
    static void writeCSV(const std::string& file_name, std::vector<vector<int>> data, string header) {
        std::string filename = WRITE_PATH + file_name;
        std::ofstream file;

        bool writeHeader = false;

        std::ifstream infile(filename);
        if (!infile.good() || infile.peek() == std::ifstream::traits_type::eof()) writeHeader = true;
        infile.close();

        file.open(filename, ios_base::app);
        
        if (file.is_open()) {
            if (writeHeader) file << header << endl;
            for (const auto& row : data) {
                std::stringstream row_stream;
                bool first_value = true;
                for (const auto& value : row) {
                    if (!first_value) row_stream << ",";
                    row_stream << value;
                    first_value = false;
                }
                file << row_stream.str() << std::endl;
            }
            file.close();
        } else std::cerr << "Error opening file: " << filename << std::endl;
    }
    
};
