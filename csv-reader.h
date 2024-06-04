#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#define PATH "./datasets/"

class IRISReader {
public:
    static void shuffleCSV(const std::string& file_name) {
        std::string filename = PATH + file_name;
        std::ifstream file(filename);
        std::vector<std::string> lines;
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) lines.push_back(line);
            file.close();
            std::random_shuffle(lines.begin(), lines.end());
            std::ofstream outFile(filename);
            if (outFile.is_open()) {
                for (const auto& l : lines)
                    outFile << l << std::endl;
                outFile.close();
                std::cout << "File shuffled and saved successfully." << std::endl;
            } else std::cerr << "Error: Unable to open file for writing." << std::endl;
        } else std::cerr << "Error: Unable to open file for reading." << std::endl;
    }

    static std::vector<std::vector<double>> readCSV(const std::string& file_name) {
        std::string filename = PATH + file_name;
        std::ifstream file(filename);
        std::vector<std::vector<double>> data;
        if (file.is_open()) {
            std::string header;
            std::getline(file, header);
            std::string line;
            while (std::getline(file, line)) {
                std::vector<double> row;
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    if (cell == "Iris-setosa" || cell == "Adelie") {
                        row.push_back(0);
                    } else if (cell == "Iris-versicolor" || cell == "Chinstrap") {
                        row.push_back(1);
                    } else if (cell == "Iris-virginica" || cell == "Gentoo") {
                        row.push_back(2);
                    } else {
                        row.push_back(std::stod(cell));
                    }
                }
                data.push_back(row);
            }
            file.close();
        }
        return data;
    }

    static int numRows(const std::string& file_name) {
        std::string filename = PATH + file_name;
        int count = 0;
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) count++;
            file.close();
        }
        return count - 1;
    }

    static int numColumns(const std::string& file_name) {
        std::string filename = PATH + file_name;
        int count = 0;
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) count++;
            file.close();
        }
        return count;
    }

    static int numClasses(const std::string& file_name) {
        std::string filename = PATH + file_name;
        return 3;
    }

    static std::vector<std::vector<double>> readFirstNRows(const std::string& file_name, int n) {
        std::string filename = PATH + file_name;
        std::ifstream file(filename);
        std::vector<std::vector<double>> data;
        if (file.is_open()) {
            std::string header;
            std::getline(file, header);
            std::string line;
            int count = 0;
            while (std::getline(file, line) && count < n) {
                std::vector<double> row;
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    if (cell == "Iris-setosa" || cell == "Adelie") {
                        row.push_back(0);
                    } else if (cell == "Iris-versicolor" || cell == "Chinstrap") {
                        row.push_back(1);
                    } else if (cell == "Iris-virginica" || cell == "Gentoo") {
                        row.push_back(2);
                    } else {
                        row.push_back(std::stod(cell));
                    }
                }
                data.push_back(row);
                count++;
            }
            file.close();
        }
        return data;
    }

    static std::vector<std::vector<double>> readFromRowN(const std::string& file_name, int n) {
        std::string filename = PATH + file_name;
        std::ifstream file(filename);
        std::vector<std::vector<double>> data;
        if (file.is_open()) {
            for (int i = 0; i < n - 1; ++i) {
                std::string line;
                std::getline(file, line);
            }
            std::string line;
            while (std::getline(file, line)) {
                std::vector<double> row;
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    if (cell == "Iris-setosa" || cell == "Adelie") {
                        row.push_back(0);
                    } else if (cell == "Iris-versicolor" || cell == "Chinstrap") {
                        row.push_back(1);
                    } else if (cell == "Iris-virginica" || cell == "Gentoo") {
                        row.push_back(2);
                    } else {
                        row.push_back(std::stod(cell));
                    }
                }
                data.push_back(row);
            }
            file.close();
        }
        return data;
    }

};
