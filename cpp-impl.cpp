#include "csv-reader.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

void ShowVector(const vector<double>& vector, int valsPerRow, int decimals, bool newLine) {
    for (size_t i = 0; i < vector.size(); ++i) {
        if (i % valsPerRow == 0) cout << endl;
        cout << fixed << setprecision(decimals) << vector[i] << " ";
    }
    if (newLine) cout << endl;
}

void ShowMatrix(const vector<vector<double>>& matrix, int decimals, bool newLine) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        cout << setw(3) << i << ": ";
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            cout << fixed << setprecision(decimals) << matrix[i][j] << " ";
        }
        cout << endl;
    }
    if (newLine) cout << endl;
}

class NeuralNetwork {
public:
    NeuralNetwork(int numInput, int numHidden, int numOutput);

    void SetWeights(const vector<double>& weights);
    vector<double> GetWeights() const;
    vector<double> ComputeOutputs(const vector<double>& xValues);
    vector<double> Train(const vector<vector<double>>& trainData, int numParticles, int maxEpochs, double exitError);
    double Accuracy(const vector<vector<double>>& data);

private:
    int numInput;
    int numHidden;
    int numOutput;
    vector<double> inputs;
    vector<vector<double>> ihWeights; 
    vector<double> hBiases;
    vector<double> hOutputs;
    vector<vector<double>> hoWeights; 
    vector<double> oBiases;
    vector<double> outputs;

    vector<vector<double>> MakeMatrix(int rows, int cols);
    double HyperTanFunction(double x);
    vector<double> Softmax(const vector<double>& oSums);
    double MeanSquaredError(const vector<vector<double>>& trainData, const vector<double>& weights);
    void Shuffle(vector<int>& sequence, mt19937& rnd);
};


NeuralNetwork::NeuralNetwork(int numInput, int numHidden, int numOutput)
    : numInput(numInput), numHidden(numHidden), numOutput(numOutput),
      inputs(numInput), hBiases(numHidden), hOutputs(numHidden),
      oBiases(numOutput), outputs(numOutput) {
    ihWeights = MakeMatrix(numInput, numHidden);
    hoWeights = MakeMatrix(numHidden, numOutput);
}

vector<vector<double>> NeuralNetwork::MakeMatrix(int rows, int cols) {
    vector<vector<double>> result(rows, vector<double>(cols));
    return result;
}

void NeuralNetwork::SetWeights(const vector<double>& weights) {
    int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    if (weights.size() != numWeights)
        throw runtime_error("Bad weights array length");

    int k = 0;
    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            ihWeights[i][j] = weights[k++];
    for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
    for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
            hoWeights[i][j] = weights[k++];
    for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
}

vector<double> NeuralNetwork::GetWeights() const {
    int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    vector<double> result(numWeights);
    int k = 0;
    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            result[k++] = ihWeights[i][j];
    for (int i = 0; i < numHidden; ++i)
        result[k++] = hBiases[i];
    for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
            result[k++] = hoWeights[i][j];
    for (int i = 0; i < numOutput; ++i)
        result[k++] = oBiases[i];
    return result;
}

vector<double> NeuralNetwork::ComputeOutputs(const vector<double>& xValues) {
    if (xValues.size() != numInput)
        throw runtime_error("Bad xValues array length");

    vector<double> hSums(numHidden, 0.0);
    vector<double> oSums(numOutput, 0.0);

    for (size_t i = 0; i < xValues.size(); ++i)
        inputs[i] = xValues[i];

    for (int j = 0; j < numHidden; ++j)
        for (int i = 0; i < numInput; ++i)
            hSums[j] += inputs[i] * ihWeights[i][j];

    for (int i = 0; i < numHidden; ++i)
        hSums[i] += hBiases[i];

    for (int i = 0; i < numHidden; ++i)
        hOutputs[i] = HyperTanFunction(hSums[i]);

    for (int j = 0; j < numOutput; ++j)
        for (int i = 0; i < numHidden; ++i)
            oSums[j] += hOutputs[i] * hoWeights[i][j];

    for (int i = 0; i < numOutput; ++i)
        oSums[i] += oBiases[i];

		vector<double> softOut = Softmax(oSums);
    copy(softOut.begin(), softOut.end(), outputs.begin());

    return outputs;
}

double NeuralNetwork::HyperTanFunction(double x) {
    if (x < -20.0) return -1.0;
    else if (x > 20.0) return 1.0;
    else return tanh(x);
}

vector<double> NeuralNetwork::Softmax(const vector<double>& oSums) {
    double max = *max_element(oSums.begin(), oSums.end());
    double scale = 0.0;
    for (double val : oSums)
        scale += exp(val - max);

    vector<double> result(oSums.size());
    for (size_t i = 0; i < oSums.size(); ++i)
        result[i] = exp(oSums[i] - max) / scale;

    return result;
}

double NeuralNetwork::MeanSquaredError(const vector<vector<double>>& trainData, const vector<double>& weights) {
    SetWeights(weights);
    double sumSquaredError = 0.0;
    for (const auto& row : trainData) {
        vector<double> xValues(row.begin(), row.begin() + numInput);
        vector<double> tValues(row.begin() + numInput, row.end());
        vector<double> yValues = ComputeOutputs(xValues);
        for (int j = 0; j < tValues.size(); ++j) {
            sumSquaredError += pow(yValues[j] - tValues[j], 2);
        }
    }
    return sumSquaredError / trainData.size();
}

void NeuralNetwork::Shuffle(vector<int>& sequence, mt19937& rnd) {
    shuffle(sequence.begin(), sequence.end(), rnd);
}

vector<double> NeuralNetwork::Train(const vector<vector<double>>& trainData, int numParticles, int maxEpochs, double exitError) {
    mt19937 mt(time(nullptr));
    uniform_real_distribution<double> distPosition(-5.0, 5.0);
    uniform_real_distribution<double> distVelocity(-1.0, 1.0);
    uniform_real_distribution<double> distProb(0.0, 1.0);

    int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    vector<double> bestGlobalPosition(numWeights, 0.0);
    double bestGlobalError = numeric_limits<double>::max();

    struct Particle {
        vector<double> position;
        double error;
        vector<double> velocity;
        vector<double> bestPosition;
        double bestError;
    };

    vector<Particle> swarm(numParticles);
    for (auto& particle : swarm) {
        particle.position.resize(numWeights);
        for (double& w : particle.position) {
            w = distPosition(mt);
        }
        particle.velocity.resize(numWeights);
        for (double& v : particle.velocity) {
            v = distVelocity(mt);
        }
        particle.bestPosition = particle.position;
        particle.error = MeanSquaredError(trainData, particle.position);
        particle.bestError = particle.error;
        if (particle.error < bestGlobalError) {
            bestGlobalError = particle.error;
            bestGlobalPosition = particle.position;
        }
    }

    double w = 0.729;
    double c1 = 1.49445;
    double c2 = 1.49445;
    vector<int> sequence(numParticles);
    iota(sequence.begin(), sequence.end(), 0);

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        if (bestGlobalError < exitError) break;

        mt19937 rnd(time(NULL));
        Shuffle(sequence, rnd);
        for (int i : sequence) {
            Particle& currP = swarm[i];
            for (int j = 0; j < numWeights; ++j) {
                double r1 = distProb(mt);
                double r2 = distProb(mt);
                currP.velocity[j] = (w * currP.velocity[j]) +
                                    (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                                    (c2 * r2 * (bestGlobalPosition[j] - currP.position[j]));
                currP.position[j] += currP.velocity[j];
            }

            currP.error = MeanSquaredError(trainData, currP.position);
            if (currP.error < currP.bestError) {
                currP.bestError = currP.error;
                currP.bestPosition = currP.position;
            }

            if (currP.error < bestGlobalError) {
                bestGlobalError = currP.error;
                bestGlobalPosition = currP.position;
            }

            
        }
    }
    SetWeights(bestGlobalPosition);
    return bestGlobalPosition;
}

double NeuralNetwork::Accuracy(const vector<vector<double>>& data) {
    int numCorrect = 0;
    int numWrong = 0;
    for (const auto& row : data) {
        vector<double> xValues(row.begin(), row.begin() + numInput);
        int actual = max_element(row.begin() + numInput, row.end()) - (row.begin() + numInput);
        vector<double> yValues = ComputeOutputs(xValues);
        int predicted = max_element(yValues.begin(), yValues.end()) - yValues.begin();
        cout << (actual == predicted ? "  \t" : " X\t") << actual << " vs " << predicted << endl;
        if (predicted == actual) {
            ++numCorrect;
        } else {
            ++numWrong;
        }
    }
    return static_cast<double>(numCorrect) / (numCorrect + numWrong);
}


class Particle {
public:
    Particle(const vector<double>& position, double error, const vector<double>& velocity, const vector<double>& bestPosition, double bestError);

    vector<double> position;
    double error;
    vector<double> velocity;
    vector<double> bestPosition;
    double bestError;
};

Particle::Particle(const vector<double>& position, double error, const vector<double>& velocity, const vector<double>& bestPosition, double bestError)
    : position(position), error(error), velocity(velocity), bestPosition(bestPosition), bestError(bestError) {}

int main() {

    cout << "\nSeminar: Implementacija neuronske mreže nad podacima iz IRIS.csv, koristeći PSO.\n";
    cout << "Autori: Joshua Lee Fletcher, Noa Midzic, Marko Novak\n";

    string file_name = "";
    int dataset = 0;
    cout << " << Odaberite koji dataset želite koristiti (1, 2): " << endl;
    cout << "\t1 - IRIS.csv" << endl;
    cout << "\t2 - PENGUINS.csv" << endl;
    cout << "\t3 - MINES.csv" << endl;
    cout << endl << " >> "; cin >> dataset; cout << endl;

    if (!dataset || dataset > 3) return -1;
    switch(dataset) {
        case 1: file_name = "IRIS.csv"; break;
        case 2: file_name = "PENGUINS.csv"; break;
        case 3: file_name = "MINES.csv"; break;
        default: file_name = "ERROR"; break;
    }

    // originalni IRIS dataset je po redu prvih 50 redova Iris-setosa, zatim 50 redova Iris-versicolor, te na kraju 50 redova Iris-virginica
    // to nije pogodno za treniranje jer će neuronska mreža imati više podataka za Iris-setosa nego za Iris-versicolor i Iris-virginica ovisno o količini podataka koje uzmemo
    // stoga je potrebno pomiješati podatke, ili ih rasporediti tako da se svaka klasa naizmjenice pojavljuje
    IRISReader::shuffleCSV(file_name);
    
    vector<vector<double>> trainData = IRISReader::readFirstNRows(file_name, IRISReader::numRows(file_name) * .5);
    vector<vector<double>> testData = IRISReader::readFromRowN(file_name, IRISReader::numRows(file_name) * .5);

    try {
	    NeuralNetwork neuralNetwork(4, 6, 3);
        cout << "Training the neural network..." << endl << endl;
	    vector<double> best = neuralNetwork.Train(trainData, 6, 1200, 0.05);
	
	    ShowVector(best, 10, 3, true);
	
        cout << endl << "Training finished" << endl << endl;
	    neuralNetwork.SetWeights(best);

        cout << "Testing training accuracy..." << endl << endl;
	    double trainAcc = neuralNetwork.Accuracy(trainData);
	
        cout << endl << "Testing test accuracy..." << endl << endl;
	    double testAcc = neuralNetwork.Accuracy(testData);
        cout << endl;
	
        cout << "Training accuracy = " << trainAcc << endl;
        cout << "Test accuracy = " << testAcc << endl;
    } catch (const exception& ex) {
        cout << ex.what() << endl;
    }

    return 0;
}
