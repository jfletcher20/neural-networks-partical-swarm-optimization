#include "csv-reader.h"
#include "csv-writer.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include <string>
#include <cmath>
#include <map>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int numInput, int numHidden, unsigned int numHiddenLayers, int numOutput, string dataset_name, double rangeMin, double rangeMax);

    void SetWeights(const vector<double>& weights);
    vector<double> GetWeights() const;
    vector<double> ComputeOutputs(const vector<double>& xValues);
    vector<double> Train(const vector<vector<double>>& trainData, int numParticles, int maxEpochs, double exitError);
    double Accuracy(const vector<vector<double>>& data, int dataType, map<string, int> testRun);
    int numWeights;

private:
    string dataset_name;
    double rangeMin, rangeMax;
    int numInput, numHidden, numOutput;
    unsigned int numHiddenLayers;
    vector<vector<double>> hBiases, ihWeights, hoWeights;
    vector<vector<vector<double>>> hhWeights;
    vector<double> inputs, outputs, oBiases, hOutputs;

    vector<vector<double>> MakeMatrix(int rows, int cols);
    double HyperTanFunction(double x);
    vector<double> Softmax(const vector<double>& oSums);
    double MeanSquaredError(const vector<vector<double>>& trainData, const vector<double>& weights);
    void Shuffle(vector<int>& sequence, mt19937& rnd);
};

NeuralNetwork::NeuralNetwork(int numInput, int numHidden, unsigned int numHiddenLayers, int numOutput, string dataset_name, double rangeMin, double rangeMax)
    : numInput(numInput), numHidden(numHidden), numHiddenLayers(numHiddenLayers), numOutput(numOutput),
      inputs(numInput), hOutputs(numHidden),
      oBiases(numOutput), outputs(numOutput), dataset_name(dataset_name), rangeMin(rangeMin), rangeMax(rangeMax) {

    numWeights = (numInput * numHidden) + (numHidden * numOutput);
    numWeights += numHiddenLayers * (numHidden * numHidden) + numHiddenLayers * numHidden + numOutput;

    ihWeights = MakeMatrix(numInput, numHidden);
    hBiases = MakeMatrix(numHiddenLayers, numHidden);

    for(int i = 0; i < numHiddenLayers; i++)
        hhWeights.push_back(MakeMatrix(numHidden, numHidden));

    hoWeights = MakeMatrix(numHidden, numOutput);
}

vector<vector<double>> NeuralNetwork::MakeMatrix(int rows, int cols) {
    vector<vector<double>> result(rows, vector<double>(cols));
    return result;
}

void NeuralNetwork::SetWeights(const vector<double>& weights) {
    if (weights.size() != numWeights)
        throw runtime_error("Bad weights array length");

    int k = 0;
    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            ihWeights[i][j] = weights[k++];
    for (int i = 0; i < numHiddenLayers; ++i)
        for (int j = 0; j < numHidden; ++j) {
            hBiases[i][j] = weights[k++];
            for (int lj = 0; lj < numHidden; ++lj) hhWeights[i][j][lj] = weights[k++];
        }
    for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
            hoWeights[i][j] = weights[k++];
    for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
}

vector<double> NeuralNetwork::GetWeights() const {
    vector<double> result(numWeights);
    int k = 0;
    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            result[k++] = ihWeights[i][j];
    for (int i = 0; i < numHiddenLayers; ++i)
        for (int j = 0; j < numHidden; ++j)
        result[k++] = hBiases[i][j];
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

    for (int i = 0; i < numHiddenLayers; ++i)
			for (int j = 0; j < numHidden; ++j)
				hSums[i] += hBiases[i][j];

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
    uniform_real_distribution<double> distPosition(rangeMin, rangeMax);
    uniform_real_distribution<double> distVelocity(-1.0, 1.0);
    uniform_real_distribution<double> distProb(0.0, 1.0);

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
        for (double& w : particle.position) w = distPosition(mt);

        particle.velocity.resize(numWeights);
        for (double& v : particle.velocity) v = distVelocity(mt);

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

double NeuralNetwork::Accuracy(const vector<vector<double>>& data, int dataType, map<string, int> testRun) {
    int numCorrect = 0;
    int numWrong = 0;

    vector<vector<int>> all_results;

    for (const auto& row : data) {
        vector<double> xValues(row.begin(), row.begin() + numInput);
        int actual = max_element(row.begin() + numInput, row.end()) - (row.begin() + numInput);
        vector<double> yValues = ComputeOutputs(xValues);
        int predicted = max_element(yValues.begin(), yValues.end()) - yValues.begin();

        vector<int> current_results;
        current_results.push_back(testRun[dataset_name]);
        current_results.push_back(actual);
        current_results.push_back(predicted);
        current_results.push_back(dataType);
        all_results.push_back(current_results);

        actual != predicted ? cout << " Krivo pogodio [" << dataset_name << "] <iteracija: " << all_results.size() << ">\t" << actual << " vs " << predicted << endl : cout << "";

        if (predicted == actual) ++numCorrect;
        else ++numWrong;
    }
    CSVWriter::writeCSV(dataset_name + "-outputs.csv", all_results, "TestRun,Actual,Predicted,Train(0)OrTest(1)");

    vector<int> numCorrectWrong;
    numCorrectWrong.push_back(testRun[dataset_name]);
    numCorrectWrong.push_back(numCorrect);
    numCorrectWrong.push_back(numWrong);
    numCorrectWrong.push_back(dataType);
    vector<vector<int>> numCorrectWrongData;
    numCorrectWrongData.push_back(numCorrectWrong);

    CSVWriter::writeCSV(dataset_name + "-distribution.csv", numCorrectWrongData, "TestRun,NumCorrect,NumWrong,Train(0)OrTest(1)");

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

void runDataset(int dataset, int iteration = 1) {
    string dataset_name = "";
    double rangeMin, rangeMax;
    map<string, int> testRun;
    switch(dataset) {
        case 1:
            dataset_name = "IRIS";
            rangeMin = -1.0;
            rangeMax = 1.0;
            break;
        case 2:
            dataset_name = "PENGUINS";
            rangeMin = -106.0;
            rangeMax = 106.0;
            break;
        case 3:
            dataset_name = "MINES";
            rangeMin = -5.0;
            rangeMax = 5.0;
            break;
        default:
            dataset_name = "ERROR";
            break;
    }
    cout << " << Training neural network on dataset " << dataset_name << endl;

    string file_name = dataset_name + ".csv";
    testRun.insert({ dataset_name, iteration });

    /* Neki su datasetovi sortirani, sto nije pogodno za treniranje neuronske mreže.
     *
     * Npr.: IRIS dataset po redu ima 50 Iris-setosa, 50 versicolor, te 50 virginica
     * To nije pogodno za treniranje jer se uzimaju prvih 50% podataka za treniranje,
     * te će neuronska mreža imati više podataka za setosa nego za ostale klase, ovisno
     * o količini podataka koje uzmemo.
     * 
     * Stoga je potrebno pomiješati podatke, ili ih rasporediti tako da se svaka klasa
     * naizmjenice pojavljuje
    */
    // stvara probleme kod višedretvenosti pa je isključeno nakon prvog pokretanja
    // CSVReader::shuffleCSV(file_name); // iskljuciti u slucaju da je dataset vec pomijesan

    vector<vector<double>> trainData = CSVReader::readFirstNRows(file_name, CSVReader::numRows(file_name) * .5);
    vector<vector<double>> testData = CSVReader::readFromRowN(file_name, CSVReader::numRows(file_name) * .5);

    try {
        
        NeuralNetwork neuralNetwork(4, 6, 5, 3, dataset_name, rangeMin, rangeMax);

        cout << " [" << dataset_name << "]\tTraining the neural network..." << endl << endl;
        vector<double> best = neuralNetwork.Train(trainData, 10, 1200, 0.05);

        cout << endl << " [" << dataset_name << "]\tTraining finished" << endl << endl;
        neuralNetwork.SetWeights(best);

        cout << " [" << dataset_name << "]\tTesting training accuracy..." << endl << endl;
        double trainAcc = neuralNetwork.Accuracy(trainData, 0, testRun);

        cout << endl << " [" << dataset_name << "]\tTesting test accuracy..." << endl << endl;
        double testAcc = neuralNetwork.Accuracy(testData, 1, testRun);
        cout << endl;

        cout << " [" << dataset_name << "]\tTraining accuracy = " << trainAcc << endl;
        cout << " [" << dataset_name << "]\tTest accuracy = " << testAcc << endl;

        testRun[dataset_name]++;

    } catch (const exception& ex) {
        cout << ex.what() << endl;
    }
}

int main() {
    cout << "\nSeminar: Implementacija neuronske mreze nad podacima izabranog dataseta, koristeci PSO.\n";
    cout << "Autori: Joshua Lee Fletcher, Noa Midzic, Marko Novak\n";

    int dataset = 0;

    while (true) {
        cout << endl << " >---------------------------------------[ >- NN+PSO -< ]---------------------------------------<" << endl << endl;
        cout << endl << "  Odaberite koji dataset zelite koristiti (1, 2, 3) ili 0 za izlaz, ili 4 za ponovljene testove: " << endl << endl;
        cout << "\t0 - Izlaz" << endl;
        cout << "\t1 - IRIS.csv" << endl;
        cout << "\t2 - PENGUINS.csv" << endl;
        cout << "\t3 - MINES.csv" << endl;
        cout << "\t4 - Ponovljeni testovi" << endl;
        cout << endl << " >> "; cin >> dataset; cout << endl;

        if (dataset == 0) break;
        if (!dataset || dataset > 4) {
            cout << "Nevazeci unos, pokusajte ponovo." << endl;
            continue;
        }
        if (dataset == 4) {
            int numTestRuns = 0;
            cout << "Unesite broj ponovljenih testova: ";
            cin >> numTestRuns;
            cout << endl;

            vector<thread> threads;
            for (int i = 0; i < numTestRuns; ++i)
                for (int j = 1; j <= 3; ++j) {
                    threads.push_back(thread(runDataset, j, i + 1));
                    // Podaci se u datoteku upisuju prebrzo. Posljedica toga je spajanje redaka ili preskakanje redaka
                    sleep(2); // 2s je dovoljno da svaka dretva jedna za drugom pravilno upisuje podatke u .csv, ali
                    // je dobro svejedno provjeravati za svaki slučaj.
                }
            for (auto& t : threads) t.join();
            
        } else runDataset(dataset);
    }

	return 0;
}
