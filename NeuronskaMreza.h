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

class NeuronskaMrezna {
private:
    string dataset_naziv;
    double rasponMin, rasponMax;
    int brojUlaznih, brojSkrivenih, brojIzlaznih;
    unsigned int brojSkrivenihSlojeva;
    vector<vector<double>> skriveniBiasevi, ulazSkriveneTezine, skriveneIzlazTezine, skrivenoIzlazi;
    vector<vector<vector<double>>> skriveniTezine;
    vector<double> ulazi, izlazi, izlazniBiasevi;

    vector<vector<double>> KreirajMatricu(int rows, int stupci);
    double TanH(double x);
    vector<double> Softmax(const vector<double>& zbrojIzlazni);
    double SrednjaKvadratnaPogreska(const vector<vector<double>>& podaciZaTreniranje, const vector<double>& tezine);
    void Izmijesaj(vector<int>& niz, mt19937& rnd);

public:
    NeuronskaMrezna(int brojUlaznih, int brojSkrivenih, unsigned int brojSkrivenihSlojeva, int brojIzlaznih, string dataset_naziv, double rasponMin, double rasponMax);

    void SetTezine(const vector<double>& tezine);
    vector<double> GetTezine() const;
    vector<double> IzracunajIzlaze(const vector<double>& xVrijednosti);
    vector<double> Treniraj(const vector<vector<double>>& podaciZaTreniranje, int brojCestica, int maxEpoha, double zadovoljavajucaGreska);
    double Tocnost(const vector<vector<double>>& podaci, int vrstaPodataka, map<string, int> testnoPokretanje);
    int brojTezina;
};