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

class NeuronskaMrezna {
public:
    NeuronskaMrezna(int brojUlaznih, int brojSkrivenih, unsigned int brojSkrivenihSlojeva, int brojIzlaznih, string dataset_naziv, double rasponMin, double rasponMax);

    void SetTezine(const vector<double>& tezine);
    vector<double> GetTezine() const;
    vector<double> IzracunajIzlaze(const vector<double>& xVrijednosti);
    vector<double> Treniraj(const vector<vector<double>>& podaciZaTreniranje, int brojCestica, int maxEpoha, double zadovoljavajucaGreska);
    double Tocnost(const vector<vector<double>>& podaci, int vrstaPodataka, map<string, int> testnoPokretanje);
    int brojTezina;

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
};

NeuronskaMrezna::NeuronskaMrezna(int brojUlaznih, int brojSkrivenih, unsigned int brojSkrivenihSlojeva, int brojIzlaznih, string dataset_naziv, double rasponMin, double rasponMax)
    : brojUlaznih(brojUlaznih), brojSkrivenih(brojSkrivenih), brojSkrivenihSlojeva(brojSkrivenihSlojeva), brojIzlaznih(brojIzlaznih),
      ulazi(brojUlaznih), skrivenoIzlazi(brojSkrivenihSlojeva, vector<double>(brojSkrivenih)),
      izlazniBiasevi(brojIzlaznih), izlazi(brojIzlaznih), dataset_naziv(dataset_naziv), rasponMin(rasponMin), rasponMax(rasponMax) {

    brojTezina = (brojUlaznih * brojSkrivenih) + (brojSkrivenihSlojeva - 1) * (brojSkrivenih * brojSkrivenih) + (brojSkrivenih * brojIzlaznih);
    brojTezina += brojSkrivenihSlojeva * brojSkrivenih + brojIzlaznih;

    ulazSkriveneTezine = KreirajMatricu(brojUlaznih, brojSkrivenih);
    skriveniBiasevi = KreirajMatricu(brojSkrivenihSlojeva, brojSkrivenih);
    for (unsigned int i = 0; i < brojSkrivenihSlojeva - 1; ++i)
        skriveniTezine.push_back(KreirajMatricu(brojSkrivenih, brojSkrivenih));

    skriveneIzlazTezine = KreirajMatricu(brojSkrivenih, brojIzlaznih);
}

vector<vector<double>> NeuronskaMrezna::KreirajMatricu(int redovi, int stupci) {
    vector<vector<double>> rezultat(redovi, vector<double>(stupci));
    return rezultat;
}

void NeuronskaMrezna::SetTezine(const vector<double>& tezine) {
    if (tezine.size() != brojTezina)
        throw runtime_error("Neispravna duljina polja tezina");

    int k = 0;
    for (int i = 0; i < brojUlaznih; ++i)
        for (int j = 0; j < brojSkrivenih; ++j)
            ulazSkriveneTezine[i][j] = tezine[k++];

    for (unsigned int l = 0; l < brojSkrivenihSlojeva - 1; ++l)
        for (int i = 0; i < brojSkrivenih; ++i)
            for (int j = 0; j < brojSkrivenih; ++j)
                skriveniTezine[l][i][j] = tezine[k++];

    for (unsigned int l = 0; l < brojSkrivenihSlojeva; ++l)
        for (int i = 0; i < brojSkrivenih; ++i)
            skriveniBiasevi[l][i] = tezine[k++];

    for (int i = 0; i < brojSkrivenih; ++i)
        for (int j = 0; j < brojIzlaznih; ++j)
            skriveneIzlazTezine[i][j] = tezine[k++];

    for (int i = 0; i < brojIzlaznih; ++i)
        izlazniBiasevi[i] = tezine[k++];
}

vector<double> NeuronskaMrezna::GetTezine() const {
    vector<double> rezultat(brojTezina);
    int k = 0;
    for (int i = 0; i < brojUlaznih; ++i)
        for (int j = 0; j < brojSkrivenih; ++j)
            rezultat[k++] = ulazSkriveneTezine[i][j];
    for (int i = 0; i < brojSkrivenihSlojeva; ++i)
        for (int j = 0; j < brojSkrivenih; ++j)
        rezultat[k++] = skriveniBiasevi[i][j];
    for (int i = 0; i < brojSkrivenih; ++i)
        for (int j = 0; j < brojIzlaznih; ++j)
            rezultat[k++] = skriveneIzlazTezine[i][j];
    for (int i = 0; i < brojIzlaznih; ++i)
        rezultat[k++] = izlazniBiasevi[i];
    return rezultat;
}

vector<double> NeuronskaMrezna::IzracunajIzlaze(const vector<double>& xVrijednosti) {
    if (xVrijednosti.size() != brojUlaznih)
        throw runtime_error("Neispravna duljina polja xVrijednosti");

    vector<double> zbrojSkriveni(brojSkrivenih, 0.0);
    vector<double> zbrojIzlazni(brojIzlaznih, 0.0);

    for (int i = 0; i < brojUlaznih; ++i)
        ulazi[i] = xVrijednosti[i];

    // input za prvi skriveni sloj
    for (int j = 0; j < brojSkrivenih; ++j)
        for (int i = 0; i < brojUlaznih; ++i)
            zbrojSkriveni[j] += ulazi[i] * ulazSkriveneTezine[i][j];

    for (int j = 0; j < brojSkrivenih; ++j)
        zbrojSkriveni[j] += skriveniBiasevi[0][j];

    for (int i = 0; i < brojSkrivenih; ++i)
        skrivenoIzlazi[0][i] = TanH(zbrojSkriveni[i]);

    // prijenos tezina sa skrivenog sloja na skriveni sloj
    for (unsigned int l = 1; l < brojSkrivenihSlojeva; ++l) {
        fill(zbrojSkriveni.begin(), zbrojSkriveni.end(), 0.0);
        for (int j = 0; j < brojSkrivenih; ++j)
            for (int i = 0; i < brojSkrivenih; ++i)
                zbrojSkriveni[j] += skrivenoIzlazi[l-1][i] * skriveniTezine[l-1][i][j];

        for (int j = 0; j < brojSkrivenih; ++j)
            zbrojSkriveni[j] += skriveniBiasevi[l][j];

        for (int i = 0; i < brojSkrivenih; ++i)
            skrivenoIzlazi[l][i] = TanH(zbrojSkriveni[i]);
    }

    // prijenos tezina sa skrivenog sloja na izlazni sloj
    for (int j = 0; j < brojIzlaznih; ++j)
        for (int i = 0; i < brojSkrivenih; ++i)
            zbrojIzlazni[j] += skrivenoIzlazi[brojSkrivenihSlojeva-1][i] * skriveneIzlazTezine[i][j];

    for (int i = 0; i < brojIzlaznih; ++i)
        zbrojIzlazni[i] += izlazniBiasevi[i];

    vector<double> softRez = Softmax(zbrojIzlazni);
    copy(softRez.begin(), softRez.end(), izlazi.begin());

    return izlazi;
}

double NeuronskaMrezna::TanH(double x) {
    if (x < -20.0) return -1.0;
    else if (x > 20.0) return 1.0;
    else return tanh(x);
}

vector<double> NeuronskaMrezna::Softmax(const vector<double>& zbrojIzlazni) {
    double max = *max_element(zbrojIzlazni.begin(), zbrojIzlazni.end());
    double skaliranje = 0.0;
    for (double zI : zbrojIzlazni)
        skaliranje += exp(zI - max);

    vector<double> rezultat(zbrojIzlazni.size());
    for (size_t i = 0; i < zbrojIzlazni.size(); ++i)
        rezultat[i] = exp(zbrojIzlazni[i] - max) / skaliranje;

    return rezultat;
}

double NeuronskaMrezna::SrednjaKvadratnaPogreska(const vector<vector<double>>& podaciZaTreniranje, const vector<double>& tezine) {
    SetTezine(tezine);
    double zbrojKvadrataGreski = 0.0;
    for (const auto& row : podaciZaTreniranje) {
        vector<double> xVrijednosti(row.begin(), row.begin() + brojUlaznih);
        vector<double> tVrijednosti(row.begin() + brojUlaznih, row.end());
        vector<double> yVrijednosti = IzracunajIzlaze(xVrijednosti);
        for (int j = 0; j < tVrijednosti.size(); ++j) {
            zbrojKvadrataGreski += pow(yVrijednosti[j] - tVrijednosti[j], 2);
        }
    }
    return zbrojKvadrataGreski / podaciZaTreniranje.size();
}

void NeuronskaMrezna::Izmijesaj(vector<int>& niz, mt19937& rnd) {
    shuffle(niz.begin(), niz.end(), rnd);
}

vector<double> NeuronskaMrezna::Treniraj(const vector<vector<double>>& podaciZaTreniranje, int brojCestica, int maxEpoha, double zadovoljavajucaGreska) {
    mt19937 mt(time(nullptr));
    uniform_real_distribution<double> distPozicija(rasponMin, rasponMax);
    uniform_real_distribution<double> distBrzina(-1.0, 1.0);
    uniform_real_distribution<double> distVjerojatnost(0.0, 1.0);

    vector<double> najboljaGlobalnaPozicija(brojTezina, 0.0);
    double najboljaGlobalnaGreska = numeric_limits<double>::max();

    struct Cestica {
        vector<double> pozicija;
        double greska;
        vector<double> brzina;
        vector<double> najboljaPozicija;
        double najboljaGreska;
    };

    vector<Cestica> roj(brojCestica);
    for (auto& cestica : roj) {

        cestica.pozicija.resize(brojTezina);
        for (double& w : cestica.pozicija) w = distPozicija(mt);

        cestica.brzina.resize(brojTezina);
        for (double& v : cestica.brzina) v = distBrzina(mt);

        cestica.najboljaPozicija = cestica.pozicija;
        cestica.greska = SrednjaKvadratnaPogreska(podaciZaTreniranje, cestica.pozicija);
        cestica.najboljaGreska = cestica.greska;

        if (cestica.greska < najboljaGlobalnaGreska) {
            najboljaGlobalnaGreska = cestica.greska;
            najboljaGlobalnaPozicija = cestica.pozicija;
        }

    }

    double w = 0.729;
    double c1 = 1.49445;
    double c2 = 1.49445;
    vector<int> niz(brojCestica);
    iota(niz.begin(), niz.end(), 0);

    for (int epoha = 0; epoha < maxEpoha; ++epoha) {
        if (najboljaGlobalnaGreska < zadovoljavajucaGreska) break;

        mt19937 rnd(time(NULL));
        Izmijesaj(niz, rnd);
        for (int i : niz) {
            Cestica& currP = roj[i];
            for (int j = 0; j < brojTezina; ++j) {
                double r1 = distVjerojatnost(mt);
                double r2 = distVjerojatnost(mt);
                currP.brzina[j] = (w * currP.brzina[j]) +
                                    (c1 * r1 * (currP.najboljaPozicija[j] - currP.pozicija[j])) +
                                    (c2 * r2 * (najboljaGlobalnaPozicija[j] - currP.pozicija[j]));
                currP.pozicija[j] += currP.brzina[j];
            }

            currP.greska = SrednjaKvadratnaPogreska(podaciZaTreniranje, currP.pozicija);
            if (currP.greska < currP.najboljaGreska) {
                currP.najboljaGreska = currP.greska;
                currP.najboljaPozicija = currP.pozicija;
            }

            if (currP.greska < najboljaGlobalnaGreska) {
                najboljaGlobalnaGreska = currP.greska;
                najboljaGlobalnaPozicija = currP.pozicija;
            }

        }
    }
    SetTezine(najboljaGlobalnaPozicija);
    return najboljaGlobalnaPozicija;
}

double NeuronskaMrezna::Tocnost(const vector<vector<double>>& podaci, int vrstaPodataka, map<string, int> testnoPokretanje) {
    int numCorrect = 0;
    int numWrong = 0;

    vector<vector<int>> all_results;

    for (const auto& row : podaci) {
        vector<double> xVrijednosti(row.begin(), row.begin() + brojUlaznih);
        int actual = max_element(row.begin() + brojUlaznih, row.end()) - (row.begin() + brojUlaznih);
        vector<double> yVrijednosti = IzracunajIzlaze(xVrijednosti);
        int predicted = max_element(yVrijednosti.begin(), yVrijednosti.end()) - yVrijednosti.begin();

        vector<int> current_results;
        current_results.push_back(testnoPokretanje[dataset_naziv]);
        current_results.push_back(actual);
        current_results.push_back(predicted);
        current_results.push_back(vrstaPodataka);
        all_results.push_back(current_results);

        actual != predicted ? cout << " Krivo pogodio [" << dataset_naziv << "] <iteracija: " << all_results.size() << ">\t" << actual << " vs " << predicted << endl : cout << "";

        if (predicted == actual) ++numCorrect;
        else ++numWrong;
    }
    CSVWriter::writeCSV(dataset_naziv + "-izlazi.csv", all_results, "TestRun,Actual,Predicted,Train(0)OrTest(1)");

    vector<int> numCorrectWrong;
    numCorrectWrong.push_back(testnoPokretanje[dataset_naziv]);
    numCorrectWrong.push_back(numCorrect);
    numCorrectWrong.push_back(numWrong);
    numCorrectWrong.push_back(vrstaPodataka);
    vector<vector<int>> numCorrectWrongData;
    numCorrectWrongData.push_back(numCorrectWrong);

    CSVWriter::writeCSV(dataset_naziv + "-distribution.csv", numCorrectWrongData, "TestRun,NumCorrect,NumWrong,Train(0)OrTest(1)");

    return static_cast<double>(numCorrect) / (numCorrect + numWrong);
}

class Cestica {
public:
    Cestica(const vector<double>& pozicija, double greska, const vector<double>& brzina, const vector<double>& najboljaPozicija, double najboljaGreska);

    vector<double> pozicija;
    double greska;
    vector<double> brzina;
    vector<double> najboljaPozicija;
    double najboljaGreska;
};

Cestica::Cestica(const vector<double>& pozicija, double greska, const vector<double>& brzina, const vector<double>& najboljaPozicija, double najboljaGreska)
    : pozicija(pozicija), greska(greska), brzina(brzina), najboljaPozicija(najboljaPozicija), najboljaGreska(najboljaGreska) {}

void pokreni(int dataset, int iteracija = 1) {
    string dataset_naziv = "";
    double rasponMin, rasponMax;
    map<string, int> testnoPokretanje;
    switch(dataset) {
        case 1:
            dataset_naziv = "IRIS";
            rasponMin = -1.0;
            rasponMax = 1.0;
            break;
        case 2:
            dataset_naziv = "PENGUINS";
            rasponMin = -106.0;
            rasponMax = 106.0;
            break;
        case 3:
            dataset_naziv = "MINES";
            rasponMin = -5.0;
            rasponMax = 5.0;
            break;
        default:
            dataset_naziv = "ERROR";
            break;
    }
    cout << " << Treniranje neuronske mreze na datesetu " << dataset_naziv << endl;

    string nazivDatoteke = dataset_naziv + ".csv";
    testnoPokretanje.insert({ dataset_naziv, iteracija });

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
    // CSVReader::shuffleCSV(nazivDatoteke); // iskljuciti u slucaju da je dataset vec pomijesan

    vector<vector<double>> podaciZaTreniranje = CSVReader::readFirstNRows(nazivDatoteke, CSVReader::numRows(nazivDatoteke) * .5);
    vector<vector<double>> testniPodaci = CSVReader::readFromRowN(nazivDatoteke, CSVReader::numRows(nazivDatoteke) * .5);

    try {
        
        NeuronskaMrezna neuronskaMrezna(4, 6, 5, 3, dataset_naziv, rasponMin, rasponMax);

        cout << " [" << dataset_naziv << "]\tTreniranje neuronske mreze..." << endl << endl;
        vector<double> najbolje = neuronskaMrezna.Treniraj(podaciZaTreniranje, 10, 1200, 0.05);

        cout << endl << " [" << dataset_naziv << "]\tTreniranje je gotovo" << endl << endl;
        neuronskaMrezna.SetTezine(najbolje);

        cout << " [" << dataset_naziv << "]\tTestiranje tocnosti treninga..." << endl << endl;
        double tocnostTrening = neuronskaMrezna.Tocnost(podaciZaTreniranje, 0, testnoPokretanje);

        cout << endl << " [" << dataset_naziv << "]\tTestiranje tocnosti testiranja..." << endl << endl;
        double tocnostTestiranje = neuronskaMrezna.Tocnost(testniPodaci, 1, testnoPokretanje);
        cout << endl;

        cout << " [" << dataset_naziv << "]\tTocnost treninga = " << tocnostTrening << endl;
        cout << " [" << dataset_naziv << "]\tTocnost testiranja = " << tocnostTestiranje << endl;

        testnoPokretanje[dataset_naziv]++;

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
                    threads.push_back(thread(pokreni, j, i + 1));
                    // Podaci se u datoteku upisuju prebrzo. Posljedica toga je spajanje redaka ili preskakanje redaka
                    sleep(2); // 2s je dovoljno da svaka dretva jedna za drugom pravilno upisuje podatke u .csv, ali
                    // je dobro svejedno provjeravati za svaki slučaj.
                }
            for (auto& t : threads) t.join();
            
        } else pokreni(dataset);
    }

	return 0;
}
