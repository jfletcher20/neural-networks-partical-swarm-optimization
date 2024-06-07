#include "csv-reader.h"
#include "csv-writer.h"

#include "NeuronskaMreza.cpp"

using namespace std;

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
