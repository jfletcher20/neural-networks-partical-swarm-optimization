#include "NeuronskaMreza.h"

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

    double w = 0.4;
    double c1 = 2.1;
    double c2 = 1.6;
    vector<int> niz(brojCestica);
    iota(niz.begin(), niz.end(), 0);

    for (int epoha = 0; epoha < maxEpoha; ++epoha) {
        if (najboljaGlobalnaGreska < zadovoljavajucaGreska) break;

        mt19937 rnd(time(NULL));
        Izmijesaj(niz, rnd);
        for (int i : niz) {
            Cestica& cestica = roj[i];
            for (int j = 0; j < brojTezina; ++j) {
                double r1 = distVjerojatnost(mt);
                double r2 = distVjerojatnost(mt);
                cestica.brzina[j] = (w * cestica.brzina[j]) +
                                    (c1 * r1 * (cestica.najboljaPozicija[j] - cestica.pozicija[j])) +
                                    (c2 * r2 * (najboljaGlobalnaPozicija[j] - cestica.pozicija[j]));
                cestica.pozicija[j] += cestica.brzina[j];
            }

            cestica.greska = SrednjaKvadratnaPogreska(podaciZaTreniranje, cestica.pozicija);
            if (cestica.greska < cestica.najboljaGreska) {
                cestica.najboljaGreska = cestica.greska;
                cestica.najboljaPozicija = cestica.pozicija;
            }

            if (cestica.greska < najboljaGlobalnaGreska) {
                najboljaGlobalnaGreska = cestica.greska;
                najboljaGlobalnaPozicija = cestica.pozicija;
            }

        }
    }
    return najboljaGlobalnaPozicija;
}

double NeuronskaMrezna::Tocnost(const vector<vector<double>>& podaci, int vrstaPodataka, map<string, int> testnoPokretanje) {
    int brojTocnih = 0;
    int brojKrivih = 0;

    vector<vector<int>> sviRezultati;

    for (const auto& row : podaci) {
        vector<double> xVrijednosti(row.begin(), row.begin() + brojUlaznih);
        int stvardni = max_element(row.begin() + brojUlaznih, row.end()) - (row.begin() + brojUlaznih);
        vector<double> yVrijednosti = IzracunajIzlaze(xVrijednosti);
        int predvideni = max_element(yVrijednosti.begin(), yVrijednosti.end()) - yVrijednosti.begin();

        vector<int> trenutniRezultati;
        trenutniRezultati.push_back(testnoPokretanje[dataset_naziv]);
        trenutniRezultati.push_back(stvardni);
        trenutniRezultati.push_back(predvideni);
        trenutniRezultati.push_back(vrstaPodataka);
        sviRezultati.push_back(trenutniRezultati);

        stvardni != predvideni ? cout << " Krivo pogodio [" << dataset_naziv << "] <iteracija: " << sviRezultati.size() << ">\t" << stvardni << " vs " << predvideni << endl : cout << "";

        if (predvideni == stvardni) ++brojTocnih;
        else ++brojKrivih;
    }
    CSVWriter::writeCSV(dataset_naziv + "-izlazi.csv", sviRezultati, "TestRun,stvardni,predvideni,Train(0)OrTest(1)");

    vector<int> brojTocnihKrivih;
    brojTocnihKrivih.push_back(testnoPokretanje[dataset_naziv]);
    brojTocnihKrivih.push_back(brojTocnih);
    brojTocnihKrivih.push_back(brojKrivih);
    brojTocnihKrivih.push_back(vrstaPodataka);
    vector<vector<int>> brojTocnihKrivihPodaci;
    brojTocnihKrivihPodaci.push_back(brojTocnihKrivih);

    CSVWriter::writeCSV(dataset_naziv + "-distribution.csv", brojTocnihKrivihPodaci, "TestRun,brojTocnih,brojKrivih,Train(0)OrTest(1)");

    return static_cast<double>(brojTocnih) / (brojTocnih + brojKrivih);
}
