# NMR-INATOR: Quantum & Ensemble Spectroscopy Suite

**NMR-INATOR** è una suite di chimica computazionale e spettroscopia predittiva in Python progettata per simulare spettri **NMR (¹H, ¹³C)** e **Infrarosso (IR)** ad alta fedeltà fisica, sfruttando motori quantistici (*ab initio* / DFT con **Psi4**), regole topologiche di simmetria (**RDKit**) e campionamento conformazionale ponderato su distribuzioni di **Boltzmann** tramite integrazione con **energy-inator**.

---

## Caratteristiche Principali

### 1. Spettroscopia NMR (¹H e ¹³C)
* **Chemical Shifts Predetti**: Calcolo di spostamenti chimici accurati basati su codici HOSE (interfaccia online Zakodium / NMRDB con fallback euristico robusto offline).
* **Equivalenze Chimiche e Simmetria Topologica**: Raggruppamento automatico di nuclei simmetricamente o conformazionalmente equivalenti (es. rotazione rapida dei gruppi metilici $-\text{CH}_3$, anelli aromatici simmetrici) per generare le corrette integrazioni ($1\text{H}, 2\text{H}, 3\text{H}$, ecc.).
* **Effetti Geometrici 3D**:
  * **Accoppiamenti Scalari $^3J_{\text{HH}}$ di Karplus**: Calcolo delle costanti di accoppiamento dai veri angoli diedri conformazionali 3D.
  * **Anisotropia Magnetica e Correnti d'Anello (Modello di McConnell)**: Schermatura upfield (verso campi alti) e deschermatura indotte dalla corrente d'anello degli anelli aromatici.
  * **Legami a Idrogeno Intramolecolari**: Spostamenti verso campi bassi (downfield) per protoni $-\text{OH}$ e $-\text{NH}$ chelati.
* **Simulazione Line-Shape Realistica**: Profilo continuo convoluto lorentziano con sdoppiamento in multipletti (singoli, doppietti, tripletti, quartetti, doppi doppietti, multipletti complessi) e traccia cumulativa dell'integrale.
* **Convenzione Spettroscopica**: Asse $\delta$ (ppm) invertito da destra (0 ppm) a sinistra (12-14 ppm o 220 ppm per ¹³C).

### 2. Spettroscopia Infrarossa (IR Vibrazionale - Psi4)
* **Matrice Hessiana e Modi Normali**: Calcolo quantistico *ab initio* / DFT con Psi4 di tutte le frequenze armoniche $\nu_i$ e delle intensità dipolari $I_i$ ($\text{km/mol}$).
* **Fattori di Scala Armonica**: Correzione automatica dell'anarmonicità in base al metodo e al basis set (database NIST CCCBDB, es. 0.965 per B3LYP).
* **Convoluzione di Riga**: Simulazione dello spettro continuo sia in **% Trasmittanza (%T)** (picchi verso il basso con linea base al 100%) che in **Assorbanza**.
* **Riconoscimento Bande Diagnostiche**: Etichettatura automatica delle bande funzionali (Carbonili $\text{C=O}$, Ossidrili $\text{O-H}$, Ammine, Alchini, Alcheni, Aromatici, C-H alifatici).

### 3. Modalità di Esecuzione

| Modalità | Descrizione | Quando Usarla |
| :--- | :--- | :--- |
| **Modalità Completa (Ensemble Boltzmann)** | Esegue la ricerca conformazionale completa con `energy-inator` (RDKit $\to$ MMFF94 $\to$ Psi4), calcola le energie relative $\Delta E$ e le popolazioni di Boltzmann $P_k$, e genera lo **spettro reale medio in soluzione**: <br>• $\langle \delta \rangle = \sum_k P_k \delta_k$ (media temporale NMR)<br>• $I_{\text{IR}}(\nu) = \sum_k P_k I_k(\nu)$ (somma istantanea IR) | Per molecole flessibili e per confrontare le predizioni con spettri sperimentali registrati a temperatura ambiente. |
| **Modalità Singola (Conformero Unico)** | Ottimizza con Psi4 o calcola direttamente le proprietà spettrali sul singolo conformero 3D fornito. | Per calcoli rapidi (pochi secondi), per file `.xyz` già ottimizzati esternamente, o per studiare le differenze spettrali di una singola conformazione. |

### 4. Esportazione nei Formati Standard
* **JCAMP-DX (`.jdx`)**: Formato ufficiale IUPAC per spettri NMR e IR, compatibile nativamente con **MestReNova, Bruker TopSpin, Varian, ACD/Labs**.
* **Tabelle CSV**: Elenco picchi, costanti $J$, integrali, frequenze e intensità.
* **Report JSON Completo**: Tutti i metadati, conformeri e serie numeriche.

---

## Installazione e Setup

Il software condivide l'ambiente Conda/Python di `energy-inator`:

```bash
# Entra nella directory del progetto
cd nmr-inator

# Installa nmr-inator in modalità modificabile (editable)
..\energy-inator\.venv\python.exe -m pip install -e .
```

---

## Utilizzo da Riga di Comando (CLI)

```bash
# 1. Calcolo veloce in modalità singola (1H e 13C NMR su etanolo)
python -m nmrinator.cli --smiles "CCO" --mode single --no-ir -o output_etanolo

# 2. Calcolo completo con IR (Psi4) e NMR su singolo conformero
python -m nmrinator.cli --smiles "CC(=O)C" --mode single --method b3lyp --basis 6-31G(d)

# 3. Calcolo completo con Ensemble Boltzmann di conformeri (Energyinator)
python -m nmrinator.cli --smiles "CCCC" --mode full --confs 30 --top-confs 3

# 4. Input da file XYZ 3D già noto
python -m nmrinator.cli --xyz molecola.xyz --mode single
```

---

## Interfaccia Web Interattiva (Streamlit GUI)

Per avviare la GUI con un click su Windows:
* Fare doppio click sul file **`avvia_gui.bat`**

Oppure da terminale:
```bash
..\energy-inator\.venv\python.exe -m streamlit run gui\app_gui.py
```

La GUI include:
* Disegno della struttura 2D con **Ketcher** o inserimento SMILES / upload file (.xyz, .mol, .sdf).
* **Console di Avanzamento in Tempo Reale** con pulsante **STOP** per interruzione immediata.
* **Archiviazione Permanente delle Sessioni**: salvataggio automatico in `output/session_YYYYMMDD_HHMMSS_<formula>/` con ripristino istantaneo di grafici, tabelle e log dal pannello laterale *Sessioni Archiviate*.
* Grafici interattivi **Plotly** per ¹H NMR, ¹³C NMR e Spettro IR.
* Zoom, ispezione dei picchi e tabella dei multipletti con assegnazione atomica.
* Pulsanti di download istantaneo dei file grafici ad alta risoluzione (**PNG, SVG**) e formati standard interoperabili (**JCAMP-DX**, **CSV** e **JSON**).
