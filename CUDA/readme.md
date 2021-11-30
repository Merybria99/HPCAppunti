# **GPGPU**

Generalmente il calcolo grafico viene effettuato in maniera vettoriale da delle architetture dedicate, dette GPU.
Sfruttando le architetture per il calcolo grafico, basato su sturrure vettoriali e matriciali, è possibile estendere l'uso delle GPU alla parallelizzazione di problemi **General Pourpose**. Da qui nascono le **GPGPU**.

Le **GPGPU** eseguono calcoli, dal punto di vista della CPU, con il paradigma **SPMD** (single-processor-multiple-data).

### **GPU**

Nonstante le architetture siano molto potenti, in quanto arrivano fino **a** **1024 thread**, è difficile sfruttare a pieno la potenza di calcolo in quanto **la logica di funzionamento è differente rispetto ad OMP e MPI**.

Data la sempicità architetturale delle unità elaborative, il **calcolo** che esse devono effettuare deve essere di ****bassa** difficoltà**.

Altra problematica è costituita da una **limitata località spaziale**, data da cache di piccole dimensioni. Questo vincolo è presente in quanto vista la massiva presenza di unità di calcolo è necessario, per avere delle **cache locali**, che esse siano di **dimensioni ridotte**, anche se comunque **complesse e multilivello**.

Per questo motivo si definisce il parallelismo delle GPU come **parallelismo a GRANA FINE**, ovvero ogni unità **elaborativa** lavora su pochi dati.

Il **bottleneck** delle operazioni è definito nel **passaggio dei dati nella memoria della GPU**.

### **Confronto tra CPU e GPU**

- #### **CPU**
Le cpu hanno una **logica di cotrollo complessa**, ad esempio interruzioni o jumps.

Presentano **pochi core**.

Presentano delle **cache multilivello** di grandi dimensioni.

La **velocità di clock è elevata**. 

Le pipes sono caratterizzate da **pochi stadi** dati dalla presenza nelle istruzioni di interruzioni o di cicli.


- #### **GPU**
Le GPU presentano unità di **elaborazione multiple**, molto più **semplici** delle CPU e più simili a delle **ALU**.

Sono ottimizzate per il **calcolo grafico** e con molti **più stadi di pipes,** possibili grazie alla **assenza di interruzioni** all'interno delle istruzioni elaborate.

Il **throughput è molto alto** poichè le elaborazioni sono effettuate da **migliaia di unità elaborative** in parallelo.

La **logica di controllo** impiegata è molto **semplice**, anche se la sua complessità sta crescendo e avvicinandosi a quella delle CPU.


## **Architettura delle GPU CUDA**

La CPU, al momento della esecuzione di un task in parallelo sulla GPU, esegue un **kernel**, ovvero una **unica funzione** comune ad ogni unità elaborativa della GPU. Quando la GPU esegue il kernel, **non può interrompere la computazione** finchè essa non termina.

In questo paragrafo si descrive la struttura della architettura di GPU Cuda.

![Immagine tipo Structure](immagini/cudaArc.png)

La architettura Cuda viene sostanzialmente a comporsi di tre livelli.

- **Griglie**: sono la prima suddivisione della architettura. Esse contengono delle sottostrutture interne e possono essere articolate in 1D, 2D e 3D.

- **Blocchi**: i blocchi sono contenuti all'interno delle griglie e contengono a loro volta delle sottosezioni contenenti effettivamenete i blocchi di elaborazione della GPU.
Anche essi sono articolati in 1D, 2D o 3D.

- **Thread**: Propri di ogni blocco, i threada sono delle piccole **unità di calcolo**. Essi, come i casi precedenti, possono articolarsi in architetture a 1D, 2D o 3D. Ogni thread, per il proprio corretto funzionamento ha al suo interno una memoria dedicata, di solito privata.

### Memorie presenti nella griglia

All'interno di ogni griglia sono presenti tre diverse tipologie di memoria condivisa dai blocchi nella griglia.

- **Memoria globale**; 

- **Memoria costante**;

- **Memoria texture**: impiegata per calcoli grafici particolari come la rifrazione della luce su un oggetto.


### Memorie presenti nei blocchi

In ogni blocco è presente una **memoria shared** condivisa tra i thred in esso presenti.


Questa ha una larghezza di banda molto elevata, permettendo così di **poter implementare una comunicazione tra thread solo se presenti all'interno dello stesso blocco**.

### Memorie all'interno dei thread 

All'interno di ogni thread sono presenti sia una **memoria locale,** che funge da cache per l'unità di elaborazione, sia dei **registri**, che devono seguire distribuzione spaziale dei thread.


# **CUDA** 
CUDA è una **API** di programmazione per schede video NVIDIA. 
I linguaggi impiegati per l'interazione C e Fortran.

### **API**
Le api di CUDA sono suddivise principalmente in **due livelli** mutuamente esclusivi nell'impiego:

- **Driver API**: riguardanti l'interazione di basso livello;

- **Runtime API**: comandi per gestione delle componenti della gpu;
  
Il livello **runtime parla con il livello driver**.
Questo perchè le ottimizzazioni effettuate devono avere un riscontro hardware, ottenuto mediante l'impiego dei driver.


## Warps
Sono un **raggruppamento di thread** all'interno degli SM, generalmente di 32 threads.

In particolare i thread, i quali effettuano la stessa **istruzione nello stesso tempo**, sono scelti dal driver per essere raggruppati in Warp (array di ALU).

Idealmente, sarebbe inoltre necessario, che le istruzioni seguenti per ogni thread nel warp siano uguali tra loro.
Se così accade le prestazioni sono MOLTO incrementate.

Inoltre, per permettere il passaggio di dati nei warp parallelamente all'esecuzione delle istruzioni negli stessi, sono predisposti dei **moduli speciali LDST** (load and store) i quali effettuano una fase di pre-fetching per inserire i dati nelle cache locali.****
****
## **Compute capability**

La compute capabilty costituisce la **misura della capacità di calcolo** di un dispositivo definita come:

una **coppia X.y** dove:
- **X** è il major number che identifica l'**architettura del chipset**.

- **y** è il minor number e rappresenta la **differenza di release** dello stesso chipset.

E' importante notare che l'efficienza **non dipende dall' aumento della compute capability**.

All'interno della tabella identificativa delle GPU è possibile andare a vedere i **parametri di analisi** delle prestazioni andando a visionare i campi:

- **Streaming Multiprocessor**:parametro che indica il numero di **raggruppamenti di thread a livello del blocco**. Questo parametro identifica in quante unità suddividere il carico per ogni blocco;

- **Thread / Warp**: rappresenta il numero totale di thread per ogni warp;
- **MaxWarps / SM**: numero massimo di warp per ogni SM;
- **Max Thread / SM**: numero massimo di thread per ogni SM;
- **Max ThreadBlock / SM**: numero massimo di blocchi di thread per ogni SM;

****
## **Threads and streming multiprocessor**

<img src="immagini/streMul.jpg" width="450"/>
<br>

Ogni **thread è eseguito da un Core**.
Blocchi di thread snno eseguiti da blocchi di core organizzati in **streaming multiprocessor**.
La loro esecuzione è schedulata secondo il **Round-Robin** ed è **indipendente** tra i vari blocchi.

I blocchi di thread sono organizzati in **griglie**. La loro esecuzione è tanto più parallela quanto più sono gli streaming  multiprocessor.



****
## **Gestione dei dati**
I movimenti dei dati possono essere effettuati **o dalla CPU ,definita anche host, alla GPU, definita anche device,(H2D) o viceversa (D2H)**.

Questa fase costituisce il bottleneck della esecuzione parallela su GPGPU.
I movimenti vengono effettuati **o prima o dopo la computazione** in quanto l'esecuzione del kernel non può essere interrotta.

I movimenti dei dati sono gestiti dal protocollo **PCIe e DDR**.

In alcune architetture di NVIDIA il protocollo impiegato per la comunicazione dei dati è basato su una infrastruttura proprietaria, **NVLink più costosa ma con prestazioni altissime**.

****

## **Latenze**

La fase di prefetching o riordino delle istruzioni all'interno delle GPGPU è stata inserita per permettere un continuo funzionamento delle unità di calcolo. 

Questo poichè la latenza massima per l'esecuzione di una istruzione, anche critica, è circa 30 cicli di clock, mentre una operazione di load and store può avere una latenza fino a 800 cicli di clock.

Questa asimmetria, se non gestita, causerebbe delle interruzioni di funzionamento delle unità di calcolo. 
 
La soluzione può avvenire tramite due strade, anche combinate:

- **Thread-Level Parallelism**
  
<img src="immagini/tlp.png" width="450"/>
<br>
Schedula tanti thread quanti è possibile trovare all'interno del warp.
I thread vengono eseguiti appena essi sono pronti, in maniera indipendente tra loro.


- **Instruction-level Parallelism**
  Un modo con granularità ancora più fine in quanto i thread vengono raggruppati analizzando le isatruzioni da eseguire.

****
# Estensione della sintassi per CUDA

Il **kernel** si definisce in C con la **keyword \_\_global\_\_** prima di definire una funzione.

Il kernel viene eseguito N volte da N diversi Thread CUDA sulla GPU.
Ogni thread ha un **un id univoco**, consultabile con la variabile *threadIDx* .

Ogni variabile dichiarata come dim3 realizza una struttura dati i cui campi sono consultabili con .x, .y , .z . 
A seconda del numero dei parametri passati al costruttore, si dichiara una matrice, un vettore o un parallelogramma.

Per dichiarare  il numero di thread che eseguono il kernel si usa una specifica affiancata al nome del kernel da eseguire.


35