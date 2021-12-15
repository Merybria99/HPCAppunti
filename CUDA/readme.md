# **GPGPU**

Generalmente il calcolo grafico viene effettuato in maniera vettoriale da delle architetture dedicate, dette GPU.
Sfruttando le architetture per il calcolo grafico, basato su sturrure vettoriali e matriciali, è possibile estendere
l'uso delle GPU alla parallelizzazione di problemi **General Pourpose**. Da qui nascono le **GPGPU**.

Le **GPGPU** eseguono calcoli, dal punto di vista della CPU, con il paradigma **SPMD** (single-processor-multiple-data).

### **GPU**

Nonstante le architetture siano molto potenti, in quanto arrivano fino **a** **1024 thread**, è difficile sfruttare a
pieno la potenza di calcolo in quanto **la logica di funzionamento è differente rispetto ad OMP e MPI**.

Data la sempicità architetturale delle unità elaborative, il **calcolo** che esse devono effettuare deve essere di ****
bassa** difficoltà**.

Altra problematica è costituita da una **limitata località spaziale**, data da cache di piccole dimensioni. Questo
vincolo è presente in quanto vista la massiva presenza di unità di calcolo è necessario, per avere delle **cache
locali**, che esse siano di **dimensioni ridotte**, anche se comunque **complesse e multilivello**.

Per questo motivo si definisce il parallelismo delle GPU come **parallelismo a GRANA FINE**, ovvero ogni unità **
elaborativa** lavora su pochi dati.

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

Sono ottimizzate per il **calcolo grafico** e con molti **più stadi di pipes,** possibili grazie alla **assenza di
interruzioni** all'interno delle istruzioni elaborate.

Il **throughput è molto alto** poichè le elaborazioni sono effettuate da **migliaia di unità elaborative** in parallelo.

La **logica di controllo** impiegata è molto **semplice**, anche se la sua complessità sta crescendo e avvicinandosi a
quella delle CPU.

## **Architettura delle GPU CUDA**

La CPU, al momento della esecuzione di un task in parallelo sulla GPU, esegue un **kernel**, ovvero una **unica
funzione** comune ad ogni unità elaborativa della GPU. Quando la GPU esegue il kernel, **non può interrompere la
computazione** finchè essa non termina.

In questo paragrafo si descrive la struttura della architettura di GPU Cuda.

![Immagine tipo Structure](immagini/cudaArc.png)

La architettura Cuda viene sostanzialmente a comporsi di tre livelli.

- **Griglie**: sono la prima suddivisione della architettura. Esse contengono delle sottostrutture interne e possono
  essere articolate in 1D, 2D e 3D.

- **Blocchi**: i blocchi sono contenuti all'interno delle griglie e contengono a loro volta delle sottosezioni
  contenenti effettivamenete i blocchi di elaborazione della GPU. Anche essi sono articolati in 1D, 2D o 3D.

- **Thread**: Propri di ogni blocco, i threada sono delle piccole **unità di calcolo**. Essi, come i casi precedenti,
  possono articolarsi in architetture a 1D, 2D o 3D. Ogni thread, per il proprio corretto funzionamento ha al suo
  interno una memoria dedicata, di solito privata.

### Memorie presenti nella griglia

All'interno di ogni griglia sono presenti tre diverse tipologie di memoria condivisa dai blocchi nella griglia.

- **Memoria globale**;

- **Memoria costante**;

- **Memoria texture**: impiegata per calcoli grafici particolari come la rifrazione della luce su un oggetto.

### Memorie presenti nei blocchi

In ogni blocco è presente una **memoria shared** condivisa tra i thred in esso presenti.

Questa ha una larghezza di banda molto elevata, permettendo così di **poter implementare una comunicazione tra thread
solo se presenti all'interno dello stesso blocco**.

### Memorie all'interno dei thread

All'interno di ogni thread sono presenti sia una **memoria locale,** che funge da cache per l'unità di elaborazione, sia
dei **registri**, che devono seguire distribuzione spaziale dei thread.

# **CUDA**

CUDA è una **API** di programmazione per schede video NVIDIA. I linguaggi impiegati per l'interazione C e Fortran.

### **API**

Le api di CUDA sono suddivise principalmente in **due livelli** mutuamente esclusivi nell'impiego:

- **Driver API**: riguardanti l'interazione di basso livello;

- **Runtime API**: comandi per gestione delle componenti della gpu;

Il livello **runtime parla con il livello driver**. Questo perchè le ottimizzazioni effettuate devono avere un riscontro
hardware, ottenuto mediante l'impiego dei driver.

## Warps

Sono un **raggruppamento di thread** all'interno degli SM, generalmente di 32 threads.

In particolare i thread, i quali effettuano la stessa **istruzione nello stesso tempo**, sono scelti dal driver per
essere raggruppati in Warp (array di ALU).

Idealmente, sarebbe inoltre necessario, che le istruzioni seguenti per ogni thread nel warp siano uguali tra loro. Se
così accade le prestazioni sono MOLTO incrementate.

Inoltre, per permettere il passaggio di dati nei warp parallelamente all'esecuzione delle istruzioni negli stessi, sono
predisposti dei **moduli speciali LDST** (load and store) i quali effettuano una fase di pre-fetching per inserire i
dati nelle cache locali.****
****

## **Compute capability**

La compute capabilty costituisce la **misura della capacità di calcolo** di un dispositivo definita come:

una **coppia X.y** dove:

- **X** è il major number che identifica l'**architettura del chipset**.

- **y** è il minor number e rappresenta la **differenza di release** dello stesso chipset.

E' importante notare che l'efficienza **non dipende dall' aumento della compute capability**.

All'interno della tabella identificativa delle GPU è possibile andare a vedere i **parametri di analisi** delle
prestazioni andando a visionare i campi:

- **Streaming Multiprocessor**:parametro che indica il numero di **raggruppamenti di thread a livello del blocco**.
  Questo parametro identifica in quante unità suddividere il carico per ogni blocco;

- **Thread / Warp**: rappresenta il numero totale di thread per ogni warp;
- **MaxWarps / SM**: numero massimo di warp per ogni SM;
- **Max Thread / SM**: numero massimo di thread per ogni SM;
- **Max ThreadBlock / SM**: numero massimo di blocchi di thread per ogni SM;

****

## **Threads and streming multiprocessor**

<img src="immagini/streMul.jpg" width="450"/>
<br>

Ogni **thread è eseguito da un Core**. Blocchi di thread snno eseguiti da blocchi di core organizzati in **streaming
multiprocessor**. La loro esecuzione è schedulata secondo il **Round-Robin** ed è **indipendente** tra i vari blocchi.

I blocchi di thread sono organizzati in **griglie**. La loro esecuzione è tanto più parallela quanto più sono gli
streaming multiprocessor.



****

## **Gestione dei dati**

I movimenti dei dati possono essere effettuati **o dalla CPU ,definita anche host, alla GPU, definita anche device,(H2D)
o viceversa (D2H)**.

Questa fase costituisce il bottleneck della esecuzione parallela su GPGPU. I movimenti vengono effettuati **o prima o
dopo la computazione** in quanto l'esecuzione del kernel non può essere interrotta.

I movimenti dei dati sono gestiti dal protocollo **PCIe e DDR**.

In alcune architetture di NVIDIA il protocollo impiegato per la comunicazione dei dati è basato su una infrastruttura
proprietaria, **NVLink più costosa ma con prestazioni altissime**.

****

## **Latenze**

La fase di prefetching o riordino delle istruzioni all'interno delle GPGPU è stata inserita per permettere un continuo
funzionamento delle unità di calcolo.

Questo poichè la latenza massima per l'esecuzione di una istruzione, anche critica, è circa 30 cicli di clock, mentre
una operazione di load and store può avere una latenza fino a 800 cicli di clock.

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

Il kernel viene eseguito N volte da N diversi Thread CUDA sulla GPU. Ogni thread ha un **un id univoco**, consultabile
con la variabile **threadIdx** .

Ogni variabile dichiarata come dim3 realizza una struttura dati i cui campi sono consultabili con .x, .y , .z . A
seconda del numero dei parametri passati al costruttore, si dichiara una matrice, un vettore o un parallelogramma.

Per dichiarare il numero di thread che eseguono il kernel si usa una specifica affiancata al nome del kernel da
eseguire.

```c
kernelFunction<<<numBlocks,numTreads>>>(args...)
```

In questo caso possiamo evidenziare **due paramentri**:

- **numBlocks** (o gridDim): rappresenta la **taglia della griglia** in termini di **blocchi di thread**, lungo ogni
  dimensione, si dichiara con **dim3**.
- **numThreads** (o blockDim): la **taglia dei blocchi** rappresentata in thread per o**gni dimensione**, si rappresenta
  con **dim3**

*Esempio:*

```c
  dim3 numThreads(32);
  //il numero di blocchi è dichiarato comunque dim3 e possiamo definire fino a 3 dimensioni, ogni dimensione è data dalla grandezza della struttura dati da manipolare in quella dimensione, rapportata ai thread disponibili.
  dim3 numBlocks((N-1)/numThreads.x+1);
  kernelFunction<<<numBlocks,numThreads>>>(args...);

```

***

## Metodologie di calcololo dell' offset

A seconda della topologia scelta per la nostra griglia e per i suoi blocchi all'interno, il calcolo dell' indice,
rappresentativo per ogni thread, cambia.

### **Griglia di 1D e Blocchi di 1D**

<img src="immagini/index1D.jpg" width="450"/>
<br>
In questo caso è necessario calcolare l'offest per ogni singolo thread analizzando soltanto la dimensione su x.

```c
  int index = blockIdx.x * blockDim.x + threadIdx.x;
```

Il **primo pezzo** della assegnazione rappresenta **quanti blocchi saltare**, in termini di thread;

Il **secondo pezzo** invece, dato il *b-esimo blocco*, ci dice **quale thread stiamo considerando** di quest' ultimo;

### **Griglia di 1D e Blocchi di 3D**

<img src="immagini/index-1DGrid-3DBlocks.jpg" width="450"/>
<br>
In questo caso è necessario calcolare l'offset andando a considerare le tre dimensioni del blocco.

```c
  int index = blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
              threadIdx.z * blockDim.y * blockDim.x + 
              threadIdx.y * blockDim.x +
              threadIdx.x;
```

Il **primo termine** rappresenta il numero di **blocchi da saltare** per arrivare al blocco interessato (rappresentato
da blockIdx.x);

Il **secondo** rappresenta il **numero di layer** di thread nel blocco interessato che sono allineati sulla stessa z e
che sono **da saltare**;

Il **terzo termine** rappresenta, dato il layer selezionato,**le righe da saltare** per accedere alla riga corretta;

Il **quarto termine** rappresenta l'**id del thread** nella riga;

### **Griglia d 2D e Blocchi di 2D**

In questo caso il calcolo dell'offset si esplica nella valutazione dell'offset sia nella dimensione x e y della griglia.

<img src="immagini/22.png" width="450"/>
<br>

Questa metodologia viene generalmente applicata quando si desidera **mappare in memoria una matrice**.

1. Nel caso in cui una matrice abbia **dimensione tale da non poter essere mappata** completamente nella griglia allora
   è **necessario suddividerla** in sottomatrici;

2. Nel caso in cui la matrice sia completamente **mappata ed esattamente pari alla griglia**;

3. Altresì è possibile che la matrice sia mappabile all'interno della griglia ma abbia **taglia ridotta** rispetto ad
   essa. Quindi sono presenti aree della griglia che non hanno **mappate al di sopra degli elementi della matrice**.

Il primo caso generalmente è riconducibile, tramite la decomposizione in sottomatrici, nei due successivi.

In generale gli offset fanno riferiento alla dimensione della griglia e dei blocchi e non a quella della matrice da
mappare e sono calcolabili come segue:

```c
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j = blockIdx.y * blockDim.y + thredaIdx.y;
```

Il primo elemento di entrambe le formule viene impiegato per calcolare il numero di blocchi, espresso in thread, da
saltare.

I secondi termini rappresentano i thread negli specifici blocchi.

```c
//rappresentazione dell'indice dopo aver mappato la matrice in memoria attraverso una notazione linearizzata.
  int index = j * MatrixWidth + i;

  //nel caso in cui si vogliano mappare i dati della matrice con l'intero insieme di blocchi, allora:

  MatrixWidth =  gridDim.x * blockDim.x;
```

*Esempio*:

Kernel del prodotto tra due matrici N*N, ogni thread esegue l'operazione su un singolo elemento delle stesse.

```c 
    __global__ void matrixAdd(int N, const float ∗A, const float ∗B, float ∗C) {
      int i = blockIdx.x ∗ blockDim.x + threadIdx.x;
      int j = blockIdx.y ∗ blockDim.y + threadIdx.y;

      int index = j ∗ N + i;

      if ( i < N && j < N )
        C[index] = A[index] + B[index];
    }
```

Il calcolo di **i e j** si effettua per ottenere la **misura degli offset** rispetto alla griglia in cui è mappata.

Il calcolo dell'index effettuato per avere il **riferimento all'indice della matrice** e viene anche in questo caso
effettuato in maniera linearizzata.

L'**if** all'interno della funzione viene **impiegato per verificare se l'accesso ai dati rientra nell'area di mappatura
della matrice o si sta andando oltre**. Se si sta andando oltre si potrebbe accedere a blocchi impiegati in altro modo,
non inerenti alle matrici.

L'inserimento dell'**if è molto dispensioso** a livello di compilazione (tradotto in una jump e poichè ci sono pipe che
hanno cattive prestazioni con istruzioni di salto, le prestazioni calano e la pipe lunga non viene sfruttata) e quindi
un **modo alternativo per aumentare le prestazioni mantenendo la correttezza è comunque operare sull'intero blocco**,
nonstante la matrice non sia mappata sull'intero blocco.

**Alcuni thread** in parallelo, semplicemente, effettueranno delle **operazioni non rilevanti**.

***

# Gestione della memoria in CUDA

Per poter gestire i dati all'interno della GPGPU è necessario allocare della memoria sul device e poi inizializzarla ai
dati che desideriamo gestire.

All'interno di CUDA tutte le funzioni hanno come parametro di ritorno **un codice di errore** che può essere anche **
cudaSuccess**.

## **cudaMalloc**

Per gestire l'allocazione della memoria sulla GPU si utilizza la funzione:

```c
    cudaError_t cudaMalloc ( void** devPtr, size_t size )
```

```c 
  double *array_device;
  cudaMalloc((void**) &array_dev, N * sizeof(double));
```

In questo esempio viene allocato un riferimento alla area di memoria che si desidera allocare sulla GPGPU anche se è
memorizzato sull'host.

E' da notare come, essendo in C il passaggio dei parametri unicamente per copia ,si necessita passare l'indirizzo di
memoria della variabile su cui si vuole scrivere il valore di ritorno (puntatore), questo poichè la cudaMalloc, come
detto precendentemente, ritorna il codice di errore.

Essendo che il valore di ritorno rappresenta un puntatore ad una area di memoria si necessita passare un doppo
puntatore.

Si nota come, inoltre, la cudaMalloc abbia bisogno di un void** in quanto la memoria allocata è di tipo generico,
pertanto si necessita un casting per la compatibilità del tipo.

Per poter deallocare la memoria viene adoperata la funzione **cudaFree** nella quale va specificato il puntatore
all'area di memoria da dover deallocare.

## **cudaMemset**

Funzione impiegata per poter inizializzare le aree di memoria allocate sul device.

```c
    cudaError_t cudaMemset (void * devPtr, int value, size_t count )
```

Il funzionamento è di tipo ottimizzato in quanto sfrutta il DMA della architettura facendo sniffig del bus e facendo al
momento opportuno bus stealing.

Per questa funzione la memoria puntata dal primo argmento alvalore specificato nel secondo parametro.

## **cudaMemcpy**

Per permettere la inizializzazione della memoria allocata in modo bidirezionale.

```c
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```

Vengono grazie a questa direttiva copiati i dati specificati in tre direzioni:

- **H2D**: ovvero dall'host alla GPU. Questa direzione si specifica con la keyword *cudaMemcpyHostToDevice*;

- **D2H**: ovvero dalla GPU alla CPU. Questa direzione si specifica con la keyword *cudaMemcpyDeviceToHost*;

- **cudaMemcpyDeviceToDevice**: ovvero la copia dei dati dalla GPU alla GPU;

*Esempio*:

```c
    cudaMemcpy(array_dev,array_host, sizeof(array_host,cudaMemcpyDeviceToHost)
    cudaMemcpy(array_host,array_dev, sizeof(array_dev,cudaMemcpyHostToDevice)
```

La direzione della copia viene però a mancare nel caso in cui utilizziamo Cuda 4.0 o maggiori in quanto essa riesce a
comprendere in modo automatico la direzione della copia.

## **Esempio completo delle funzioni**

```c
  #include <stdio.h>
#include <stdlib.h>

void  initVector(double *u, int n, double c) {
  int i;
  for (i=0; i<n; i++)
      u[i] = c;
}

__global__ void gpuVectAdd(double *u, double *v, double *z, int N) 
{
  // define index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // check that the thread is not out of the vector boundary
  if (i >= N ) return;

  int index = i; 
  // write the operation for the sum of vectors 
  z[index] = u[index] + v[index];
}


int main(int argc, char *argv[]) {

  // size of vectors
  const int N = 1000;

  // allocate memory on host
  double * u = (double *) malloc(N * sizeof(double));
  double * v = (double *) malloc(N * sizeof(double));
  double * z = (double *) malloc(N * sizeof(double));

  initVector((double *) u, N, 1.0);
  initVector((double *) v, N, 2.0);
  initVector((double *) z, N, 0.0);

  // allocate memory on device
  double *u_dev, *v_dev, *z_dev;
  cudaMalloc((void **) &u_dev, N*sizeof(double));
  cudaMalloc((void **) &v_dev, N*sizeof(double));
  cudaMalloc((void **) &z_dev, N*sizeof(double));

  // copy data from host to device
  cudaMemcpy(u_dev, u, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(v_dev, v, N*sizeof(double), cudaMemcpyHostToDevice);

  dim3 block(32);
  dim3 grid((N-1)/block.x + 1);

  // define the execution configuration
  gpuVectAdd<<<grid, block>>>(u_dev, v_dev, z_dev, N);

  // copy data from device to host
  cudaMemcpy(z, z_dev, N*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i=0;i<N;i++)
  	printf("z\[%d\] = %f\n",i,z[i]);

  // free resources on device
  cudaFree(u_dev);
  cudaFree(v_dev);
  cudaFree(z_dev);

  // free resources on host
  free(u);
  free(v);
  free(z);

  return 0;
}s
```

## Compilatore in CUDA

Il compilatore per il processing dei file CUDA lavora attraverso una divisione in due fasi: una fase di **frontend** ed
una fase di **backend**.

Nella fase di forntend si occupa della della definizione di codice oggetto **sia per la GPU che per la CPU**. Inoltre il
ompilatore genera del codice Qbin e non lo genera per architetture predefinite, ma per diverse architetture. Deve essere
quindi **specificata l'architettura per la quale deve essere prodotto il codice**.

```c
nvcc --arch = compute_37 --code = sm_37 (caso con K80)
```

nella quale si specificano sia la compute capability che lo streming multiprocessor della architettura in questo momento
impiegata.

## **Error handling**

Tutte le system call Nvidia definiscono una variabile cudaError_t come valore di ritorno che permette di verificare lo
stato della eseuzione della direttiva.

Il valore della cudaError_t è rappresentato attraverso un intero al quale ci si può riferire nel caso di successo
attraverso la define cudaSuccess.

Per la gestione degli errori nel caso in cui essi avvengano nella esecuzione del Kernel, il quale ha un funzionamento
bloccante, è necessario andare a leggere una variabile che mantenga al suo interno il valore dell'ultimo errore
rintracciato:

```c
    cudaGetLastError();
```

la funzione deve essere chiamata a seguito di una sincronizzazione esplicita con il termine della esecuzione del kernel.
Ciò può essere implementato attraverso la specifica:

```c
    cudaDeviceSynchronize();
```

Alternativamente è possibile adoperare la macro:

```c
    #define CUDA_CHECK(X) {\
      cudaError_t _m_cudaStat = X;\
      if(cudaSuccess != _m_cudaStat) {\
        fprintf(stderr,"\nCUDA_ERROR: %s in file %s line %d\n",\
        cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__);\
        exit(1);\
      } \
    }

    ...
    
    CUDA_CHECK( cudaMemcpy(d_buf, h_buf, buffSize, cudaMemcpyHostToDevice));
```

***

## CUDA Events

Un evento è una particolare variabile impiegata all'interno del codice per marcarlo. Esso ha due finalità:

- ottenere il tempo di una esecuzione;
- identificare punti di sincronizzazione della CPU e della GPU;

### **Ottenimento del tempo**

Vengono definite all'interno del programma dei punti di acquisizione di dati istanti temporali.

Si crea un evento di tipo start e stop per l'acquisizione degli istanti temporali. All'inizio del codice da monitorare
si effettua un EvenRecord di start e al termine si effettua un EventRecord di stop.

Se si vuole effettivamente assicurare che il tempo preso sia quello del Kernel si deve effettuare una chiamata
sincronizzata in modo da non registrare l'evento prima della sua fine.

Per il calcolo della differenxa si chiama la funzione cudaEventElapsedTime nella quale si salva la differenza
all'interno della variabile specificata come prima variabile.

*Esempio:*

```c
  cudaEvent t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ...
  kernel<<<grid, block>>>(...);
  ...
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed;
  // execution time between events in ms
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ```

## **Prestazioni di una programma in Cuda**

### Bandwidth di un programma in Cuda

All'interno di una applicazione Cuda è importante valutare la memory bandwidth.

La situazione ideale sarebbe quella di effettuare un unico trasferimento per l'intera mole di dati dalla CPU alla GPU.

Spesso ciò non è possibile quindi è necessario ottimizzare i trasferimenti ed evitare dei tempi di idle della CPU e
della GPU.

Per la valutazione della bandwidth del bus allora si effettua un comando che permette di valutare la prestazione del bus
dei trasferimenti specificando:

- **il valore di start delle misurazioni**

- **il valore di stop delle misurazioni**

- **l'offset delle misurazioni**

```c 
./bandwidthTest --mode=range --start=<B> --end=<B> --increment=<B>
```

La misurazione viene effettuata in **MFlop** ovvero il numero di floating point operations per second.

## Ottimizzazione delle prestazioni

Attraverso la scelta dei parametri di configurazione del kernel:

- **gridSize** : numero di blocchi nella griglia;
- **blockSize** : numero di thread nel blocco;

è possibile prestazioni diverse sulla stessa struttura hardware a seconda dei loro valori.

Per decidere questi valori è necessario:

- rispettare i limiti di thread per blocco, imposti dalla GPU;

- selezionare la configuarzione della griglia in modo che possa processare tutti gli elementi;

- selezionare la taglia dei blocchi per evitare deadlock tra thread e minimizzare la dipendenza tra i kernel eseguiti
  nei vari thread.

- impiegarei qualificatori propri di CUDA per riferirsi agli id dei blocchi (threadIdx, blockIdx);

### Come effettuare una corretta ottimizzazione

Deve essere ispezionata la scheda tecnica dei CUDA per la GPU effettivamente impiegata. Sono di interesse le grandezze
di SM disponibili e il numero di core per ognuno di essi.

Oltre al dimensionamento dei core per SM da effettaure deve essere anche effettuata una corretta distribuzione dei
registri ad essi. Chiaramente deve essere rispettata la necessità di un numero minimo di registri per ogni kernel o
altrimenti verrà rallentata la velocità di esecuzione a causa dei tempi di attesa.

### **Esempio di risoluzione di una configurazione**

``` 
32768 registri per SM; Kernel con una griglia 32x8 blocchi di thread; il kernel necessita di 30 registri. (il kernel viene eseguito da ogi thread)

Quanti blocchi di thread possono essere ospitati su un solo SM?
```

1. calcolo di registri totali necessari:

``` 
numero di registri totali per blocco= 
 numero di thread per blocco * numero di registri per thread 
```

risultato : 32\*8\*30 =7680

2. calcolo dei possibili blocchi che lo streaming multiprocessor può ospitare.

```c 
numero di blocchi allocabili =
parte intera di (numero di registri disponibili per SM / numero di registri per blocco)
```

risultato: 32768 / 7680 = 4

Nel caso in cui il numer di registri varia, varia altresì il numero di blocchi allocabili.

In generale la taglia appropriata di blocchi per una data architettura viene calcolaata attraverso la **TILE_WIDTH**
ovvero un dimensionamento 2^x * 2^x thread per blocco.

Alla variazione dei valori di x della tile_width viene calcolata l'effettiva occupazione dei thread disponibili sulla
scheda video considerata tramite la compute capability (si deve rispettare il numero massimo di blocchi per ogni SM).

Si **sceglie la configurazione con la migliore occupazione dei thread disponibili su streaming multiprocessor**.

### Secondo esempio di risoluzione con uso di TILE WIDTH

```
Ipotesi: architettura fermi in cui ogni SM può gestire 1536 threads e al massimo 8 blocchi residenti

Bisogna effettuare delle prove con diversi TILE WIDTH e scegliere il piu conveniente 
```

- Scelgo una tile width iniziale di **8**  
  <br>
  8 \* 8 = 64 threads per blocco. La scheda fermi ne mette a disposizione 1536, quindi possono essere allocati
  totalmente 1536 / 64 =24 blocchi. Essendo il **massimo numero di blocchi 8**, possono essere impiegati solamente 8 dei
  24 blocchi. Il numero dei thread effettivamente allocati per ogni SM sarà quindi **64 \* 8 = 512** thread.

  Avremo quindi una occupazione del 33%.


- L'occupazione non è quindi ottimizzata, scegliamo una tile width **16**
  <br>
  16 \* 16 = 256 threads per blocco. La scheda fermi ne mette a disposizione 1536, quindi possono essere allocati
  totalmente 1536 / 256 = 6 blocchi.

  Essendo il **massimo numero di blocchi 8**, possono essere impiegati completamente i 6 blocchi a disposizione.
  Il numero dei thread effettivamente allocati sarà la totalità di quelli a disposizione.
  


  





























