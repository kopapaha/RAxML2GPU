extern "C" int Thorough;

extern "C" int alignLength;
extern "C" int nofSpecies;

extern "C" float *d_sumBuffer;
extern "C" float *d_EV;
extern "C" float *d_tipVector;
extern "C" double *d_gammaRates;
extern "C" double *d_patrat;
extern "C" double *d_ei;
extern "C" double *d_eign;
extern "C" int *d_rateCategory;
extern "C" int *d_wgt;

extern "C" unsigned int *d_scalerThread;
extern "C" double *d_partitionLikelihood;

extern "C" double *d_dlnLdlz;
extern "C" double *d_d2lnLdlz2;

extern "C" float *d_tmpCatSpace;
extern "C" float *d_tmpDiagSpace;

extern "C" float *d_wr2;
extern "C" float *d_wr;


extern "C" tree *d_tree;
extern "C" traversalInfo *d_ti;
extern "C" float **d_xVector;
extern "C" unsigned char **d_yVector;

extern "C" float *d2_umpX1;
extern "C" float *d2_umpX2;
extern "C" float *d2_left;
extern "C" float *d2_right;


__device__ int Thorough_d = -1;
__device__ float *d_umpX1;
__device__ float *d_umpX2;
__device__ float *d_left;
__device__ float *d_right;

__device__ int alignLength_d;
__device__ int nofSpecies_d;

__device__ float *sumBuffer;
__device__ float *EV;
__device__ float *tipVector;
__device__ double *gammaRates;

__device__ double *patrat;
__device__ double *ei;
__device__ double *eign;
__device__ int *rateCategory;
__device__ int *wgt;

__device__ unsigned int *scalerThread;
__device__ double *partitionLikelihood;

__device__ double *dlnLdlz;
__device__ double *d2lnLdlz2;

__device__ float *tmpCatSpace;
__device__ float *tmpDiagSpace;

__device__ float *wr2;
__device__ float *wr;


__device__ tree *tr;
__device__ traversalInfo *ti;
__device__ float **xVector;
__device__ unsigned char **yVector;


__device__ int initScalerThread = 0;


__device__ volatile int endGPUexecution = FALSE; 
__device__ volatile int execKernel = -1;
__device__ volatile double tr_coreLZ0mw; //for master-worker scheme
__device__ volatile boolean firstIteration_mw = TRUE;
__device__ volatile int chkLast = FALSE;


/**
 * global barrier
 * GTX 480 max dims: 120x128 (blocks, threads/block)
 *  
 */

struct barrier{
  volatile int blockFinish;
  volatile int sense;
};
__device__ struct barrier b = {0, 0};

