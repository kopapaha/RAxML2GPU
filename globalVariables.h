/*  RAxML-VI-HPC (version 2.2) a program for sequential and parallel estimation of phylogenetic trees 
 *  Copyright August 2006 by Alexandros Stamatakis
 *
 *  Partially derived from
 *  fastDNAml, a program for estimation of phylogenetic trees from sequences by Gary J. Olsen
 *  
 *  and 
 *
 *  Programs of the PHYLIP package by Joe Felsenstein.
 *
 *  This program is free software; you may redistribute it and/or modify its
 *  under the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 *  for more details.
 * 
 *
 *  For any other enquiries send an Email to Alexandros Stamatakis
 *  Alexandros.Stamatakis@epfl.ch
 *
 *  When publishing work that is based on the results from RAxML-VI-HPC please cite:
 *
 *  Alexandros Stamatakis:"RAxML-VI-HPC: maximum likelihood-based phylogenetic analyses with thousands of taxa and mixed models". 
 *  Bioinformatics 2006; doi: 10.1093/bioinformatics/btl446
 */
#ifndef globalVariables_H
#define globalVariables_H

#ifdef MEMORG




/*
 * listAddr
 * bufflist
 * listAddr2d
 * h_nodep
 * d_nodep
 * d_p0
 *      
 */


float *d2_umpX1;
float *d2_umpX2;
float *d2_left;
float *d2_right;

int transTree = 0;

node *d_p0; //tips + 3*inter
node **d_nodep; // 2*tr->mxtips
node **h_nodep; // 2*tr->mxtips
node *buffList; //tips + 3*inter
node **listAddr; //tips + 3*inter
node **listAddr2d; //tips + 3*inter



int timesCalledNewView = 0, 
    timesCalledNewz = 0,
    timesCalledEvaluate = 0;

float oneKernelTnewz = (float)0;

float T2Devaluate = (float)0; 
float KenrelTevaluate = (float)0; 
float T2Hevaluate = (float)0;

float T2DnewView = (float)0; 
float KenrelTnewView = (float)0; 
float T2HnewView = (float)0;

float T2Dnewz = (float)0; 
float KenrelTnewz = (float)0; 
float T2Hnewz = (float)0;


int alignLength; 
int nofSpecies;


int firstIter = 1;
int trueRes = 0;
int falseRes = 0;

double fullTimeKe =0.0; 
double fullTimeTr = 0.0;

//boolean firstIter = TRUE;

void *yVectorBase;

void *d_globalp; //gpu global pointer
void *globalp; //ram global pointer

void *d_startMisaligned;
void *globalpStart;
void *globalpEnd;
void *d_globalpEnd;
void *d_globalpStart;

size_t dataTransferSizeInit ,dataTransferSizeTestKernel;

size_t dataMallocSize;


size_t EV_size; //pl->evLength == 16
size_t tipVector_size; //pl->tipVectorLength == 64
size_t left_size; // pl->leftLength==16 maxCategories==25
size_t right_size; // pl->rightLength==16 maxCategories==25
size_t patrat_size; // rdta->sites + 1
size_t ei_size; // pl->eiLength == 12
size_t eign_size; // pl->eignLength == 3
size_t rateCategory_size; //cdta->rateCategory= (int *)malloc((rdta->sites + 1) * sizeof(int));
size_t wgt_size; //(int *)malloc((rdta->sites + 1) * sizeof(int));
size_t gammaRates_size;
size_t globalScaler_size;
size_t scalerThread_size;

size_t traversalInfo_size; 
  
size_t yVector_size;

size_t memoryRequirements; // (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width; ////
size_t xVector_size; //tr->innerNodes * memoryRequirements * sizeof(float)////
  
//test mem access
int gCounter =0;
int kk;
//int *d_testData;
//int *testData;
int ydim1 = 10;
int ydim2 = 9;
/*  
  size_t EV_size =  16 * sizeof(float); //pl->evLength == 16
  size_t tipVector_size =  64 * sizeof(float); //pl->tipVectorLength == 64
  size_t left_size = 16 * 25 * sizeof(float); // pl->leftLength==16 maxCategories==25
  size_t right_size = 16 * 25 * sizeof(float); // pl->rightLength==16 maxCategories==25
  size_t patrat_size = (2048 + 1) * sizeof(double); // rdta->sites + 1
  size_t ei_size = 12 * sizeof(double); // pl->eiLength == 12
  size_t eign_size = 3 * sizeof(double); // pl->eignLength == 3
  size_t rateCategory_size = (2048 +1) * sizeof(int); //cdta->rateCategory= (int *)malloc((rdta->sites + 1) * sizeof(int));
  size_t wgt_size = (2048 + 1)*sizeof(int); //(int *)malloc((rdta->sites + 1) * sizeof(int));
  
  size_t traversalInfo_size = 107*sizeof(traversalInfo); //
  
  size_t yVector_size =  (107 + 1) * sizeof(unsigned char *) + (107 + 1) * 4*(2048/4 + 1)* sizeof(unsigned char);
 
  size_t memoryRequirements = (size_t) 4*4*2048; // (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width; ////
  size_t xVector_size = 107*sizeof(float*) + 107*memoryRequirements* sizeof(float); //tr->innerNodes * memoryRequirements * sizeof(float)////

  
  size_t dataTransferSize = sizeof(tree) + wgt_size + yVector_size +  rateCategory_size + patrat_size + ei_size + eign_size + traversalInfo_size + xVector_size + EV_size + tipVector_size + left_size + right_size;

 */
float **xVector_dp;

double * h_dlnLdlz;
double * h_d2lnLdlz2;
double * d_dlnLdlz;
double * d_d2lnLdlz2;

float *d_sumBuffer;
float *d_EV;
float *d_tipVector;
float *d_left;
float *d_right;
double *d_patrat;
double *d_ei;
double *d_eign;
int *d_rateCategory;
int *d_wgt;
double *d_gammaRates;
unsigned int *d_globalScaler;
unsigned int *d_scalerThread;
double *d_partitionLikelihood;
float *d_wr2;
float *d_wr;

float * d_tmpCatSpace;
float * d_tmpDiagSpace;

double *h_partitionLikelihood;
int *h_scalerThread;

tree *d_tree;
traversalInfo *d_ti;
float **d_xVector;
unsigned char **d_yVector;

#endif


#ifdef _IPTOL
double lastCheckpointTime;
double checkPointInterval = 3600.0;
#endif

#ifdef PARALLEL
int numOfWorkers;
#endif

#ifdef _WAYNE_MPI
int processes;
#endif

int processID;
infoList iList;
FILE   *INFILE;

int Thorough = 0;
int timesCalled=0; //addFlag


char run_id[128] = "", 
  workdir[1024] = "", 
  seq_file[1024] = "", 
  tree_file[1024]="", 
  weightFileName[1024] = "", 
  modelFileName[1024] = "", 
  excludeFileName[1024] = "",
  bootStrapFile[1024] = "", 
  permFileName[1024] = "", 
  resultFileName[1024] = "", 
  logFileName[1024] = "", 
  checkpointFileName[1024] = "", 
  infoFileName[1024] = "", 
  randomFileName[1024] = "",   
  bootstrapFileName[1024] = "", 
  bipartitionsFileName[1024] = "",
  bipartitionsFileNameBranchLabels[1024] = "",
  ratesFileName[1024] = "", 
  perSiteLLsFileName[1024] = "", 
  lengthFileName[1024] = "", 
  lengthFileNameModel[1024] = "",
  proteinModelFileName[1024] = "",
  secondaryStructureFileName[1024] = "";

char *protModels[12] = {"DAYHOFF", "DCMUT", "JTT", "MTREV", "WAG", "RTREV", "CPREV", "VT", "BLOSUM62", "MTMAM", "LG", "GTR"};

const char inverseMeaningBINARY[4] = {'_', '0', '1', '-'};
const char inverseMeaningDNA[16]   = {'_', 'A', 'C', 'M', 'G', 'R', 'S', 'V', 'T', 'W', 'Y', 'H', 'K', 'D', 'B', '-'};
const char inverseMeaningPROT[23]  = {'A','R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 
			       'T', 'W', 'Y', 'V', 'B', 'Z', '-'};
const char inverseMeaningGeneric32[33] = {'0', '1', '2', '3', '4', '5', '6', '7', 
				    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
				    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
				    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
				    '-'};
const char inverseMeaningGeneric64[33] = {'0', '1', '2', '3', '4', '5', '6', '7', 
				    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
				    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
				    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
				    '-'};

const unsigned int bitVectorIdentity[256] = {0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,
					     27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,
					     52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,
					     77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,
					     102 ,103 ,104 ,105 ,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,
					     123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,133 ,134 ,135 ,136 ,137 ,138 ,139 ,140 ,141 ,142 ,143 ,
					     144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,152 ,153 ,154 ,155 ,156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,
					     165 ,166 ,167 ,168 ,169 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184 ,185 ,
					     186 ,187 ,188 ,189 ,190 ,191 ,192 ,193 ,194 ,195 ,196 ,197 ,198 ,199 ,200 ,201 ,202 ,203 ,204 ,205 ,206 ,
					     207 ,208 ,209 ,210 ,211 ,212 ,213 ,214 ,215 ,216 ,217 ,218 ,219 ,220 ,221 ,222 ,223 ,224 ,225 ,226 ,227 ,
					     228 ,229 ,230 ,231 ,232 ,233 ,234 ,235 ,236 ,237 ,238 ,239 ,240 ,241 ,242 ,243 ,244 ,245 ,246 ,247 ,248 ,
					     249 ,250 ,251 ,252 ,253 ,254 ,255};



const unsigned int bitVectorAA[23] = {1, 2, 4, 8, 16, 32, 64, 128, 
				      256, 512, 1024, 2048, 4096, 
				      8192, 16384, 32768, 65536, 131072, 262144, 
				      524288, 12 /* N | D */, 96 /*Q | E*/, 1048575 /* - */};

const unsigned int bitVectorSecondary[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
					      10, 11, 12, 13, 14, 15, 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 
					      208, 224, 240, 0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 
					      255, 0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 
					      3584, 3840, 0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 
					      3341, 3598, 3855, 0, 272, 544, 816, 1088, 1360, 1632, 1904, 2176, 2448, 2720, 2992, 
					      3264, 3536, 3808, 4080, 0, 273, 546, 819, 1092, 1365, 1638, 1911, 2184, 2457, 2730, 
					      3003, 3276, 3549, 3822, 4095, 0, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 
					      36864, 40960, 45056, 49152, 53248, 57344, 61440, 0, 4097, 8194, 12291, 16388, 20485, 24582, 
					      28679, 32776, 36873, 40970, 45067, 49164, 53261, 57358, 61455, 0, 4112, 8224, 12336, 16448, 
					      20560, 24672, 28784, 32896, 37008, 41120, 45232, 49344, 53456, 57568, 61680, 0, 4113, 8226, 
					      12339, 16452, 20565, 24678, 28791, 32904, 37017, 41130, 45243, 49356, 53469, 57582, 61695, 
					      0, 4352, 8704, 13056, 17408, 21760, 26112, 30464, 34816, 39168, 43520, 47872, 52224, 56576, 
					      60928, 65280, 0, 4353, 8706, 13059, 17412, 21765, 26118, 30471, 34824, 39177, 43530, 47883, 
					      52236, 56589, 60942, 65295, 0, 4368, 8736, 13104, 17472, 21840, 26208, 30576, 34944, 39312, 
					      43680, 48048, 52416, 56784, 61152, 65520, 0, 4369, 8738, 13107, 17476, 21845, 26214, 30583, 
					      34952, 39321, 43690, 48059, 52428, 56797, 61166, 65535};

const unsigned int bitVector32[33] = {1,     2,    4,    8,   16,   32,    64,   128,
                                      256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                                      65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
                                      16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648u, 
				      4294967295u};

/*const unsigned int bitVector64[65] = {};*/

const unsigned int mask32[32] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 
					262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 
					268435456, 536870912, 1073741824, 2147483648U};

const char *secondaryModelList[21] = { "S6A (GTR)", "S6B", "S6C", "S6D", "S6E", "S7A (GTR)", "S7B", "S7C", "S7D", "S7E", "S7F", "S16 (GTR)", "S16A", "S16B", "S16C", 
				       "S16D", "S16E", "S16F", "S16I", "S16J", "S16K"};

double masterTime;
int partCount = 0;
int optimizeRateCategoryInvocations = 1;





partitionLengths pLengths[MAX_MODEL] = {
  
  /* BINARY */
  {4,   4, 1,  4,  2, 1, 2,  8, 2, 2, FALSE, 3, inverseMeaningBINARY, 2, FALSE, bitVectorIdentity},
  
  /* DNA */
  {16, 16, 3, 16, 12, 6, 4, 64, 6, 4, FALSE, 15, inverseMeaningDNA, 4, FALSE, bitVectorIdentity},
        
  /* AA */
  {400, 400, 19, 400, 380, 190, 20, 460, 190, 20, FALSE, 22, inverseMeaningPROT, 20, TRUE, bitVectorAA},
  
  /* SECONDARY_DATA */

  {256, 256, 15, 256, 240, 120, 16, 4096, 120, 16, FALSE, 255, (char*)NULL, 16, TRUE, bitVectorSecondary},

  
  /* SECONDARY_DATA_6 */
  {36, 36, 5, 36, 30, 15, 6, 384, 15, 6, FALSE, 63, (char*)NULL, 6, TRUE, bitVectorIdentity},

  
  /* SECONDARY_DATA_7 */
  {49,   49,    6,   49, 42,  21, 7, 896, 21, 7, FALSE, 127, (char*)NULL, 7, TRUE, bitVectorIdentity},

  /* 32 states */
  {1024, 1024, 31, 1024, 992, 496, 32, 1056, 496, 32, FALSE, 32, inverseMeaningGeneric32, 32, TRUE, bitVector32},
  
  /* 64 states */
  {4096, 4096, 63, 4096, 4032, 2016, 64, 4160, 64, 2016, FALSE, 64, (char*)NULL, 64, TRUE, (unsigned int*)NULL}
};

partitionLengths pLength;

     




#ifdef _USE_PTHREADS
volatile int             NumberOfJobs;
volatile int             jobCycle;
volatile int             threadJob;
volatile int             NumberOfThreads;
volatile double          *reductionBuffer;
volatile double          *reductionBufferTwo;
volatile double          *reductionBufferThree;
volatile int             *reductionBufferParsimony;
volatile int             *barrierBuffer;

volatile branchInfo      **branchInfos;
pthread_mutex_t          mutex;
#endif
#endif