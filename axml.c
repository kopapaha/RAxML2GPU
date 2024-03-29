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
//test
#ifdef MEMORG
#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#endif

#ifdef WIN32
#include <direct.h>
#endif

#ifndef WIN32
#include <sys/times.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>

#if defined PARALLEL || defined  _WAYNE_MPI
#include <mpi.h>
#endif



#ifdef _USE_PTHREADS
#include <pthread.h>
#endif

#include <xmmintrin.h>


#include "axml.h"
#include "globalVariables.h"

void initKernel(void);
/***************** UTILITY FUNCTIONS **************************/

#ifdef _IPTOL

#include "dmtcpaware.h"

static void dmtcpWriteCheckpoint()
{      
  if(dmtcpIsEnabled())
    { 
      int r;
      double t;
      
      printf(" dmtcp start checkpointing\n");
      t = gettime();
      r = dmtcpCheckpoint();
      
      if(r<=0)  
	printf("Error, checkpointing failed: %d\n",r);
      
      if(r==1)
	printf("***** after checkpoint *****\n");
      
      if(r==2)
	printf("***** after restart *****\n");
      
      printf(" dmtcp end checkpointing: %f seconds\n", gettime() - t);
    }
  else   
    printf("WARNING: dmtcp disabled -- no checkpointing available\n");   
}


void writeCheckpoint(analdef *adef)
{
  double 
    delta,
    t = gettime();

  if((delta = (t - lastCheckpointTime)) > checkPointInterval)
    {
      printf("Writing Checkpoint after %f\n", delta);
      lastCheckpointTime = t;
      dmtcpWriteCheckpoint();
    }

}

#endif

#ifdef _USE_FPGA_LOG

double *temp_lut, 
  con_val, 
  value_minus_inf,
  value_nan,
  value_inf;


typedef union
{
  double value;
  
  struct
  {
    unsigned int rght_p;
    unsigned int lft_p;
  } 
    parts;
  
} ieee_double_shape_type;

static void log_approx_init (int bits_num)
{	
  int  
    man_LUT_sz = pow(2, bits_num),
    i = 0,
    y = 0;  	

  double 
    log_mant = 0.0,
    temp_sum = 0.0;	

  unsigned int   
    t = 0,
    div_i = 0,
    temp_arr2[32],
    temp_arr[32];

  con_val = log(2);
  value_minus_inf = -1.0 / 0.0;
  value_nan = 0.0 / 0.0;
  value_inf = pow(10, 308);
  temp_lut = (double*)malloc(sizeof(double) * man_LUT_sz);

	
  for (t = 0; t < man_LUT_sz; t++)
    {
      div_i = t;		
      for (i = 0; i < 32; i++)
	{
	  temp_arr[i] = div_i & 1;
	  div_i = div_i >> 1;
	}
	
      for(i = 31, y = 0;i > -1; i--, y++)
	temp_arr2[y]=temp_arr[i];

      temp_sum =0.0;
      
      for(i = 31; i > (31 - bits_num); i--)	
	temp_sum = temp_sum + temp_arr2[i] / pow(2, -31 + i + bits_num);	

      log_mant = log(1.0 + temp_sum);
      temp_sum = 0.0;

      temp_lut[t] = log_mant;
    }		
}


static double get_dec_exp (unsigned int input)
{
  input = ((input << 1) >> 21);	
  return (input - 1023.0);	
}

static double get_log_mant (unsigned int input , int bits_num)
{		
  input = (input << 12) >> (12 + 20 - bits_num);	
  return temp_lut[input];	
}

double log_approx (double input)
{	
  double result;

  ieee_double_shape_type model_input;
  
  model_input.value = input;	
  
  if(input<0)    
    result = value_nan;    
  else    
    if(input==0)	
      result = value_minus_inf;	
    else
      {			
	if(input > value_inf)	   
	  result = value_inf + value_inf;	    
	else	    
	  result = con_val * get_dec_exp(model_input.parts.lft_p) + get_log_mant(model_input.parts.lft_p, 12);
	 
      }
	
  return result;
}


#endif

#ifdef _USE_FPGA_EXP

#define arg_FPGAexp   12
#define lutsz_FPGAexp 4096

#define sin_lo_bo  103 // single lower input bound (-103)
#define sin_up_bo  88  // single upper input dound


float as_FPGAexp[25];
float pow_FPGAexp[sin_lo_bo+sin_up_bo];
float lut_FPGAexp[lutsz_FPGAexp];

unsigned int constAND_FPGAexp = 2155872255;

typedef union
{
  float value;
  struct  {  unsigned int lft_p; } parts;  
} ieee_single_shape_type_exp;


static float div2_FPGAexp (float input)
{
   ieee_single_shape_type_exp 
     input_arg;
   
   unsigned int 
     tmp;
   
   input_arg.value = input;

   tmp = ((input_arg.parts.lft_p << 1) >>24);
   
   tmp = tmp - 1;
   tmp = tmp << 23;
   
   input_arg.parts.lft_p = tmp | (input_arg.parts.lft_p & constAND_FPGAexp);

   return input_arg.value;    
}


static unsigned int *fillBitVec_FPGAexp (int input)
{
  unsigned int 
    *tws = (unsigned int*)malloc(sizeof(unsigned int) * arg_FPGAexp),
    tmp=0;
  
  int 
    i = 0;
 
  for(i = arg_FPGAexp - 1; i > -1; i--)
    {
      tmp = input & 1;
      tws[i] = tmp;
      input = input >> 1;     
    }
  
  return tws;
}



static void exp_approx_init(void)
{
  int 
    i = 0,
    j = 0;
  
  float 
    part_mult = 1.0;
  
  double 
    e = 2.718281828459045;
  
  unsigned int 
    *tws;
  
  ieee_single_shape_type_exp 
    tmpval;
 
  /* Initialization of FPGAexpLUT */
  
  as_FPGAexp[0]  =  1.648721270700128;
  as_FPGAexp[1]  =  1.284025416687742; 
  as_FPGAexp[2]  =  1.133148453066826; 
  as_FPGAexp[3]  =  1.064494458917859; 
  as_FPGAexp[4]  =  1.031743407499103; 
  as_FPGAexp[5]  =  1.015747708586686; 
  as_FPGAexp[6]  =  1.007843097206488; 
  as_FPGAexp[7]  =  1.003913889338348; 
  as_FPGAexp[8]  =  1.001955033591003; 
  as_FPGAexp[9]  =  1.000977039492417; 
  as_FPGAexp[10] =  1.000488400478694; 
  as_FPGAexp[11] =  1.000244170429748; 
  as_FPGAexp[12] =  1.000122077763384; 
  
 /* 
    as_FPGAexp[13] =  1.000061037018933; 
    as_FPGAexp[14] =  1.000030518043791; 
    as_FPGAexp[15] =  1.0000152589054785; 
    as_FPGAexp[16] =  1.0000076294236351; 
    as_FPGAexp[17] =  1.0000038147045416; 
    as_FPGAexp[18] =  1.0000019073504518; 
    as_FPGAexp[19] =  1.0000009536747712; 
    as_FPGAexp[20] =  1.0000004768372719; 
    as_FPGAexp[21] =  1.0000002384186075; 
    as_FPGAexp[22] =  1.0000001192092967; 
    as_FPGAexp[23] =  1.0000000596046466; 
    as_FPGAexp[24] =  1.0000000298023228;
 */

  for(i = 0; i < sin_lo_bo + sin_up_bo; i++)  
    pow_FPGAexp[i] = (float)pow(e, i - sin_lo_bo); 
  
  for(i = 0; i < lutsz_FPGAexp; i++)
    {
      tws = fillBitVec_FPGAexp(i);
      part_mult = 1.0;

      for(j = 0; j < arg_FPGAexp; j++)	
	if(1 == tws[j])   	    
	  part_mult *= as_FPGAexp[j];
	            
      tmpval.value = part_mult;
      tmpval.parts.lft_p = (tmpval.parts.lft_p >> 5) << 5; 
      lut_FPGAexp[i] = tmpval.value;
    }
}


/* 
   Unit Evaluation:
   maximum 30DSPs    6 DSPs standard + X more
   4 BRAMs of 36KBs
   1 BRAM of 18Kbs or distributed memory
   latency 164 cycles
*/


double exp_approx(double x)
{
  int 
    i,
    x_int,
    wsint = 0;
  
  float 
    poweroftwos,
    zs, 
    fxs; 
  
  x_int = (int)(floor(x));			//Xilinx Floating Point Operator : float2fixed

  zs = ((float)x) - ((float)x_int); 		// single precision sub

  poweroftwos = 0.5;  				// 32bit register
  wsint = 0; 

  for(i = 0; i < arg_FPGAexp; i++)		// 12 iterations
  {		                             	
    wsint = wsint << 1;				// 12 bit vector
    
    if(poweroftwos < zs)             	// comparator
      {    
	wsint = wsint | 1 ;      
	zs = zs - poweroftwos;            	// sub
      }

    poweroftwos = div2_FPGAexp(poweroftwos);	// as constant on the port b of the comps
  }
  
  fxs = lut_FPGAexp[wsint];			// 1 LUT

  fxs = fxs * ( 1.0 + zs );			// 1 add 1 mult	

  fxs=fxs * pow_FPGAexp[x_int+sin_lo_bo];      // adder , lut and mult

  return ((double)fxs); 
}

#endif




void *malloc_aligned(size_t size) 
{
  void *ptr = (void *)NULL;
  const size_t align = 16;
  int res;
  

#if defined (__APPLE__)
  /* 
     presumably malloc on MACs always returns 
     a 16-byte aligned pointer
  */

  ptr = malloc(size);
  
  if(ptr == (void*)NULL) 
   assert(0);

#else
  res = posix_memalign( &ptr, align, size );

  if(res != 0) 
    assert(0);
#endif 
   
  return ptr;
}



static void printBoth(FILE *f, const char* format, ... )
{
  va_list args;
  va_start(args, format);
  vfprintf(f, format, args );
  va_end(args);

  va_start(args, format);
  vprintf(format, args );
  va_end(args);
}

void printBothOpen(const char* format, ... )
{
  FILE *f = myfopen(infoFileName, "a");

  va_list args;
  va_start(args, format);
  vfprintf(f, format, args );
  va_end(args);

  va_start(args, format);
  vprintf(format, args );
  va_end(args);

  fclose(f);
}

void printBothOpenMPI(const char* format, ... )
{
#ifdef _WAYNE_MPI
  if(processID == 0)
#endif
    {
      FILE *f = myfopen(infoFileName, "a");

      va_list args;
      va_start(args, format);
      vfprintf(f, format, args );
      va_end(args);
      
      va_start(args, format);
      vprintf(format, args );
      va_end(args);
      
      fclose(f);
    }
}


boolean getSmoothFreqs(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].smoothFrequencies;
}

const unsigned int *getBitVector(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].bitVector;
}


int getStates(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].states;
}

int getUndetermined(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].undetermined;
}



char getInverseMeaning(int dataType, unsigned char state)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return  pLengths[dataType].inverseMeaning[state];
}

partitionLengths *getPartitionLengths(pInfo *p)
{
  int 
    dataType  = p->dataType,
    states    = p->states,
    tipLength = p->maxTipStates;

  assert(states != -1 && tipLength != -1);

  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  pLength.leftLength = pLength.rightLength = states * states;
  pLength.eignLength = states -1;
  pLength.evLength   = states * states;
  pLength.eiLength   = states * states - states;
  pLength.substRatesLength = (states * states - states) / 2;
  pLength.frequenciesLength = states;
  pLength.tipVectorLength   = tipLength * states;
  pLength.symmetryVectorLength = (states * states - states) / 2;
  pLength.frequencyGroupingLength = states;
  pLength.nonGTR = FALSE;

  return (&pLengths[dataType]); 
}



static boolean isCat(analdef *adef)
{
  if(adef->model == M_PROTCAT || adef->model == M_GTRCAT || adef->model == M_BINCAT || adef->model == M_32CAT || adef->model == M_64CAT)
    return TRUE;
  else
    return FALSE;
}

static boolean isGamma(analdef *adef)
{
  if(adef->model == M_PROTGAMMA || adef->model == M_GTRGAMMA || adef->model == M_BINGAMMA || 
     adef->model == M_32GAMMA || adef->model == M_64GAMMA)
    return TRUE;
  else
    return FALSE;

}


static int stateAnalyzer(tree *tr, int model, int maxStates)
{
  boolean
    counter[256],
    previous,
    inputError = FALSE;
  
  int
    lower = tr->partitionData[model].lower,
    upper = tr->partitionData[model].upper,
    i,
    j,
    states = 0;

  for(i = 0; i < 256; i++)
    counter[i] = FALSE;

  for(i = 0; i < tr->rdta->numsp; i++)
    {
      unsigned char *yptr =  &(tr->rdta->y0[i * tr->originalCrunchedLength]);

      for(j = lower; j < upper; j++)
	if(yptr[j] != getUndetermined(GENERIC_32))
	  counter[yptr[j]] = TRUE;		
    
    }
  
  for(i = 0; i < maxStates; i++)
    {      
      if(counter[i])
	states++;
    }
  

  previous = counter[0];
  
  for(i = 1; i < 256; i++)
    {
      if(previous == FALSE && counter[i] == TRUE)
	{
	  inputError = TRUE;
	  break;
	}     		
      else
	{
	  if(previous == TRUE && counter[i] ==  FALSE)
	    previous = FALSE;
	}
    }
  
  if(inputError)
    {
      printf("Multi State Error, characters must be used in the order they are available, i.e.\n");
      printf("0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V\n");
      printf("You are using the following caharcters: \n");
      for(i = 0; i < 256; i++)
	if(counter[i])
	  printf("%c ", inverseMeaningGeneric32[i]);
      printf("\n");
      exit(-1);
    }

  return states;
}




static void setRateHetAndDataIncrement(tree *tr, analdef *adef)
{
  int model;

  if(isCat(adef))
    tr->rateHetModel = CAT;
  else
    {
      if(adef->useInvariant)
	tr->rateHetModel = GAMMA_I;
      else
	tr->rateHetModel = GAMMA;
    }

  switch(tr->rateHetModel)
    {
    case GAMMA:
    case GAMMA_I:
      tr->discreteRateCategories = 4;      
      break;
    case CAT:
      if((adef->boot && !adef->bootstrapBranchLengths) || (adef->mode == CLASSIFY_ML) || (tr->catOnly))
	tr->discreteRateCategories = 1; 
      else
	tr->discreteRateCategories = 4;
      break;
    default:
      assert(0);
    }

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      int 
	states = -1,
	maxTipStates = getUndetermined(tr->partitionData[model].dataType) + 1;
      
      switch(tr->partitionData[model].dataType)
	{
	case BINARY_DATA:
	case DNA_DATA:
	case AA_DATA:
	case SECONDARY_DATA:
	case SECONDARY_DATA_6:
	case SECONDARY_DATA_7:
	  states = getStates(tr->partitionData[model].dataType);	 
	  break;	
	case GENERIC_32:
	case GENERIC_64:
	  states = stateAnalyzer(tr, model, getStates(tr->partitionData[model].dataType));	 	 	 	  
	  break;
	default:
	  assert(0);
	}

      tr->partitionData[model].states       = states;
      tr->partitionData[model].maxTipStates = maxTipStates;
    }
}


double gettime(void)
{
#ifdef WIN32
  time_t tp;
  struct tm localtm;
  tp = time(NULL);
  localtm = *localtime(&tp);
  return 60.0*localtm.tm_min + localtm.tm_sec;
#else
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec * 0.000001;
#endif
}

int gettimeSrand(void)
{
#ifdef WIN32
  time_t tp;
  struct tm localtm;
  tp = time(NULL);
  localtm = *localtime(&tp);
  return 24*60*60*localtm.tm_yday + 60*60*localtm.tm_hour + 60*localtm.tm_min  + localtm.tm_sec;
#else
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec;
#endif
}

double randum (long  *seed)
{
  long  sum, mult0, mult1, seed0, seed1, seed2, newseed0, newseed1, newseed2;
  double res;

  mult0 = 1549;
  seed0 = *seed & 4095;
  sum  = mult0 * seed0;
  newseed0 = sum & 4095;
  sum >>= 12;
  seed1 = (*seed >> 12) & 4095;
  mult1 =  406;
  sum += mult0 * seed1 + mult1 * seed0;
  newseed1 = sum & 4095;
  sum >>= 12;
  seed2 = (*seed >> 24) & 255;
  sum += mult0 * seed2 + mult1 * seed1;
  newseed2 = sum & 255;

  *seed = newseed2 << 24 | newseed1 << 12 | newseed0;
  res = 0.00390625 * (newseed2 + 0.000244140625 * (newseed1 + 0.000244140625 * newseed0));

  return res;
}

static int filexists(char *filename)
{
  FILE *fp;
  int res;
  fp = fopen(filename,"r");

  if(fp)
    {
      res = 1;
      fclose(fp);
    }
  else
    res = 0;

  return res;
}


FILE *myfopen(const char *path, const char *mode)
{
  FILE *fp = fopen(path, mode);

  if(strcmp(mode,"r") == 0 || strcmp(mode,"rb") == 0)
    {
      if(fp)
	return fp;
      else
	{
	  if(processID == 0)
	    printf("The file %s you want to open for reading does not exist, exiting ...\n", path);
	  errorExit(-1);
	  return (FILE *)NULL;
	}
    }
  else
    {
      if(fp)
	return fp;
      else
	{
	  if(processID == 0)
	    printf("The file %s RAxML wants to open for writing or appending can not be opened [mode: %s], exiting ...\n",
		   path, mode);
	  errorExit(-1);
	  return (FILE *)NULL;
	}
    }


}


int countTrees(FILE *f)
{
  int numberOfTrees = 0, ch;

  while((ch = fgetc(f)) != EOF)
    {
      if(ch == ';')
	numberOfTrees++;
    }

  rewind(f);

  return numberOfTrees;
}


/********************* END UTILITY FUNCTIONS ********************/


/******************************some functions for the likelihood computation ****************************/


boolean isTip(int number, int maxTips)
{
  assert(number > 0);

  if(number <= maxTips)
    return TRUE;
  else
    return FALSE;
}





void getxsnode (nodeptr p, int model)  
{  
  assert(p->xs[model] || p->next->xs[model] || p->next->next->xs[model]);
  assert(p->xs[model] + p->next->xs[model] + p->next->next->xs[model] == 1);
  
  assert(p == p->next->next->next);

  p->xs[model] = 1;
  
  if(p->next->xs[model])
    {      
      p->next->xs[model] = 0;
      return;
    }
  else
    {
      p->next->next->xs[model] = 0;
      return;
    }  

  assert(0);
}



void getxnode (nodeptr p)
{
  nodeptr  s;

  if ((s = p->next)->x || (s = s->next)->x)
    {
      p->x = s->x;
      s->x = 0;
    }

  assert(p->x);
}





void hookup (nodeptr p, nodeptr q, double *z, int numBranches)
{
  int i;

  p->back = q;
  q->back = p;

  for(i = 0; i < numBranches; i++)
    p->z[i] = q->z[i] = z[i];
}

void hookupDefault (nodeptr p, nodeptr q, int numBranches)
{
  int i;

  p->back = q;
  q->back = p;

  for(i = 0; i < numBranches; i++)
    p->z[i] = q->z[i] = defaultz;
}


/***********************reading and initializing input ******************/

static void getnums (rawdata *rdta)
{
  if (fscanf(INFILE, "%d %d", & rdta->numsp, & rdta->sites) != 2)
    {
      if(processID == 0)
	printf("ERROR: Problem reading number of species and sites\n");
      errorExit(-1);
    }

  if (rdta->numsp < 4)
    {
      if(processID == 0)
	printf("TOO FEW SPECIES\n");
      errorExit(-1);
    }

  if (rdta->sites < 1)
    {
      if(processID == 0)
	printf("TOO FEW SITES\n");
      errorExit(-1);
    }

  return;
}





boolean whitechar (int ch)
{
  return (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r');
}


static void uppercase (int *chptr)
{
  int  ch;

  ch = *chptr;
  if ((ch >= 'a' && ch <= 'i') || (ch >= 'j' && ch <= 'r')
      || (ch >= 's' && ch <= 'z'))
    *chptr = ch + 'A' - 'a';
}


/*

 *     case yVectorPP:
    {
	  tr->partitionData[0].yVector = (unsigned char **)globalp;
	  d_tr_partitionData0_yVector = (unsigned char **)d_globalp;
	  d_globalp = d_tr_partitionData0_yVector + len;
	  globalp = tr->partitionData[0].yVector + len;
	  for( i=0; i<len; i++)
	  {
	    tr->partitionData[0].yVector[i] = (unsigned char *)d_globalp;

	    d_globalp = tr->partitionData[0].yVector[i] + nofsites;
	  }
    }
    break; 
    case yVectorP:
    {
	  tr->partitionData[0].yVector = (unsigned char **)malloc(len*sizeof(unsigned char *));
	  for( i=0; i<len; i++)
	  {
	    tr->partitionData[0].yVector[i] = (unsigned char *)globalp;
	 
	    globalp = tr->partitionData[0].yVector[i] + nofsites;
	  }
    } 
 
 */

static void getyspace (rawdata *rdta)
{
  size_t size = 4 * ((size_t)(rdta->sites / 4 + 1));
  int    i;
  unsigned char *y0;
  
/*
 * #ifdef MEMORG
//step 2  
  printf("getySpace()\n");
    printf("\n!!!!STEP 2\n");
  rdta->y = (unsigned char **) globalp;
  d_yVector = (unsigned char **)d_globalp;
  globalp = rdta->y + (rdta->numsp + 1);
  y0 = (unsigned char *) globalp;
  //rdta->y0 = y0;
  d_globalp = d_yVector +(rdta->numsp + 1);
  	  for( i=0; i<rdta->numsp+1; i++)
	  {
	    rdta->y[i] = (unsigned char *)d_globalp;

	    d_globalp = rdta->y[i] + size;
	  }
  
  
          rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
	  assert(rdta->y); 
          for( i=0; i < rdta->numsp + 1; i++)
	  {
	    rdta->y[i] = (unsigned char *)globalp;
	 
	    globalp = rdta->y[i] + size;
	  }
 */
  
 
//#else
  rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
  assert(rdta->y);   

  y0 = (unsigned char *) malloc(((size_t)(rdta->numsp + 1)) * size * sizeof(unsigned char));
  assert(y0);   

  rdta->y0 = y0;

  for (i = 0; i <= rdta->numsp; i++)
    {
      rdta->y[i] = y0;
      y0 += size;
    }
//#endif
  return;
}


static unsigned int KISS32(void)
{
  static unsigned int 
    x = 123456789, 
    y = 362436069,
    z = 21288629,
    w = 14921776,
    c = 0;

  unsigned int t;

  x += 545925293;
  y ^= (y<<13); 
  y ^= (y>>17); 
  y ^= (y<<5);
  t = z + w + c; 
  z = w; 
  c = (t>>31); 
  w = t & 2147483647;

  return (x+y+w);
}


static void deviceListAlloc(tree *tr, nodeptr p0)
{
    int i, tips, inter;
    cudaError_t error;
    //nodeptr p0 = tr->nodep[1];
    
    tips  = tr->mxtips;  
    inter = tr->mxtips - 1;
    
    listAddr = (node **)malloc((tips + 3*inter) * sizeof(node *));    
    for(i=0; i<tips + 3*inter; i++)
    {
        listAddr[i] = p0++;
    }
    
    
    cudaMalloc((void **)&d_p0, (tips + 3*inter) * sizeof(node));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
        printf("CUDA Error AFTER cudaMalloc d_p0: %s\n", cudaGetErrorString(error));
    // we can't recover from the error -- exit the program
        return;
    }    
    
    
    p0 = d_p0;
    listAddr2d = (node **)malloc((tips + 3*inter) * sizeof(node *));
    for(i=0; i<tips + 3*inter; i++)
    {
        listAddr2d[i] = p0++;
    }    
    
    //TODO check p0 range
    buffList = (node *)malloc((tips + 3*inter) * sizeof(node));

    h_nodep = (node **)malloc((2*tr->mxtips) * sizeof(node *));
    cudaMalloc((void **)&d_nodep, (2*tr->mxtips) * sizeof(node *));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
        printf("CUDA Error AFTER cudaMalloc d_nodep: %s\n", cudaGetErrorString(error));
    // we can't recover from the error -- exit the program
        return;
    }    
    
    
}


static boolean setupTree (tree *tr, analdef *adef)
{
  nodeptr  p0, p, q;
  int
    i,
    j,
    k,
    tips,
    inter; 

  printf("setupTree()\n");
  if(!adef->readTaxaOnly)
    {
      tr->bigCutoff = FALSE;

      tr->patternPosition = (int*)NULL;
      tr->columnPosition = (int*)NULL;

      tr->maxCategories = MAX(4, adef->categories);

      tr->partitionContributions = (double *)malloc(sizeof(double) * tr->NumberOfModels);

      for(i = 0; i < tr->NumberOfModels; i++)
	tr->partitionContributions[i] = -1.0;

      //tr->perPartitionLH = (double *)malloc(sizeof(double) * tr->NumberOfModels);
      tr->storedPerPartitionLH = (double *)malloc(sizeof(double) * tr->NumberOfModels);

      for(i = 0; i < tr->NumberOfModels; i++)
	{
	  tr->perPartitionLH[i] = 0.0;
	  tr->storedPerPartitionLH[i] = 0.0;
	}

      if(adef->grouping)
	tr->grouped = TRUE;
      else
	tr->grouped = FALSE;

      if(adef->constraint)
	tr->constrained = TRUE;
      else
	tr->constrained = FALSE;

      tr->treeID = 0;
    }

  tips  = tr->mxtips;
  inter = tr->mxtips - 1;

  if(!adef->readTaxaOnly)
    {
//#ifdef MEMORG
 //     tr->yVector = (unsigned char **)globalp;
 //     globalp = tr->yVector + tr->mxtips + 1;
//#else
      tr->yVector      = (unsigned char **)  malloc((tr->mxtips + 1) * sizeof(unsigned char *));
//#endif
      tr->fracchanges  = (double *)malloc(tr->NumberOfModels * sizeof(double));
      tr->likelihoods  = (double *)malloc(adef->multipleRuns * sizeof(double));
    }

  tr->numberOfTrees = -1;

  tr->treeStarts  = (char**)NULL;
  tr->treeBuffer = (char*)NULL;

  tr->treeStringLength = tr->mxtips * (nmlngth+128) + 256 + tr->mxtips * 2;

  tr->tree_string  = (char*)calloc(tr->treeStringLength, sizeof(char)); 

  /*TODO, must that be so long ?*/
  
  assert(tr->multiGene == 0);

  if(!adef->readTaxaOnly)
    {
      
      if(tr->multiGene)
	{
	  for(i = 0; i < tr->NumberOfModels; i++)
	    {
	      tr->td[i].count = 0;
	      tr->td[i].ti    = (traversalInfo *)malloc(sizeof(traversalInfo) * tr->mxtips);
	    }
	}
      else
	{
          printf("tr->multiGene == 0?  %d",tr->multiGene);
	  tr->td[0].count = 0;
          //MEM 1
#ifdef MEMORG
          printf("\n!!!!STEP 2\n");
         printf("!!!!! tr->mxtips %d\n", tr->mxtips);
         
         //tr->td[0].ti = (traversalInfo *)globalp;
         tr->td[0].ti = (traversalInfo *)(((uintptr_t)globalp + 16) & ~0x0F);
         //d_ti = d_globalp;
         d_ti = (traversalInfo *)(((uintptr_t)d_globalp + 16) & ~0x0F); 
         globalp = tr->td[0].ti + tr->mxtips; //tr->mxtips==107;
         d_globalp = d_ti + tr->mxtips;
         
#else
          //printf("!!!!! tr->mxtips %d\n", tr->mxtips); 107
         tr->td[0].ti    = (traversalInfo *)malloc(sizeof(traversalInfo) * tr->mxtips); 
#endif
	}

      for(i = 0; i < tr->NumberOfModels; i++)
	tr->fracchanges[i] = -1.0;
      tr->fracchange = -1.0;

      tr->constraintVector = (int *)malloc((2 * tr->mxtips) * sizeof(int));

      tr->nameList = (char **)malloc(sizeof(char *) * (tips + 1));
    }

  if (!(p0 = (nodeptr) malloc((tips + 3*inter) * sizeof(node))))
    {
      printf("ERROR: Unable to obtain sufficient tree memory\n");
      return  FALSE;
    }

  if (!(tr->nodep = (nodeptr *) malloc((2*tr->mxtips) * sizeof(nodeptr))))
    {
      printf("ERROR: Unable to obtain sufficient tree memory, too\n");
      return  FALSE;
    }
  
#ifndef listTransfer
  deviceListAlloc(tr, p0);
#endif
  tr->nodep[0] = (node *) NULL;    /* Use as 1-based array */

  for (i = 1; i <= tips; i++)
    {
      p = p0++;

      p->hash   =  KISS32(); /* hast table stuff */
      p->x      =  0;
      p->number =  i;
      p->next   =  p;
      p->back   = (node *)NULL;
      p->bInf   = (branchInfo *)NULL;

      
      for(k = 0; k < NUM_BRANCHES; k++)
	{
	  p->xs[k]    = 0;
	  p->backs[k] = (nodeptr)NULL;
	}

      for(k = 0; k < VECTOR_LENGTH; k++)
	p->isPresent[k] = 0;

      tr->nodep[i] = p;
    }

  for (i = tips + 1; i <= tips + inter; i++)
    {
      q = (node *) NULL;
      for (j = 1; j <= 3; j++)
	{	 
	  p = p0++;
	  if(j == 1)
	    p->x = 1;
	  else
	    p->x =  0;
	  p->number = i;
	  p->next   = q;
	  p->bInf   = (branchInfo *)NULL;
	  p->back   = (node *) NULL;
	  p->hash   = 0;

	  if(j == 1)
	    for(k = 0; k < NUM_BRANCHES; k++)
	      {
		p->xs[k]    = 1;
		p->backs[k] = (nodeptr)NULL;
	      }
	  else
	    for(k = 0; k < NUM_BRANCHES; k++)
	      {
		p->xs[k]    = 0;
		p->backs[k] = (nodeptr)NULL;
	      }

	  for(k = 0; k < VECTOR_LENGTH; k++)
	    p->isPresent[k] = 0;


	  q = p;
	}
      p->next->next->next = p;
      tr->nodep[i] = p;
    }

  tr->likelihood  = unlikely;
  tr->start       = (node *) NULL;

  for(i = 0; i < NUM_BRANCHES; i++)
    tr->startVector[i]  = (node *) NULL;

  tr->ntips       = 0;
  tr->nextnode    = 0;

  

  if(!adef->readTaxaOnly)
    {
      for(i = 0; i < tr->numBranches; i++)
	tr->partitionSmoothed[i] = FALSE;
    }

  return TRUE;
}


static void checkTaxonName(char *buffer, int len)
{
  int i;

  for(i = 0; i < len - 1; i++)
    {
      boolean valid;

      switch(buffer[i])
	{
	case '\0':
	case '\t':
	case '\n':
	case '\r':
	case ' ':
	case ':':
	case ',':
	case '(':
	case ')':
	case ';':
	case '[':
	case ']':
	  valid = FALSE;
	  break;
	default:
	  valid = TRUE;
	}

      if(!valid)
	{
	  printf("ERROR: Taxon Name \"%s\" is invalid at position %d, it contains illegal character %c\n", buffer, i, buffer[i]);
	  printf("Illegal characters in taxon-names are: tabulators, carriage returns, spaces, \":\", \",\", \")\", \"(\", \";\", \"]\", \"[\"\n");
	  printf("Exiting\n");
	  exit(-1);
	}

    }
  assert(buffer[len - 1] == '\0');
}

static boolean getdata(analdef *adef, rawdata *rdta, tree *tr)
{
  int   
    i, 
    j, 
    basesread, 
    basesnew, 
    ch, my_i, meaning,
    len,
    meaningAA[256], 
    meaningDNA[256], 
    meaningBINARY[256],
    meaningGeneric32[256],
    meaningGeneric64[256];
  
  boolean  
    allread, 
    firstpass;
  
  char 
    buffer[nmlngth + 2];
  
  unsigned char
    genericChars32[32] = {'0', '1', '2', '3', '4', '5', '6', '7', 
			  '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
			  'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
			  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'};  
  unsigned long 
    total = 0,
    gaps  = 0;

  for (i = 0; i < 256; i++)
    {      
      meaningAA[i]          = -1;
      meaningDNA[i]         = -1;
      meaningBINARY[i]      = -1;
      meaningGeneric32[i]   = -1;
      meaningGeneric64[i]   = -1;
    }

  /* generic 32 data */

  for(i = 0; i < 32; i++)
    meaningGeneric32[genericChars32[i]] = i;
  meaningGeneric32['-'] = getUndetermined(GENERIC_32);
  meaningGeneric32['?'] = getUndetermined(GENERIC_32);

  /* AA data */

  meaningAA['A'] =  0;  /* alanine */
  meaningAA['R'] =  1;  /* arginine */
  meaningAA['N'] =  2;  /*  asparagine*/
  meaningAA['D'] =  3;  /* aspartic */
  meaningAA['C'] =  4;  /* cysteine */
  meaningAA['Q'] =  5;  /* glutamine */
  meaningAA['E'] =  6;  /* glutamic */
  meaningAA['G'] =  7;  /* glycine */
  meaningAA['H'] =  8;  /* histidine */
  meaningAA['I'] =  9;  /* isoleucine */
  meaningAA['L'] =  10; /* leucine */
  meaningAA['K'] =  11; /* lysine */
  meaningAA['M'] =  12; /* methionine */
  meaningAA['F'] =  13; /* phenylalanine */
  meaningAA['P'] =  14; /* proline */
  meaningAA['S'] =  15; /* serine */
  meaningAA['T'] =  16; /* threonine */
  meaningAA['W'] =  17; /* tryptophan */
  meaningAA['Y'] =  18; /* tyrosine */
  meaningAA['V'] =  19; /* valine */
  meaningAA['B'] =  20; /* asparagine, aspartic 2 and 3*/
  meaningAA['Z'] =  21; /*21 glutamine glutamic 5 and 6*/

  meaningAA['X'] = 
    meaningAA['?'] = 
    meaningAA['*'] = 
    meaningAA['-'] = 
    getUndetermined(AA_DATA);

  /* DNA data */

  meaningDNA['A'] =  1;
  meaningDNA['B'] = 14;
  meaningDNA['C'] =  2;
  meaningDNA['D'] = 13;
  meaningDNA['G'] =  4;
  meaningDNA['H'] = 11;
  meaningDNA['K'] = 12;
  meaningDNA['M'] =  3;  
  meaningDNA['R'] =  5;
  meaningDNA['S'] =  6;
  meaningDNA['T'] =  8;
  meaningDNA['U'] =  8;
  meaningDNA['V'] =  7;
  meaningDNA['W'] =  9; 
  meaningDNA['Y'] = 10;

  meaningDNA['N'] = 
    meaningDNA['O'] = 
    meaningDNA['X'] = 
    meaningDNA['-'] = 
    meaningDNA['?'] = 
    getUndetermined(DNA_DATA);

  /* BINARY DATA */

  meaningBINARY['0'] = 1;
  meaningBINARY['1'] = 2;
  
  meaningBINARY['-'] = 
    meaningBINARY['?'] = 
    getUndetermined(BINARY_DATA);


  /*******************************************************************/

  basesread = basesnew = 0;

  allread = FALSE;
  firstpass = TRUE;
  ch = ' ';

  while (! allread)
    {
      for (i = 1; i <= tr->mxtips; i++)
	{
	  if (firstpass)
	    {
	      ch = getc(INFILE);
	      while(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r')
		ch = getc(INFILE);

	      my_i = 0;

	      do
		{
		  buffer[my_i] = ch;
		  ch = getc(INFILE);
		  my_i++;
		  if(my_i >= nmlngth)
		    {
		      if(processID == 0)
			{
			  printf("Taxon Name to long at taxon %d, adapt constant nmlngth in\n", i);
			  printf("axml.h, current setting %d\n", nmlngth);
			}
		      errorExit(-1);
		    }
		}
	      while(ch !=  ' ' && ch != '\n' && ch != '\t' && ch != '\r');

	      while(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r')
		ch = getc(INFILE);
	      
	      ungetc(ch, INFILE);

	      buffer[my_i] = '\0';
	      len = strlen(buffer) + 1;
	      checkTaxonName(buffer, len);
	      tr->nameList[i] = (char *)malloc(sizeof(char) * len);
	      strcpy(tr->nameList[i], buffer);
	    }

	  j = basesread;

	  while ((j < rdta->sites) && ((ch = getc(INFILE)) != EOF) && (ch != '\n') && (ch != '\r'))
	    {
	      uppercase(& ch);

	      assert(tr->dataVector[j + 1] != -1);

	      switch(tr->dataVector[j + 1])
		{
		case BINARY_DATA:
		  meaning = meaningBINARY[ch];
		  break;
		case DNA_DATA:
		case SECONDARY_DATA:
		case SECONDARY_DATA_6:
		case SECONDARY_DATA_7:
		  /*
		     still dealing with DNA/RNA here, hence just act if as they where DNA characters
		     corresponding column merging for sec struct models will take place later
		  */
		  meaning = meaningDNA[ch];
		  break;
		case AA_DATA:
		  meaning = meaningAA[ch];
		  break;
		case GENERIC_32:
		  meaning = meaningGeneric32[ch];
		  break;
		case GENERIC_64:
		  meaning = meaningGeneric64[ch];
		  break;
		default:
		  assert(0);
		}

	      if (meaning != -1)
		{
		  j++;
		  rdta->y[i][j] = ch;		 
		}
	      else
		{
		  if(!whitechar(ch))
		    {
		      printf("ERROR: Bad base (%c) at site %d of sequence %d\n",
			     ch, j + 1, i);
		      return FALSE;
		    }
		}
	    }

	  if (ch == EOF)
	    {
	      printf("ERROR: End-of-file at site %d of sequence %d\n", j + 1, i);
	      return  FALSE;
	    }

	  if (! firstpass && (j == basesread))
	    i--;
	  else
	    {
	      if (i == 1)
		basesnew = j;
	      else
		if (j != basesnew)
		  {
		    printf("ERROR: Sequences out of alignment\n");
		    printf("%d (instead of %d) residues read in sequence %d %s\n",
			   j - basesread, basesnew - basesread, i, tr->nameList[i]);
		    return  FALSE;
		  }
	    }
	  while (ch != '\n' && ch != EOF && ch != '\r') ch = getc(INFILE);  /* flush line *//* PC-LINEBREAK*/
	}

      firstpass = FALSE;
      basesread = basesnew;
      allread = (basesread >= rdta->sites);
    }

  for(j = 1; j <= tr->mxtips; j++)
    for(i = 1; i <= rdta->sites; i++)
      {
	assert(tr->dataVector[i] != -1);

	switch(tr->dataVector[i])
	  {
	  case BINARY_DATA:
	    meaning = meaningBINARY[rdta->y[j][i]];
	    if(meaning == getUndetermined(BINARY_DATA))
	      gaps++;
	    break;

	  case SECONDARY_DATA:
	  case SECONDARY_DATA_6:
	  case SECONDARY_DATA_7:
	    assert(tr->secondaryStructurePairs[i - 1] != -1);
	    assert(i - 1 == tr->secondaryStructurePairs[tr->secondaryStructurePairs[i - 1]]);
	    /*
	       don't worry too much about undetermined column count here for sec-struct, just count
	       DNA/RNA gaps here and worry about the rest later-on, falling through to DNA again :-)
	    */
	  case DNA_DATA:
	    meaning = meaningDNA[rdta->y[j][i]];
	    if(meaning == getUndetermined(DNA_DATA))
	      gaps++;
	    break;

	  case AA_DATA:
	    meaning = meaningAA[rdta->y[j][i]];
	    if(meaning == getUndetermined(AA_DATA))
	      gaps++;
	    break;

	  case GENERIC_32:
	    meaning = meaningGeneric32[rdta->y[j][i]];
	    if(meaning == getUndetermined(GENERIC_32))
	      gaps++;
	    break;

	  case GENERIC_64:
	    meaning = meaningGeneric64[rdta->y[j][i]];
	    if(meaning == getUndetermined(GENERIC_64))
	      gaps++;
	    break;
	  default:
	    assert(0);
	  }

	total++;
	rdta->y[j][i] = meaning;
      }

  adef->gapyness = (double)gaps / (double)total;

  


  return  TRUE;
}



static void inputweights (rawdata *rdta)
{
  int i, w, fres;
  FILE *weightFile;
  int *wv = (int *)malloc(sizeof(int) *  rdta->sites);

  weightFile = myfopen(weightFileName, "r");

  i = 0;

  while((fres = fscanf(weightFile,"%d", &w)) != EOF)
    {
      if(!fres)
	{
	  if(processID == 0)
	    printf("error reading weight file probably encountered a non-integer weight value\n");
	  errorExit(-1);
	}
      wv[i] = w;
      i++;
    }

  if(i != rdta->sites)
    {
      if(processID == 0)
	printf("number %d of weights not equal to number %d of alignment columns\n", i, rdta->sites);
      errorExit(-1);
    }

  for(i = 1; i <= rdta->sites; i++)
    rdta->wgt[i] = wv[i - 1];

  fclose(weightFile);
  free(wv);
}



static void getinput(analdef *adef, rawdata *rdta, cruncheddata *cdta, tree *tr)
{
  int i;

  if(!adef->readTaxaOnly)
    {
      INFILE = myfopen(seq_file, "r");
  
      getnums(rdta);
    }

  tr->mxtips            = rdta->numsp;
  
  if(!adef->readTaxaOnly)
    {
      rdta->wgt             = (int *)    malloc((rdta->sites + 1) * sizeof(int));
      cdta->alias           = (int *)    malloc((rdta->sites + 1) * sizeof(int));
#ifdef MEMORG
      printf("\n!!!!STEP 1\n");
      
      //cdta->aliaswgt = (int *)globalp;
      cdta->aliaswgt = (int *)(((uintptr_t)globalp + 16) & ~0x0F);
      //d_wgt = (int *)d_globalp;
      d_wgt = (int *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = cdta->aliaswgt + rdta->sites + 1;
      d_globalp = d_wgt + rdta->sites + 1;
      
      //cdta->patrat = (double *) globalp;
      cdta->patrat = (double *)(((uintptr_t)globalp + 16) & ~0x0F);
      //d_patrat = (double *) d_globalp;
      d_patrat = (double *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = cdta->patrat + rdta->sites + 1;
      d_globalp = d_patrat + rdta->sites + 1;
      
      
      //cdta->rateCategory = (int *) globalp;
      cdta->rateCategory = (int *)(((uintptr_t)globalp + 16) & ~0x0F);
      //d_rateCategory = (int *) d_globalp;
      d_rateCategory = (int *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = cdta->rateCategory + rdta->sites + 1;
      d_globalp = d_rateCategory + rdta->sites + 1;
      
      cdta->wr_FLOAT    = (float *) (((uintptr_t)globalp + 16) & ~0x0F);//malloc((rdta->sites + 1) * sizeof(float));
      d_wr = (float *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = cdta->wr_FLOAT + rdta->sites + 1;
      d_globalp = d_wr + rdta->sites + 1;
      
      cdta->wr2_FLOAT    = (float *) (((uintptr_t)globalp + 16) & ~0x0F);//malloc((rdta->sites + 1) * sizeof(float));
      d_wr2 = (float *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = cdta->wr2_FLOAT + rdta->sites + 1;
      d_globalp = d_wr2 + rdta->sites + 1;      

#else
      cdta->aliaswgt        = (int *)    malloc((rdta->sites + 1) * sizeof(int));
      cdta->patrat          = (double *) malloc((rdta->sites + 1) * sizeof(double));
      cdta->rateCategory    = (int *)    malloc((rdta->sites + 1) * sizeof(int));
      if(adef->useFloat)
	{
	  cdta->wr_FLOAT    = (float *) malloc((rdta->sites + 1) * sizeof(float));
	  cdta->wr2_FLOAT   = (float *) malloc((rdta->sites + 1) * sizeof(float));
	}
#endif

      tr->model             = (int *)    calloc((rdta->sites + 1), sizeof(int));
      tr->initialDataVector  = (int *)    malloc((rdta->sites + 1) * sizeof(int));
      tr->extendedDataVector = (int *)    malloc((rdta->sites + 1) * sizeof(int));
      cdta->wr              = (double *) malloc((rdta->sites + 1) * sizeof(double));
      cdta->wr2             = (double *) malloc((rdta->sites + 1) * sizeof(double));
      cdta->patratStored    = (double *) malloc((rdta->sites + 1) * sizeof(double));


      /*if(adef->useFloat)
	{
	  cdta->wr_FLOAT    = (float *) malloc((rdta->sites + 1) * sizeof(float));
	  cdta->wr2_FLOAT   = (float *) malloc((rdta->sites + 1) * sizeof(float));
	}*/

      if(!adef->useWeightFile)
	{
	  for (i = 1; i <= rdta->sites; i++)
	    rdta->wgt[i] = 1;
	}
      else
	{
	  assert(!adef->useSecondaryStructure);
	  inputweights(rdta);
	}
    }

  tr->multiBranch = 0;
  tr->numBranches = 1;

  if(!adef->readTaxaOnly)
    {
      if(adef->useMultipleModel)
	{
	  int ref;
	  
	  parsePartitions(adef, rdta, tr);
	  
	  for(i = 1; i <= rdta->sites; i++)
	    {
	      ref = tr->model[i];
	      tr->initialDataVector[i] = tr->initialPartitionData[ref].dataType;
	    }
	}
      else
	{
	  int dataType = -1;
	  
	  tr->initialPartitionData  = (pInfo*)malloc(sizeof(pInfo));
	  tr->initialPartitionData[0].partitionName = (char*)malloc(128 * sizeof(char));
	  strcpy(tr->initialPartitionData[0].partitionName, "No Name Provided");
	  
	  tr->initialPartitionData[0].protModels = adef->proteinMatrix;
	  tr->initialPartitionData[0].protFreqs  = adef->protEmpiricalFreqs;
	  
	  
	  tr->NumberOfModels = 1;
	  
	  if(adef->model == M_PROTCAT || adef->model == M_PROTGAMMA)
	    dataType = AA_DATA;
	  if(adef->model == M_GTRCAT || adef->model == M_GTRGAMMA)
	    dataType = DNA_DATA;
	  if(adef->model == M_BINCAT || adef->model == M_BINGAMMA)
	    dataType = BINARY_DATA;
	  if(adef->model == M_32CAT || adef->model == M_32GAMMA)
	    dataType = GENERIC_32;
	  if(adef->model == M_64CAT || adef->model == M_64GAMMA)
	    dataType = GENERIC_64;
	     
	     

	  assert(dataType == BINARY_DATA || dataType == DNA_DATA || dataType == AA_DATA || 
		 dataType == GENERIC_32  || dataType == GENERIC_64);

	  tr->initialPartitionData[0].dataType = dataType;
	  
	  for(i = 0; i <= rdta->sites; i++)
	    {
	      tr->initialDataVector[i] = dataType;
	      tr->model[i]      = 0;
	    }
	}

      if(adef->useSecondaryStructure)
	{
	  memcpy(tr->extendedDataVector, tr->initialDataVector, (rdta->sites + 1) * sizeof(int));
	  
	  tr->extendedPartitionData =(pInfo*)malloc(sizeof(pInfo) * tr->NumberOfModels);
	  
	  for(i = 0; i < tr->NumberOfModels; i++)
	    {
	      tr->extendedPartitionData[i].partitionName = (char*)malloc((strlen(tr->initialPartitionData[i].partitionName) + 1) * sizeof(char));
	      strcpy(tr->extendedPartitionData[i].partitionName, tr->initialPartitionData[i].partitionName);
	      tr->extendedPartitionData[i].dataType   = tr->initialPartitionData[i].dataType;
	      
	      tr->extendedPartitionData[i].protModels = tr->initialPartitionData[i].protModels;
	      tr->extendedPartitionData[i].protFreqs  = tr->initialPartitionData[i].protFreqs;
	    }
	  
	  parseSecondaryStructure(tr, adef, rdta->sites);
	  
	  tr->dataVector    = tr->extendedDataVector;
	  tr->partitionData = tr->extendedPartitionData;
	}
      else
	{
	  tr->dataVector    = tr->initialDataVector;
	  tr->partitionData = tr->initialPartitionData;
	}

      //tr->executeModel   = (boolean *)malloc(sizeof(boolean) * tr->NumberOfModels);

      for(i = 0; i < tr->NumberOfModels; i++)
	tr->executeModel[i] = TRUE;

      getyspace(rdta);
    } 

  setupTree(tr, adef);


  if(!adef->readTaxaOnly)
    {
      if(!getdata(adef, rdta, tr))
	{
	  printf("Problem reading alignment file \n");
	  errorExit(1);
	}
      tr->nameHash = initStringHashTable(10 * tr->mxtips);
      for(i = 1; i <= tr->mxtips; i++)
	addword(tr->nameList[i], tr->nameHash, i);

      fclose(INFILE);
    }
}



static unsigned char buildStates(int secModel, unsigned char v1, unsigned char v2)
{
  unsigned char new = 0;

  switch(secModel)
    {
    case SECONDARY_DATA:
      new = v1;
      new = new << 4;
      new = new | v2;
      break;
    case SECONDARY_DATA_6:
      {
	int
	  meaningDNA[256],
	  i;

	const unsigned char
	  allowedStates[6][2] = {{'A','T'}, {'C', 'G'}, {'G', 'C'}, {'G','T'}, {'T', 'A'}, {'T', 'G'}};

	const unsigned char
	  finalBinaryStates[6] = {1, 2, 4, 8, 16, 32};

	unsigned char
	  intermediateBinaryStates[6];

	int length = 6;

	for(i = 0; i < 256; i++)
	  meaningDNA[i] = -1;

	meaningDNA['A'] =  1;
	meaningDNA['B'] = 14;
	meaningDNA['C'] =  2;
	meaningDNA['D'] = 13;
	meaningDNA['G'] =  4;
	meaningDNA['H'] = 11;
	meaningDNA['K'] = 12;
	meaningDNA['M'] =  3;
	meaningDNA['N'] = 15;
	meaningDNA['O'] = 15;
	meaningDNA['R'] =  5;
	meaningDNA['S'] =  6;
	meaningDNA['T'] =  8;
	meaningDNA['U'] =  8;
	meaningDNA['V'] =  7;
	meaningDNA['W'] =  9;
	meaningDNA['X'] = 15;
	meaningDNA['Y'] = 10;
	meaningDNA['-'] = 15;
	meaningDNA['?'] = 15;

	for(i = 0; i < length; i++)
	  {
	    unsigned char n1 = meaningDNA[allowedStates[i][0]];
	    unsigned char n2 = meaningDNA[allowedStates[i][1]];

	    new = n1;
	    new = new << 4;
	    new = new | n2;

	    intermediateBinaryStates[i] = new;
	  }

	new = v1;
	new = new << 4;
	new = new | v2;

	for(i = 0; i < length; i++)
	  {
	    if(new == intermediateBinaryStates[i])
	      break;
	  }
	if(i < length)
	  new = finalBinaryStates[i];
	else
	  {
	    new = 0;
	    for(i = 0; i < length; i++)
	      {
		if(v1 & meaningDNA[allowedStates[i][0]])
		  {
		    /*printf("Adding %c%c\n", allowedStates[i][0], allowedStates[i][1]);*/
		    new |= finalBinaryStates[i];
		  }
		if(v2 & meaningDNA[allowedStates[i][1]])
		  {
		    /*printf("Adding %c%c\n", allowedStates[i][0], allowedStates[i][1]);*/
		    new |= finalBinaryStates[i];
		  }
	      }
	  }	
      }
      break;
    case SECONDARY_DATA_7:
      {
	int
	  meaningDNA[256],
	  i;

	const unsigned char
	  allowedStates[6][2] = {{'A','T'}, {'C', 'G'}, {'G', 'C'}, {'G','T'}, {'T', 'A'}, {'T', 'G'}};

	const unsigned char
	  finalBinaryStates[7] = {1, 2, 4, 8, 16, 32, 64};

	unsigned char
	  intermediateBinaryStates[7];

	for(i = 0; i < 256; i++)
	  meaningDNA[i] = -1;

	meaningDNA['A'] =  1;
	meaningDNA['B'] = 14;
	meaningDNA['C'] =  2;
	meaningDNA['D'] = 13;
	meaningDNA['G'] =  4;
	meaningDNA['H'] = 11;
	meaningDNA['K'] = 12;
	meaningDNA['M'] =  3;
	meaningDNA['N'] = 15;
	meaningDNA['O'] = 15;
	meaningDNA['R'] =  5;
	meaningDNA['S'] =  6;
	meaningDNA['T'] =  8;
	meaningDNA['U'] =  8;
	meaningDNA['V'] =  7;
	meaningDNA['W'] =  9;
	meaningDNA['X'] = 15;
	meaningDNA['Y'] = 10;
	meaningDNA['-'] = 15;
	meaningDNA['?'] = 15;
	

	for(i = 0; i < 6; i++)
	  {
	    unsigned char n1 = meaningDNA[allowedStates[i][0]];
	    unsigned char n2 = meaningDNA[allowedStates[i][1]];

	    new = n1;
	    new = new << 4;
	    new = new | n2;

	    intermediateBinaryStates[i] = new;
	  }

	new = v1;
	new = new << 4;
	new = new | v2;

	for(i = 0; i < 6; i++)
	  {
	    /* exact match */
	    if(new == intermediateBinaryStates[i])
	      break;
	  }
	if(i < 6)
	  new = finalBinaryStates[i];
	else
	  {
	    /* distinguish between exact mismatches and partial mismatches */

	    for(i = 0; i < 6; i++)
	      if((v1 & meaningDNA[allowedStates[i][0]]) && (v2 & meaningDNA[allowedStates[i][1]]))
		break;
	    if(i < 6)
	      {
		/* printf("partial mismatch\n"); */

		new = 0;
		for(i = 0; i < 6; i++)
		  {
		    if((v1 & meaningDNA[allowedStates[i][0]]) && (v2 & meaningDNA[allowedStates[i][1]]))
		      {
			/*printf("Adding %c%c\n", allowedStates[i][0], allowedStates[i][1]);*/
			new |= finalBinaryStates[i];
		      }
		    else
		      new |=  finalBinaryStates[6];
		  }
	      }
	    else
	      new = finalBinaryStates[6];
	  }	
      }
      break;
    default:
      assert(0);
    }

  return new;

}



static void adaptRdataToSecondary(tree *tr, rawdata *rdta)
{
  int *alias = (int*)calloc(rdta->sites, sizeof(int));
  int i, j, realPosition;  

  for(i = 0; i < rdta->sites; i++)
    alias[i] = -1;

  for(i = 0, realPosition = 0; i < rdta->sites; i++)
    {
      int partner = tr->secondaryStructurePairs[i];
      if(partner != -1)
	{
	  assert(tr->dataVector[i+1] == SECONDARY_DATA || tr->dataVector[i+1] == SECONDARY_DATA_6 || tr->dataVector[i+1] == SECONDARY_DATA_7);

	  if(i < partner)
	    {
	      for(j = 1; j <= rdta->numsp; j++)
		{
		  unsigned char v1 = rdta->y[j][i+1];
		  unsigned char v2 = rdta->y[j][partner+1];

		  assert(i+1 < partner+1);

		  rdta->y[j][i+1] = buildStates(tr->dataVector[i+1], v1, v2);
		}
	      alias[realPosition] = i;
	      realPosition++;
	    }
	}
      else
	{
	  alias[realPosition] = i;
	  realPosition++;
	}
    }

  assert(rdta->sites - realPosition == tr->numberOfSecondaryColumns / 2);

  rdta->sites = realPosition;

  for(i = 0; i < rdta->sites; i++)
    {
      assert(alias[i] != -1);
      tr->model[i+1]    = tr->model[alias[i]+1];
      tr->dataVector[i+1] = tr->dataVector[alias[i]+1];
      rdta->wgt[i+1] =  rdta->wgt[alias[i]+1];

      for(j = 1; j <= rdta->numsp; j++)
	rdta->y[j][i+1] = rdta->y[j][alias[i]+1];
    }

  free(alias);
}

static void sitesort(rawdata *rdta, cruncheddata *cdta, tree *tr, analdef *adef)
{
  int  gap, i, j, jj, jg, k, n, nsp;
  int  
    *index, 
    *category = (int*)NULL;

  boolean  flip, tied;
  unsigned char  **data;

  if(adef->useSecondaryStructure)
    {
      assert(tr->NumberOfModels > 1 && adef->useMultipleModel);

      adaptRdataToSecondary(tr, rdta);
    }

  if(adef->useMultipleModel)    
    category      = tr->model;
  

  index    = cdta->alias;
  data     = rdta->y;
  n        = rdta->sites;
  nsp      = rdta->numsp;
  index[0] = -1;


  if(adef->compressPatterns)
    {
      for (gap = n / 2; gap > 0; gap /= 2)
	{
	  for (i = gap + 1; i <= n; i++)
	    {
	      j = i - gap;

	      do
		{
		  jj = index[j];
		  jg = index[j+gap];
		  if(adef->useMultipleModel)
		    {		     		      
		      assert(category[jj] != -1 &&
			     category[jg] != -1);
		     
		      flip = (category[jj] > category[jg]);
		      tied = (category[jj] == category[jg]);		     

		    }
		  else
		    {
		      flip = 0;
		      tied = 1;
		    }

		  for (k = 1; (k <= nsp) && tied; k++)
		    {
		      flip = (data[k][jj] >  data[k][jg]);
		      tied = (data[k][jj] == data[k][jg]);
		    }

		  if (flip)
		    {
		      index[j]     = jg;
		      index[j+gap] = jj;
		      j -= gap;
		    }
		}
	      while (flip && (j > 0));
	    }
	}
    }
}


static void sitecombcrunch (rawdata *rdta, cruncheddata *cdta, tree *tr, analdef *adef)
{
  int  i, sitei, j, sitej, k;
  boolean  tied;
  int 
    *aliasModel = (int*)NULL,
    *aliasSuperModel = (int*)NULL;

  if(adef->useMultipleModel)
    {
      aliasSuperModel = (int*)malloc(sizeof(int) * (rdta->sites + 1));
      aliasModel      = (int*)malloc(sizeof(int) * (rdta->sites + 1));
    } 

  i = 0;
  cdta->alias[0]    = cdta->alias[1];
  cdta->aliaswgt[0] = 0;

  if(adef->mode == PER_SITE_LL)
    {
      int i;

      tr->patternPosition = (int*)malloc(sizeof(int) * rdta->sites);
      tr->columnPosition  = (int*)malloc(sizeof(int) * rdta->sites);

      for(i = 0; i < rdta->sites; i++)
	{
	  tr->patternPosition[i] = -1;
	  tr->columnPosition[i]  = -1;
	}
    }

  printf("rdtaSites %d\n", rdta->sites);
  assert(rdta->sites == alignLength);
  
  i = 0;
  for (j = 1; j <= rdta->sites; j++)
    {
      sitei = cdta->alias[i];
      sitej = cdta->alias[j];
      if(!adef->compressPatterns)
	tied = 0;
      else
	{
	  if(adef->useMultipleModel)
	    {	     
	      tied = (tr->model[sitei] == tr->model[sitej]);
	      if(tied)
		assert(tr->dataVector[sitei] == tr->dataVector[sitej]);
	    }
	  else
	    tied = 1;
	}

      for (k = 1; tied && (k <= rdta->numsp); k++)
	tied = (rdta->y[k][sitei] == rdta->y[k][sitej]);

      if (tied)
	{
	  if(adef->mode == PER_SITE_LL)
	    {
	      tr->patternPosition[j - 1] = i;
	      tr->columnPosition[j - 1] = sitej;
	      /*printf("Pattern %d from column %d also at site %d\n", i, sitei, sitej);*/
	    }


	  cdta->aliaswgt[i] += rdta->wgt[sitej];
	  if(adef->useMultipleModel)
	    {
	      aliasModel[i]      = tr->model[sitej];
	      aliasSuperModel[i] = tr->dataVector[sitej];
	    }
	}
      else
	{
	  if (cdta->aliaswgt[i] > 0) i++;

	  if(adef->mode == PER_SITE_LL)
	    {
	      tr->patternPosition[j - 1] = i;
	      tr->columnPosition[j - 1] = sitej;
	      /*printf("Pattern %d is from cloumn %d\n", i, sitej);*/
	    }

	  cdta->aliaswgt[i] = rdta->wgt[sitej];
	  cdta->alias[i] = sitej;
	  if(adef->useMultipleModel)
	    {
	      aliasModel[i]      = tr->model[sitej];
	      aliasSuperModel[i] = tr->dataVector[sitej];
	    }
	}
    }

  cdta->endsite = i;
  if (cdta->aliaswgt[i] > 0) cdta->endsite++;

  printf("cdtaendsite %d\n", cdta->endsite);
  alignLength = cdta->endsite;
  assert(cdta->endsite == alignLength);
  
  
  if(adef->mode == PER_SITE_LL)
    {
      for(i = 0; i < rdta->sites; i++)
	{
	  int p  = tr->patternPosition[i];
	  int c  = tr->columnPosition[i];

	  assert(p >= 0 && p < cdta->endsite);
	  assert(c >= 1 && c <= rdta->sites);
	}
    }


  if(adef->useMultipleModel)
    {
      for(i = 0; i <= rdta->sites; i++)
	{
	  tr->model[i]      = aliasModel[i];
	  tr->dataVector[i] = aliasSuperModel[i];
	}
    }

  if(adef->useMultipleModel)
    {
      free(aliasModel);
      free(aliasSuperModel);
    }     
}


static boolean makeweights (analdef *adef, rawdata *rdta, cruncheddata *cdta, tree *tr)
{
  int  i;

  for (i = 1; i <= rdta->sites; i++)
    cdta->alias[i] = i;

  sitesort(rdta, cdta, tr, adef);
  sitecombcrunch(rdta, cdta, tr, adef);

  return TRUE;
}




static boolean makevalues(rawdata *rdta, cruncheddata *cdta, tree *tr, analdef *adef)
{
  int  i, j, model, fullSites = 0, modelCounter;
  printf("makeValues()\n");
  
#ifdef MEMORG
#ifndef old_yVector_mem
  assert(cdta->endsite == alignLength);
  unsigned char **ytmp;
  unsigned char *y;
  //size_t size = 4 * ((size_t)(cdta->endsite / 4 + 1)); //(2048/4+1)4
  size_t size = (size_t)cdta->endsite; //2048

  
  printf("\n!!!!STEP 3\n");
  ytmp = (unsigned char **) globalp;
  globalp = ytmp + (rdta->numsp+1);
  y = (unsigned char *) globalp;
  //y0 = (unsigned char *) globalp;
  //rdta->y0 = y0;
  /*  for(i = 0; i < rdta->numsp; i++)
    tr->yVector[i + 1] = &(rdta->y0[tr->originalCrunchedLength * i]);*/
  d_yVector = (unsigned char **)d_globalp;
  d_globalp = d_yVector +(rdta->numsp +1);
  	  for( i=0; i<rdta->numsp; i++) //SOS!!!
	  {
	    ytmp[i+1] = (unsigned char *)d_globalp;

	    d_globalp = ytmp[i+1] + size;
	  }
  //d_globalp = ytmp[i] + size;
  globalp = y + (rdta->numsp) * size;
  
          //rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
	  //assert(rdta->y); 
          //for( i=0; i < rdta->numsp + 1; i++)
	  //{
	    //rdta->y[i] = (unsigned char *)globalp;
	 
	    //globalp = rdta->y[i] + size;
	  //}
#else
  unsigned char **ytmp;
  unsigned char *y;
  //size_t size = 4 * ((size_t)(cdta->endsite / 4 + 1)); //(2048/4+1)4
  size_t size = (size_t)cdta->endsite; //2048

  
  printf("\n!!!!STEP 3\n");
  void *globalp_y, *d_globalp_y;
  
  globalp_y = yVectorBase;
  ytmp = (unsigned char **) globalp_y;
  globalp_y = ytmp + (rdta->numsp+1);
  y = (unsigned char *) globalp_y;
  //y0 = (unsigned char *) globalp;
  //rdta->y0 = y0;
  /*  for(i = 0; i < rdta->numsp; i++)
    tr->yVector[i + 1] = &(rdta->y0[tr->originalCrunchedLength * i]);*/
  //d_yVector = (unsigned char **)d_globalp;
  d_globalp_y = d_yVector +(rdta->numsp +1);
  	  for( i=0; i<rdta->numsp; i++) //SOS!!!
	  {
	    ytmp[i+1] = (unsigned char *)d_globalp_y;

	    d_globalp_y = ytmp[i+1] + size;
	  }
  //d_globalp = ytmp[i] + size;
  //globalp = y + (rdta->numsp) * size;
  
          //rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
	  //assert(rdta->y); 
          //for( i=0; i < rdta->numsp + 1; i++)
	  //{
	    //rdta->y[i] = (unsigned char *)globalp;
	 
	    //globalp = rdta->y[i] + size;
	  //}  
#endif //old_yVector_mem
  
  
#else
  unsigned char *y    = (unsigned char *)malloc(((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));
#endif//memorg
  unsigned char *yBUF = (unsigned char *)malloc(((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));

  for (i = 1; i <= rdta->numsp; i++)
    for (j = 0; j < cdta->endsite; j++)
      y[((i - 1) * cdta->endsite) + j] = rdta->y[i][cdta->alias[j]];
  
  printf("bfree\n");
  printf("rdta->numsp %d\n, cdta->endsite %d\n", rdta->numsp, cdta->endsite);
  assert(rdta->numsp == nofSpecies);
  assert(cdta->endsite == alignLength);
//#ifndef MEMORG
  free(rdta->y0);
  free(rdta->y);
//#endif
  printf("afree\n");
  
  rdta->y0 = y;
  memcpy(yBUF, y, ((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));
  rdta->yBUF = yBUF;

  if(!adef->useMultipleModel)
    tr->NumberOfModels = 1;

  if(adef->useMultipleModel)
    {
      tr->partitionData[0].lower = 0;

      model        = tr->model[0];
      modelCounter = 0;
     
      i            = 1;

      while(i <  cdta->endsite)
	{
	  if(tr->model[i] != model)
	    {
	      tr->partitionData[modelCounter].upper     = i;
	      tr->partitionData[modelCounter + 1].lower = i;

	      model = tr->model[i];	     
	      modelCounter++;
	    }
	  i++;
	}


      tr->partitionData[tr->NumberOfModels - 1].upper = cdta->endsite;      
    
      for(i = 0; i < tr->NumberOfModels; i++)		  
	tr->partitionData[i].width      = tr->partitionData[i].upper -  tr->partitionData[i].lower;
	 
      model        = tr->model[0];
      modelCounter = 0;
      tr->model[0] = modelCounter;
      i            = 1;
	
      while(i < cdta->endsite)
	{	 
	  if(tr->model[i] != model)
	    {
	      model = tr->model[i];
	      modelCounter++;
	      tr->model[i] = modelCounter;
	    }
	  else
	    tr->model[i] = modelCounter;
	  i++;
	}      
    }
  else
    {
      tr->partitionData[0].lower = 0;
      tr->partitionData[0].upper = cdta->endsite;
      tr->partitionData[0].width =  tr->partitionData[0].upper -  tr->partitionData[0].lower;
    }

  tr->rdta       = rdta;
  tr->cdta       = cdta;

  tr->invariant          = (int *)malloc(cdta->endsite * sizeof(int));
  tr->originalDataVector = (int *)malloc(cdta->endsite * sizeof(int));
  tr->originalModel      = (int *)malloc(cdta->endsite * sizeof(int));
  tr->originalWeights    = (int *)malloc(cdta->endsite * sizeof(int));

  memcpy(tr->originalModel, tr->model,            cdta->endsite * sizeof(int));
  memcpy(tr->originalDataVector, tr->dataVector,  cdta->endsite * sizeof(int));
  memcpy(tr->originalWeights, tr->cdta->aliaswgt, cdta->endsite * sizeof(int));

  tr->originalCrunchedLength = tr->cdta->endsite;
  for(i = 0; i < tr->cdta->endsite; i++)
    fullSites += tr->cdta->aliaswgt[i];

  tr->fullSites = fullSites;

  for(i = 0; i < rdta->numsp; i++)
    tr->yVector[i + 1] = &(rdta->y0[tr->originalCrunchedLength * i]);

  return TRUE;
}


static void makeMissingData(tree *tr)
{
  if(tr->multiGene)
    {
      int 
	model, 
	i, 
	j;
      
      double
	totalWidth = 0.0,
	missingWidth = 0.0;
      
      unsigned char 
	undetermined;       

#ifdef _USE_PTHREADS
      assert(0);
#endif

      assert(tr->NumberOfModels > 1 && tr->numBranches > 1);

      for(model = 0; model < tr->NumberOfModels; model++)
	{
	  int 
	    countMissing = 0,
	    width = tr->partitionData[model].upper - tr->partitionData[model].lower;	

	  tr->mxtipsVector[model] = 0;
	  
	  undetermined = getUndetermined(tr->partitionData[model].dataType);	  

	  for(i = 1; i <= tr->mxtips; i++)
	    {
	      unsigned char *tip = tr->partitionData[model].yVector[i];	      
	      

	      assert(width > 0);

	      for(j = 0; j < width; j++)
		if(tip[j] != undetermined)
		  break;

	      if(j == width)				 
		countMissing++;		
	      else
		{
		  tr->nodep[i]->isPresent[model / MASK_LENGTH] |= mask32[model % MASK_LENGTH];
		  if(!tr->startVector[model])
		    {
		      tr->startVector[model] =  tr->nodep[i];
		      /*printf("placing VR into terminal branch %d\n", i);*/
		    }
		}
	    }

	  tr->mxtipsVector[model] = tr->mxtips - countMissing;
	  assert( tr->mxtipsVector[model] + countMissing == tr->mxtips);

	  printBothOpen("Partition %d has %d missing taxa and %d present taxa\n\n", model, countMissing, tr->mxtipsVector[model]);
	  assert(countMissing < tr->mxtips);

	  totalWidth   += (double)(tr->mxtips) * (double)(width);
	  missingWidth += (double)(countMissing) * (double)(width);
	}

      printBothOpen("Percentage of gene-sampling induced gappyness in this alignment: %2.2f%s\n", 100 * (missingWidth / totalWidth), "%");

    }
}






static int sequenceSimilarity(unsigned char *tipJ, unsigned char *tipK, int n)
{
  int i;

  for(i = 0; i < n; i++)
    if(*tipJ++ != *tipK++)
      return 0;

  return 1;
}

static void checkSequences(tree *tr, rawdata *rdta, analdef *adef)
{
  int n = tr->mxtips + 1;
  int i, j;
  int *omissionList     = (int *)calloc(n, sizeof(int));
  int *undeterminedList = (int *)calloc((rdta->sites + 1), sizeof(int));
  int *modelList        = (int *)malloc((rdta->sites + 1)* sizeof(int));
  int count = 0;
  int countNameDuplicates = 0;
  int countUndeterminedColumns = 0;
  int countOnlyGaps = 0;
  int modelCounter = 1;
  unsigned char *tipI, *tipJ;

  for(i = 1; i < n; i++)
    {
      for(j = i + 1; j < n; j++)
	if(strcmp(tr->nameList[i], tr->nameList[j]) == 0)
	  {
	    countNameDuplicates++;
	    if(processID == 0)
	      printBothOpen("Sequence names of taxon %d and %d are identical, they are both called %s\n", i, j, tr->nameList[i]);
	  }
    }

  if(countNameDuplicates > 0)
    {
      if(processID == 0)
	printBothOpen("ERROR: Found %d taxa that had equal names in the alignment, exiting...\n", countNameDuplicates);
      errorExit(-1);
    }

  for(i = 1; i < n; i++)
    {
      j = 1;

      while(j <= rdta->sites)
	{	  
	  if(rdta->y[i][j] != getUndetermined(tr->dataVector[j]))
	    break;	  	  

	  j++;
	}

      if(j == (rdta->sites + 1))
	{
	  if(processID == 0)
	    printBothOpen("ERROR: Sequence %s consists entirely of undetermined values which will be treated as missing data\n",
			  tr->nameList[i]);

	  countOnlyGaps++;
	}
    }

  if(countOnlyGaps > 0)
    {
      if(processID == 0)
	printBothOpen("ERROR: Found %d sequences that consist entirely of undetermined values, exiting...\n", countOnlyGaps);

      errorExit(-1);
    }

  for(i = 0; i <= rdta->sites; i++)
    modelList[i] = -1;

  for(i = 1; i <= rdta->sites; i++)
    {
      j = 1;

      while(j < n)
	{
	  if(rdta->y[j][i] != getUndetermined(tr->dataVector[i]))
	    break;

	  
	  j++;
	}

      if(j == n)
	{
	  undeterminedList[i] = 1;

	  if(processID == 0)
	    printBothOpen("IMPORTANT WARNING: Alignment column %d contains only undetermined values which will be treated as missing data\n", i);

	  countUndeterminedColumns++;
	}
      else
	{
	  if(adef->useMultipleModel)
	    {
	      modelList[modelCounter] = tr->model[i];
	      modelCounter++;
	    }
	}
    }


  for(i = 1; i < n; i++)
    {
      if(omissionList[i] == 0)
	{
	  tipI = &(rdta->y[i][1]);

	  for(j = i + 1; j < n; j++)
	    {
	      if(omissionList[j] == 0)
		{
		  tipJ = &(rdta->y[j][1]);
		  if(sequenceSimilarity(tipI, tipJ, rdta->sites))
		    {
		      if(processID == 0)
			printBothOpen("\n\nIMPORTANT WARNING: Sequences %s and %s are exactly identical\n", tr->nameList[i], tr->nameList[j]);

		      omissionList[j] = 1;
		      count++;
		    }
		}
	    }
	}
    }

  if(count > 0 || countUndeterminedColumns > 0)
    {
      char noDupFile[2048];
      char noDupModels[2048];
      char noDupSecondary[2048];

      if(count > 0 &&processID == 0)
	{
	  printBothOpen("\nIMPORTANT WARNING\n");

	  printBothOpen("Found %d %s that %s exactly identical to other sequences in the alignment.\n", count, (count == 1)?"sequence":"sequences", (count == 1)?"is":"are");

	  printBothOpen("Normally they should be excluded from the analysis.\n\n");
	}

      if(countUndeterminedColumns > 0 && processID == 0)
	{
	  printBothOpen("\nIMPORTANT WARNING\n");

	  printBothOpen("Found %d %s that %s only undetermined values which will be treated as missing data.\n",
			countUndeterminedColumns, (countUndeterminedColumns == 1)?"column":"columns", (countUndeterminedColumns == 1)?"contains":"contain");

	  printBothOpen("Normally these columns should be excluded from the analysis.\n\n");
	}

      strcpy(noDupFile, seq_file);
      strcat(noDupFile, ".reduced");

      strcpy(noDupModels, modelFileName);
      strcat(noDupModels, ".reduced");

      strcpy(noDupSecondary, secondaryStructureFileName);
      strcat(noDupSecondary, ".reduced");

      if(processID == 0)
	{
	  if(adef->useSecondaryStructure)
	    {
	      if(countUndeterminedColumns && !filexists(noDupSecondary))
		{
		  FILE *newFile = myfopen(noDupSecondary, "w");
		  int count;

		  printBothOpen("\nJust in case you might need it, a secondary structure file with \n");
		  printBothOpen("structure assignments for undetermined columns removed is printed to file %s\n",noDupSecondary);

		  for(i = 1, count = 0; i <= rdta->sites; i++)
		    {
		      if(undeterminedList[i] == 0)
			fprintf(newFile, "%c", tr->secondaryStructureInput[i - 1]);
		      else
			count++;
		    }

		  assert(count == countUndeterminedColumns);

		  fprintf(newFile,"\n");

		  fclose(newFile);
		}
	      else
		{
		  if(countUndeterminedColumns)
		    {
		      printBothOpen("\nA secondary structure file with model assignments for undetermined\n");
		      printBothOpen("columns removed has already been printed to  file %s\n",noDupSecondary);
		    }
		}
	    }


	  if(adef->useMultipleModel && !filexists(noDupModels) && countUndeterminedColumns)
	    {
	      FILE *newFile = myfopen(noDupModels, "w");

	      printBothOpen("\nJust in case you might need it, a mixed model file with \n");
	      printBothOpen("model assignments for undetermined columns removed is printed to file %s\n",noDupModels);

	      for(i = 0; i < tr->NumberOfModels; i++)
		{
		  boolean modelStillExists = FALSE;

		  for(j = 1; (j <= rdta->sites) && (!modelStillExists); j++)
		    {
		      if(modelList[j] == i)
			modelStillExists = TRUE;
		    }

		  if(modelStillExists)
		    {
		      int k = 1;
		      int lower, upper;
		      int parts = 0;


		      switch(tr->partitionData[i].dataType)
			{
			case AA_DATA:
			  {
			    char AAmodel[1024];

			    strcpy(AAmodel, protModels[tr->partitionData[i].protModels]);
			    if(tr->partitionData[i].protFreqs)
			      strcat(AAmodel, "F");

			    fprintf(newFile, "%s, ", AAmodel);
			  }
			  break;
			case DNA_DATA:
			  fprintf(newFile, "DNA, ");
			  break;
			case BINARY_DATA:
			  fprintf(newFile, "BIN, ");
			  break;
			case GENERIC_32:
			  fprintf(newFile, "MULTI, ");
			  break;
			case GENERIC_64:
			  fprintf(newFile, "CODON, ");
			  break;
			default:
			  assert(0);
			}

		      fprintf(newFile, "%s = ", tr->partitionData[i].partitionName);

		      while(k <= rdta->sites)
			{
			  if(modelList[k] == i)
			    {
			      lower = k;
			      while((modelList[k + 1] == i) && (k <= rdta->sites))
				k++;
			      upper = k;

			      if(lower == upper)
				{
				  if(parts == 0)
				    fprintf(newFile, "%d", lower);
				  else
				    fprintf(newFile, ",%d", lower);
				}
			      else
				{
				  if(parts == 0)
				    fprintf(newFile, "%d-%d", lower, upper);
				  else
				    fprintf(newFile, ",%d-%d", lower, upper);
				}
			      parts++;
			    }
			  k++;
			}
		      fprintf(newFile, "\n");
		    }
		}
	      fclose(newFile);
	    }
	  else
	    {
	      if(adef->useMultipleModel)
		{
		  printBothOpen("\nA mixed model file with model assignments for undetermined\n");
		  printBothOpen("columns removed has already been printed to  file %s\n",noDupModels);
		}
	    }


	  if(!filexists(noDupFile))
	    {
	      FILE *newFile;

	      printBothOpen("Just in case you might need it, an alignment file with \n");
	      if(count && !countUndeterminedColumns)
		printBothOpen("sequence duplicates removed is printed to file %s\n", noDupFile);
	      if(!count && countUndeterminedColumns)
		printBothOpen("undetermined columns removed is printed to file %s\n", noDupFile);
	      if(count && countUndeterminedColumns)
		printBothOpen("sequence duplicates and undetermined columns removed is printed to file %s\n", noDupFile);

	      newFile = myfopen(noDupFile, "w");

	      fprintf(newFile, "%d %d\n", tr->mxtips - count, rdta->sites - countUndeterminedColumns);

	      for(i = 1; i < n; i++)
		{
		  if(!omissionList[i])
		    {
		      fprintf(newFile, "%s ", tr->nameList[i]);
		      tipI =  &(rdta->y[i][1]);

		      for(j = 0; j < rdta->sites; j++)
			{
			  if(undeterminedList[j + 1] == 0)			    
			    fprintf(newFile, "%c", getInverseMeaning(tr->dataVector[j + 1], tipI[j]));			      			     			 
			}

		      fprintf(newFile, "\n");
		    }
		}

	      fclose(newFile);
	    }
	  else
	    {
	      if(count && !countUndeterminedColumns)
		printBothOpen("An alignment file with sequence duplicates removed has already\n");
	      if(!count && countUndeterminedColumns)
		printBothOpen("An alignment file with undetermined columns removed has already\n");
	      if(count && countUndeterminedColumns)
		printBothOpen("An alignment file with undetermined columns and sequence duplicates removed has already\n");

	      printBothOpen("been printed to file %s\n",  noDupFile);
	    }
	}
    }

  free(undeterminedList);
  free(omissionList);
  free(modelList);
}







static void generateBS(tree *tr, analdef *adef)
{
  int i, j, k, w;
  int count;
  char outName[1024], buf[16];
  FILE *of;

  assert(adef->boot != 0);

  for(i = 0; i < adef->multipleRuns; i++)
    {
      computeNextReplicate(tr, &adef->boot, (int*)NULL, (int*)NULL, FALSE);

      count = 0;
      for(j = 0; j < tr->cdta->endsite; j++)
	count += tr->cdta->aliaswgt[j];

      assert(count == tr->rdta->sites);

      strcpy(outName, workdir);
      strcat(outName, seq_file);
      strcat(outName, ".BS");
      sprintf(buf, "%d", i);
      strcat(outName, buf);
      printf("Printing replicate %d to %s\n", i, outName);

      of = myfopen(outName, "w");

      fprintf(of, "%d %d\n", tr->mxtips, count);

      for(j = 1; j <= tr->mxtips; j++)
	{
	  unsigned char *tip   =  tr->yVector[tr->nodep[j]->number];
	  fprintf(of, "%s ", tr->nameList[j]);

	  for(k = 0; k < tr->cdta->endsite; k++)
	    {
	      for(w = 0; w < tr->cdta->aliaswgt[k]; w++)
		fprintf(of, "%c", getInverseMeaning(tr->dataVector[k], tip[k]));	      
	    }

	  fprintf(of, "\n");
	}
      fclose(of);
    }
}





static void splitMultiGene(tree *tr, rawdata *rdta)
{
  int i, l;
  int n = rdta->sites + 1;
  int *modelFilter = (int *)malloc(sizeof(int) * n);
  int length, k;
  unsigned char *tip;
  FILE *outf;
  char outFileName[2048];
  char buf[16];

  for(i = 0; i < tr->NumberOfModels; i++)
    {
      strcpy(outFileName, seq_file);
      sprintf(buf, "%d", i);
      strcat(outFileName, ".GENE.");
      strcat(outFileName, buf);
      outf = myfopen(outFileName, "w");
      length = 0;
      for(k = 1; k < n; k++)
	{
	  if(tr->model[k] == i)
	    {
	      modelFilter[k] = 1;
	      length++;
	    }
	  else
	    modelFilter[k] = -1;
	}

      fprintf(outf, "%d %d\n", rdta->numsp, length);

      for(l = 1; l <= rdta->numsp; l++)
	{
	  fprintf(outf, "%s ", tr->nameList[l]);

	  tip = &(rdta->y[l][0]);

	  for(k = 1; k < n; k++)
	    {
	      if(modelFilter[k] == 1)		
		fprintf(outf, "%c", getInverseMeaning(tr->dataVector[k], tip[k]));		 	     
	    }
	  fprintf(outf, "\n");

	}

      fclose(outf);

      printf("Wrote individual gene/partition alignment to file %s\n", outFileName);
    }

  free(modelFilter);
  printf("Wrote all %d individual gene/partition alignments\n", tr->NumberOfModels);
  printf("Exiting normally\n");
}


static int countTaxaInTopology(void)
{
  FILE *f = myfopen(tree_file, "r");   

  int
    c,   
    taxaCount = 0;

  while((c = fgetc(f)) != EOF)
    {
      if(c == '(' || c == ',')
	{
	  c = fgetc(f);
	  if(c ==  '(' || c == ',')
	    ungetc(c, f);
	  else
	    {	      	      	  	      
	      do
		{		
		  c = fgetc(f);
		}
	      while(c != ':' && c != ')' && c != ',');	    

	      taxaCount++;	     	     
	    
	      ungetc(c, f);
	    }
	}
    }
 
  printf("Found a total of %d taxa in tree file %s\n", taxaCount, tree_file);

  fclose(f);

  return taxaCount;
}







static void allocPartitions(tree *tr)
{
  int
    i,ii,
    maxCategories = tr->maxCategories;

  for(i = 0; i < tr->NumberOfModels; i++)
    {
      const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[i]));
#ifndef MEMORG
      if(tr->useFastScaling)
	tr->partitionData[i].globalScaler    = (unsigned int *)calloc(2 * tr->mxtips, sizeof(unsigned int));
#endif
      printf("pl->eiLength %d,\n pl->eignLength %d,\n \npl->evLength %d,\n pl->tipVectorLength %d,\n pl->leftLength %d,\npl->rightLength %d,\nmaxCategories %d\n",pl->eiLength, pl->eignLength, pl->evLength, pl->tipVectorLength, pl->leftLength, pl->rightLength, maxCategories);

      assert(tr->NumberOfModels == 1);
      assert(pl->eiLength == 12);
      assert(pl->eignLength == 3);
      assert(pl->evLength == 16);
      assert(pl->tipVectorLength == 64);
      assert(pl->leftLength == 16);
      assert(pl->rightLength == 16);
      assert(maxCategories == 25);
      
      
      
      
      tr->partitionData[i].left              = (double *)malloc_aligned(pl->leftLength * maxCategories * sizeof(double));
      tr->partitionData[i].right             = (double *)malloc_aligned(pl->rightLength * maxCategories * sizeof(double));
#ifdef MEMORG
      printf("STEP 3a-\n");
      //tr->partitionData[i].EIGN = (double *) globalp;
      tr->partitionData[i].EIGN = (double *)(((uintptr_t)globalp + 16) & ~0x0F);
      //d_eign = (double *) d_globalp;
      d_eign = (double *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = tr->partitionData[i].EIGN + pl->eignLength;
      d_globalp = d_eign + pl->eignLength;
      
      //tr->partitionData[i].EI = (double *) globalp;
      tr->partitionData[i].EI = (double *)(((uintptr_t)globalp + 16) & ~0x0F);
      //d_ei = (double *) d_globalp;
      d_ei = (double *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = tr->partitionData[i].EI + pl->eiLength;
      d_globalp = d_ei + pl->eiLength;
      
#else
      tr->partitionData[i].EIGN              = (double*)malloc(pl->eignLength * sizeof(double));
      tr->partitionData[i].EI                = (double*)malloc(pl->eiLength * sizeof(double));

#endif
      tr->partitionData[i].EV                = (double*)malloc_aligned(pl->evLength * sizeof(double));
      tr->partitionData[i].substRates        = (double *)malloc(pl->substRatesLength * sizeof(double));
      tr->partitionData[i].frequencies       = (double*)malloc(pl->frequenciesLength * sizeof(double));
      tr->partitionData[i].tipVector         = (double *)malloc_aligned(pl->tipVectorLength * sizeof(double));
      tr->partitionData[i].symmetryVector    = (int *)malloc(pl->symmetryVectorLength  * sizeof(int));
      tr->partitionData[i].frequencyGrouping = (int *)malloc(pl->frequencyGroupingLength  * sizeof(int));
      
      tr->partitionData[i].nonGTR = FALSE;
       
      if(tr->useFloat)
	{
#ifdef MEMORG
          printf("STEP 3a\n");
          int tmpi;
          
          //tr->partitionData[i].EV_FLOAT = (float *)globalp;
          tr->partitionData[i].EV_FLOAT = (float *)(((uintptr_t)globalp + 16) & ~0x0F); 
          globalp = tr->partitionData[i].EV_FLOAT + pl->evLength;
          //d_EV = (float *) d_globalp;
          d_EV = (float *)(((uintptr_t)d_globalp + 16) & ~0x0F);
          d_globalp = d_EV + pl->evLength;
               
          for (tmpi=0; tmpi<pl->evLength; tmpi++)
          {
              tr->partitionData[i].EV_FLOAT[tmpi] = (float)0;
          }
          
          
          //tr->partitionData[i].tipVector_FLOAT = (float *)globalp;
          tr->partitionData[i].tipVector_FLOAT = (float *)(((uintptr_t)globalp + 16) & ~0x0F);
          globalp = tr->partitionData[i].tipVector_FLOAT + pl->tipVectorLength;
          //d_tipVector = d_globalp;
          d_tipVector = (float *)(((uintptr_t)d_globalp + 16) & ~0x0F);
          d_globalp = d_tipVector + pl->tipVectorLength;

          for (tmpi=0; tmpi<pl->tipVectorLength; tmpi++)
          {
              tr->partitionData[i].tipVector_FLOAT[tmpi] = (float)0;
          }          
          
          
          //to gammaRates htan eksw apo to if usefloat
          //tr->partitionData[i].gammaRates = globalp;
          tr->partitionData[i].gammaRates = (double *)(((uintptr_t)globalp + 16) & ~0x0F);
          globalp = tr->partitionData[i].gammaRates + 4;
          //d_gammaRates = (double *) d_globalp;
          d_gammaRates = (double *)(((uintptr_t)d_globalp + 16) & ~0x0F);
          d_globalp = d_gammaRates + 4;
          
          
          //to globalScaler dhlwmeno me calloc panw arxika
          //tr->partitionData[i].globalScaler    = (unsigned int *)calloc(2 * tr->mxtips, sizeof(unsigned int));
          //tr->partitionData[i].globalScaler = globalp;
          tr->partitionData[i].globalScaler = (unsigned int *)(((uintptr_t)globalp + 16) & ~0x0F);
          globalp = tr->partitionData[i].globalScaler + 2 * tr->mxtips;
          
          for(tmpi=0; tmpi < 2 * tr->mxtips; tmpi++)
          {
              tr->partitionData[i].globalScaler[tmpi] = 0;
          }
          //d_globalScaler = (unsigned int *) d_globalp;
          d_globalScaler = (unsigned int *)(((uintptr_t)d_globalp + 16) & ~0x0F);
          d_globalp = d_globalScaler + 2 * tr->mxtips;
          
          
          //tr->partitionData[i].left_FLOAT = (float *)globalp;
          tr->partitionData[i].left_FLOAT = (float *)(((uintptr_t)globalp + 16) & ~0x0F);
          globalp = tr->partitionData[i].left_FLOAT + pl->leftLength * maxCategories;
          //d_left = d_globalp;
          d_left = (float *)(((uintptr_t)d_globalp + 16) & ~0x0F); 
          d_globalp = d_left + pl->leftLength * maxCategories;         
          
          for (tmpi=0; tmpi<pl->leftLength * maxCategories; tmpi++)
          {
              tr->partitionData[i].left_FLOAT[tmpi] = (float)0;
          }
                    
          
          //tr->partitionData[i].right_FLOAT = (float *)globalp;
          tr->partitionData[i].right_FLOAT = (float *)(((uintptr_t)globalp + 16) & ~0x0F);
          globalp = tr->partitionData[i].right_FLOAT + pl->rightLength * maxCategories;
          //d_right = d_globalp;
          d_right = (float *)(((uintptr_t)d_globalp + 16) & ~0x0F);
          d_globalp = d_right + pl->rightLength * maxCategories;         
          
          for (tmpi=0; tmpi<pl->rightLength * maxCategories; tmpi++)
          {
              tr->partitionData[i].right_FLOAT[tmpi] = (float)0;
          }          
          //test mem access
          //testData = (int *) malloc(10*sizeof(int));

          /*
          testData = globalp;
          globalp = testData +10;
          d_testData = d_globalp;
          d_globalp = d_testData + 10;
          
          for(kk = 0; kk<10; kk++)
          {
              testData[kk] = kk;
          }
          */
#else
          tr->partitionData[i].EV_FLOAT          = (float *)malloc_aligned(pl->evLength * sizeof(float));
	  tr->partitionData[i].tipVector_FLOAT   = (float *)malloc_aligned(pl->tipVectorLength * sizeof(float));
	  tr->partitionData[i].left_FLOAT        = (float *)malloc_aligned(pl->leftLength * maxCategories * sizeof(float));
	  tr->partitionData[i].right_FLOAT       = (float *)malloc_aligned(pl->rightLength * maxCategories * sizeof(float));
#endif
        }
      
#ifndef MEMORG
      tr->partitionData[i].gammaRates = (double*)malloc(sizeof(double) * 4);
#endif
      
//#ifdef MEMORG
     // globalp = tr->partitionData[i].yVector;
      //globalp = tr->partitionData[i].yVector + tr->mxtips + 1;
      //gpu
      //
      //finaly cpu
      //tr->partitionData[i].yVector = (unsigned char **)malloc(sizeof(unsigned char*) * (tr->mxtips + 1));

//#else
      tr->partitionData[i].yVector = (unsigned char **)malloc(sizeof(unsigned char*) * (tr->mxtips + 1));
//#endif
      if(!tr->useFloat)
	tr->partitionData[i].xVector = (double **)malloc(sizeof(double*) * tr->innerNodes);
      else
      {
#ifdef MEMORG
          /*        
         rdta->y = (unsigned char **) globalp;
          //d_tr_partitionData0_yVector = (unsigned char **)d_globalp;
         globalp = rdta->y + (rdta->numsp + 1);
         y0 = (unsigned char *) globalp;
        //d_globalp = d_tr_partitionData0_yVector +(rdta->numsp + 1);
  	  for( i=0; i<rdta->numsp+1; i++)
	  {
	    rdta->y[i] = (unsigned char *)d_globalp;

	    d_globalp = rdta->y[i] + size;
	  }
  
  
          rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
	  for( i=0; i<rdta->numsp+1; i++)
	  {
	    rdta->y[i] = (unsigned char *)globalp;
	 
	    globalp = rdta->y[i] + size;
	  }
           */
      printf("\n!!!!! tr->innerNodes %d\n", (int)tr->innerNodes);
      printf("\n!!!!STEP 4\n");
          
      size_t width = tr->partitionData[0].upper - tr->partitionData[0].lower; ////
      size_t memoryRequirements = (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[0].states) * width;////
      printf("\nmemoryRequirements for xVectors: %zu bytes\n", memoryRequirements);

      //tr->partitionData[0].xVector_FLOAT = (float **)globalp;////
      tr->partitionData[0].xVector_FLOAT = (float **)(((uintptr_t)globalp + 16) & ~0x0F);
      xVector_dp = tr->partitionData[0].xVector_FLOAT;
      //d_xVector = (float **)d_globalp;
      d_xVector = (float **)(((uintptr_t)d_globalp + 16) & ~0x0F);
      globalp = tr->partitionData[0].xVector_FLOAT + tr->innerNodes;////
      d_globalp = d_xVector + tr->innerNodes;
      d_globalp = (void *)(((uintptr_t)d_globalp + 16) & ~0x0F);
      for (ii=0; ii<tr->innerNodes; ii++)
      {
          tr->partitionData[0].xVector_FLOAT[ii] = (float *)d_globalp;
          d_globalp = tr->partitionData[0].xVector_FLOAT[ii] +  memoryRequirements;
      }
      
     tr->partitionData[0].xVector_FLOAT = (float **)malloc(sizeof(float*) * tr->innerNodes);////
     //goto for() tr->partitionData[model].xVector_FLOAT[i]   = &likelihoodArray_FLOAT[i * memoryRequirements];


#else
	tr->partitionData[i].xVector_FLOAT = (float **)malloc(sizeof(float*) * tr->innerNodes);
#endif
      }
      tr->partitionData[i].pVector = (parsimonyVector **)malloc(sizeof(parsimonyVector*) * tr->innerNodes);
      if(!tr->useFastScaling)
	tr->partitionData[i].expVector = (int **)malloc(sizeof(int*) * tr->innerNodes);

      tr->partitionData[i].mxtips  = tr->mxtips;

#ifdef _USE_PTHREADS
      tr->partitionData[i].yVector = (unsigned char **)malloc(sizeof(unsigned char*) * (tr->mxtips + 1));
#else
      {
	int j;

	for(j = 1; j <= tr->mxtips; j++)

	  tr->partitionData[i].yVector[j] = &(tr->yVector[j][tr->partitionData[i].lower]);
/*#ifdef MEMORG
        //memcpy();
        unsigned char **ytmp;
        ytmp = (unsigned char **) globalp;
        globalp = ytmp + (107 + 1);
        
        d_yVector =(unsigned char **) d_globalp;
        d_globalp = d_yVector + (107 + 1);
  	  for( i=0; i<107+1; i++)
	  {
	    ytmp[i] = (unsigned char *)d_globalp;

	    d_globalp = ytmp[i] + 4*(2048/4 + 1);
	  }
        
        for(j = 0; j < tr->mxtips+1; j++)
        {
            
            tr->partitionData[0].yVector[j] = globalp;
            globalp = tr->partitionData[0].yVector[j] + 4*(2048/4 + 1);
            
        }
        
       for(j = 0; j < tr->mxtips+1; j++)
	  memcpy(tr->partitionData[0].yVector[j],  &(tr->yVector[j][tr->partitionData[0].lower]), (size_t)4*(2048/4 + 1) );
#else
        
#endif
      */
      
      
      
      }
#endif

    }
}

#ifndef _USE_PTHREADS





static void allocNodex (tree *tr)
{
  size_t
    i,   
    model,
    offset,
    memoryRequirements = 0;
  int    
    *expArray = (int*)NULL;

  double *likelihoodArray = (double*)NULL;
  float  *likelihoodArray_FLOAT = (float*)NULL;  

  allocPartitions(tr);



  for(model = 0; model < (size_t)tr->NumberOfModels; model++)
    {
      size_t width = tr->partitionData[model].upper - tr->partitionData[model].lower;
      printf("\n!!!!!! tr->discreteRateCategories %d\n",tr->discreteRateCategories);
      printf("\n!!!!!! tr->partitionData[model].states %d\n",tr->partitionData[model].states);
      printf("\n!!!!!! width %d\n",(int)width);
      printf("\n!!!!!! widthUPPER %d\n",tr->partitionData[model].upper);
      printf("\n!!!!!! widthLOWER %d\n",tr->partitionData[model].lower);

      
      assert(tr->discreteRateCategories == 4);
      assert(tr->partitionData[model].states == 4);
      assert(width == alignLength);
      assert(tr->partitionData[model].upper == alignLength);
      assert(tr->partitionData[model].lower == 0);
      
      assert(tr->innerNodes == nofSpecies);
      
      memoryRequirements += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;      
    }

  tr->perSiteLL       = (double *)malloc((size_t)tr->cdta->endsite * sizeof(double));
  assert(tr->perSiteLL != NULL);

  

  if(!tr->multiGene)
    {
      if(!tr->useFloat)
	{
	  likelihoodArray = (double *)malloc_aligned(tr->innerNodes * memoryRequirements * sizeof(double));
	  assert(likelihoodArray != NULL);
	}
      else
	{
          
          
          /*
           
             rdta->y = (unsigned char **) globalp;
  //d_tr_partitionData0_yVector = (unsigned char **)d_globalp;
  globalp = rdta->y + (rdta->numsp + 1);
  y0 = (unsigned char *) globalp;
  //d_globalp = d_tr_partitionData0_yVector +(rdta->numsp + 1);
  	  for( i=0; i<rdta->numsp+1; i++)
	  {
	    rdta->y[i] = (unsigned char *)d_globalp;

	    d_globalp = rdta->y[i] + size;
	  }
  
  
          rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
	  for( i=0; i<rdta->numsp+1; i++)
	  {
	    rdta->y[i] = (unsigned char *)globalp;
	 
	    globalp = rdta->y[i] + size;
	  }
           
           */
          
#ifdef MEMORG
          printf("\n!!!!STEP 4a\n");
          //likelihoodArray_FLOAT = (float *) globalp;////

          likelihoodArray_FLOAT = (float *)(((uintptr_t)globalp + 16) & ~0x0F);
          globalp = likelihoodArray_FLOAT + tr->innerNodes * memoryRequirements;////
          globalpEnd = globalp;
          d_globalpEnd = d_globalp;    
          int tmpi;
          for (tmpi=0; tmpi< tr->innerNodes*memoryRequirements; tmpi++)
          {
              likelihoodArray_FLOAT[tmpi] = (float)0;
          }
          
          /*char *memPtr;
          for (memPtr=globalpStart; memPtr!=globalpEnd; memPtr = memPtr+1)
          {
              printf("in\n");
              *memPtr = (char)((uintptr_t)(*memPtr) & 0x00);
          }*/
          
          
        cudaError_t error;  
        cudaMalloc((void **)&d_sumBuffer, memoryRequirements * sizeof(float));
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_sumBuffer: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return;
	}
        
        
        
        h_dlnLdlz = (double *)malloc(alignLength*sizeof(double));
        h_d2lnLdlz2 = (double *)malloc(alignLength*sizeof(double));

        cudaMalloc((void **)&d_dlnLdlz, alignLength*sizeof(double));
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_dlnLdlz: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return;
	}        
        cudaMalloc((void **)&d_d2lnLdlz2, alignLength*sizeof(double));
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_d2lnLdlz2: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return;
	}        
        
        
        cudaMalloc((void **)&d_tmpCatSpace, 25*4*alignLength*sizeof(float));
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_tmpCatSpace: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return;
	}
        
        cudaMalloc((void **)&d_tmpDiagSpace, 64*alignLength*sizeof(float));
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_tmpDiagSpace: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return;
	}        
#else
	  likelihoodArray_FLOAT = (float *)malloc_aligned(tr->innerNodes * memoryRequirements * sizeof(float));
	  assert(likelihoodArray_FLOAT != NULL);
#endif
	}
    }
  
  if(!tr->multiGene)
    {
      if(!tr->useFastScaling)
	{
	  expArray = (int *)malloc((size_t)tr->cdta->endsite * tr->innerNodes * sizeof(int));
	  assert(expArray != NULL);
	}
    }

  if(!tr->useFloat)
    {
      tr->sumBuffer  = (double *)malloc_aligned(memoryRequirements * sizeof(double));
      assert(tr->sumBuffer != NULL);
    }
  else
    {
      tr->sumBuffer_FLOAT  = (float *)malloc_aligned(memoryRequirements * sizeof(float));
      assert(tr->sumBuffer_FLOAT != NULL);
    }

  assert(4 * sizeof(double) > sizeof(parsimonyVector));

  offset = 0;

  /* C-OPT for initial testing tr->NumberOfModels will be 1 */

  for(model = 0; model < (size_t)tr->NumberOfModels; model++)
    {
      size_t lower = tr->partitionData[model].lower;
      size_t width = tr->partitionData[model].upper - lower;

      /* TODO all of this must be reset/adapted when fixModelIndices is called ! */

      if(!tr->useFloat)
	tr->partitionData[model].sumBuffer       = &tr->sumBuffer[offset];
      else
	tr->partitionData[model].sumBuffer_FLOAT = &tr->sumBuffer_FLOAT[offset];

      tr->partitionData[model].perSiteLL    = &tr->perSiteLL[lower];

      /* do something about this ! */

      tr->partitionData[model].wr           = &tr->cdta->wr[lower];
      tr->partitionData[model].wr2          = &tr->cdta->wr2[lower];

      if(tr->useFloat)
	{
	  tr->partitionData[model].wr_FLOAT           = &tr->cdta->wr_FLOAT[lower];
	  tr->partitionData[model].wr2_FLOAT          = &tr->cdta->wr2_FLOAT[lower];
	}


      tr->partitionData[model].wgt          = &tr->cdta->aliaswgt[lower];
#ifdef MEMORG
      //to gpu
      //base_d_wgt; //sta8era sth vash ths aliaswgt
      //d_tr_partitionData0_wgt = base_d_wgt + lower;
#endif

      tr->partitionData[model].invariant    = &tr->invariant[lower];
      tr->partitionData[model].rateCategory = &tr->cdta->rateCategory[lower];

      offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;      
    }



  for(i = 0; i < tr->innerNodes; i++)
    {
      offset = 0;

      for(model = 0; model < (size_t)tr->NumberOfModels; model++)
	{
	  size_t width = tr->partitionData[model].upper - tr->partitionData[model].lower;

	  if(!tr->multiGene)
	    {
	      if(!tr->useFastScaling)
		tr->partitionData[model].expVector[i] = &expArray[i * tr->cdta->endsite + tr->partitionData[model].lower];	    
	  
	      if(!tr->useFloat)
		{
		  tr->partitionData[model].xVector[i]   = &likelihoodArray[i * memoryRequirements + offset];
		  tr->partitionData[model].pVector[i]   = (parsimonyVector *)tr->partitionData[model].xVector[i];
		}
	      else
		{
		  tr->partitionData[model].xVector_FLOAT[i]   = &likelihoodArray_FLOAT[i * memoryRequirements + offset];
		  tr->partitionData[model].pVector[i]   = (parsimonyVector *)tr->partitionData[model].xVector_FLOAT[i];
		}
	    }


	  offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;
	 
	}
    }
  #ifdef MEMORG
  int j;
  for(i=0; i<tr->innerNodes; i++)
  {
      for(j=0; j<memoryRequirements; j++)
      {
          tr->partitionData[0].xVector_FLOAT[i][j] = (float)0;
      }
      
  }
#endif
}

#endif


static void initAdef(analdef *adef)
{  
  adef->useSecondaryStructure  = FALSE;
  adef->bootstrapBranchLengths = FALSE;
  adef->model                  = M_GTRCAT;
  adef->max_rearrange          = 21;
  adef->stepwidth              = 5;
  adef->initial                = adef->bestTrav = 10;
  adef->initialSet             = FALSE;
  adef->restart                = FALSE;
  adef->mode                   = BIG_RAPID_MODE;
  adef->categories             = 25;
  adef->boot                   = 0;
  adef->rapidBoot              = 0;
  adef->useWeightFile          = FALSE;
  adef->checkpoints            = 0;
  adef->startingTreeOnly       = 0;
  adef->multipleRuns           = 1;
  adef->useMultipleModel       = FALSE;
  adef->likelihoodEpsilon      = 0.1;
  adef->constraint             = FALSE;
  adef->grouping               = FALSE;
  adef->randomStartingTree     = FALSE;
  adef->parsimonySeed          = 0;
  adef->proteinMatrix          = JTT;
  adef->protEmpiricalFreqs     = 0;
  adef->outgroup               = FALSE;
  adef->useInvariant           = FALSE;
  adef->permuteTreeoptimize    = FALSE;
  adef->useInvariant           = FALSE;
  adef->allInOne               = FALSE;
  adef->likelihoodTest         = FALSE;
  adef->perGeneBranchLengths   = FALSE;
  adef->generateBS             = FALSE;
  adef->bootStopping           = FALSE;
  adef->gapyness               = 0.0;
  adef->similarityFilterMode   = 0;
  adef->useExcludeFile         = FALSE;
  adef->userProteinModel       = FALSE;
  adef->externalAAMatrix       = (double*)NULL;
  adef->computeELW             = FALSE;
  adef->computeDistance        = FALSE;
  adef->thoroughInsertion      = FALSE;
  adef->compressPatterns       = TRUE;
  adef->useFloat               = FALSE;
  adef->readTaxaOnly           = FALSE;
  adef->meshSearch             = 0;
  adef->shSupports             = FALSE;
}




static int modelExists(char *model, analdef *adef)
{
  int i;
  char thisModel[1024];

  /********** BINARY ********************/

   if(strcmp(model, "BINGAMMAI\0") == 0)
    {
      adef->model = M_BINGAMMA;
      adef->useInvariant = TRUE;
      return 1;
    }

  if(strcmp(model, "BINGAMMA\0") == 0)
    {
      adef->model = M_BINGAMMA;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "BINCAT\0") == 0)
    {
      adef->model = M_BINCAT;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "BINCATI\0") == 0)
    {
      adef->model = M_BINCAT;
      adef->useInvariant = TRUE;
      return 1;
    }

  /*********** 32 state ****************************/

  if(strcmp(model, "MULTIGAMMAI\0") == 0)
    {
      adef->model = M_32GAMMA;
      adef->useInvariant = TRUE;
      return 1;
    }

  if(strcmp(model, "MULTIGAMMA\0") == 0)
    {
      adef->model = M_32GAMMA;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "MULTICAT\0") == 0)
    {
      adef->model = M_32CAT;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "MULTICATI\0") == 0)
    {
      adef->model = M_32CAT;
      adef->useInvariant = TRUE;
      return 1;
    }

  /*********** 64 state ****************************/

  if(strcmp(model, "CODONGAMMAI\0") == 0)
    {
      adef->model = M_64GAMMA;
      adef->useInvariant = TRUE;
      return 1;
    }

  if(strcmp(model, "CODONGAMMA\0") == 0)
    {
      adef->model = M_64GAMMA;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "CODONCAT\0") == 0)
    {
      adef->model = M_64CAT;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "CODONCATI\0") == 0)
    {
      adef->model = M_64CAT;
      adef->useInvariant = TRUE;
      return 1;
    }


  /*********** DNA **********************/

  if(strcmp(model, "GTRGAMMAI\0") == 0)
    {
      adef->model = M_GTRGAMMA;
      adef->useInvariant = TRUE;
      return 1;
    }

  if(strcmp(model, "GTRGAMMA\0") == 0)
    {
      adef->model = M_GTRGAMMA;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "GTRGAMMA_FLOAT\0") == 0)
    {
      adef->model = M_GTRGAMMA;
      adef->useInvariant = FALSE;
      adef->useFloat = TRUE;
      return 1;
    }

  if(strcmp(model, "GTRCAT\0") == 0)
    {
      adef->model = M_GTRCAT;
      adef->useInvariant = FALSE;
      return 1;
    }

   if(strcmp(model, "GTRCAT_FLOAT\0") == 0)
    {
      adef->model = M_GTRCAT;
      adef->useInvariant = FALSE;
      adef->useFloat = TRUE;
      return 1;

    }

  if(strcmp(model, "GTRCATI\0") == 0)
    {
      adef->model = M_GTRCAT;
      adef->useInvariant = TRUE;
      return 1;
    }




  /*************** AA GTR ********************/

  /* TODO empirical FREQS */

  if(strcmp(model, "PROTCATGTR\0") == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = GTR;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "PROTCATIGTR\0") == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = GTR;
      adef->useInvariant = TRUE;
      return 1;
    }

  if(strcmp(model, "PROTGAMMAGTR\0") == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = GTR;
      adef->useInvariant = FALSE;
      return 1;
    }

  if(strcmp(model, "PROTGAMMAIGTR\0") == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = GTR;
      adef->useInvariant = TRUE;
      return 1;
    }

  /****************** AA ************************/

  for(i = 0; i < NUM_PROT_MODELS - 1; i++)
    {
      /* check CAT */

      strcpy(thisModel, "PROTCAT");
      strcat(thisModel, protModels[i]);

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTCAT;
	  adef->proteinMatrix = i;
	  return 1;
	}

      /* check CATF */

      strcpy(thisModel, "PROTCAT");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "F");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTCAT;
	  adef->proteinMatrix = i;
	  adef->protEmpiricalFreqs = 1;
	  return 1;
	}

      /* check CAT FLOAT */

      strcpy(thisModel, "PROTCAT");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "_FLOAT");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTCAT;
	  adef->proteinMatrix = i;
	  adef->useFloat = TRUE;
	  return 1;
	}

      /* check CATF FLOAT */

      strcpy(thisModel, "PROTCAT");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "F");
      strcat(thisModel, "_FLOAT");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTCAT;
	  adef->proteinMatrix = i;
	  adef->protEmpiricalFreqs = 1;
	  adef->useFloat = TRUE;
	  return 1;
	}

      /* check CATI */

      strcpy(thisModel, "PROTCATI");
      strcat(thisModel, protModels[i]);

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTCAT;
	  adef->proteinMatrix = i;
	  adef->useInvariant = TRUE;
	  return 1;
	}

      /* check CATIF */

      strcpy(thisModel, "PROTCATI");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "F");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTCAT;
	  adef->proteinMatrix = i;
	  adef->protEmpiricalFreqs = 1;
	  adef->useInvariant = TRUE;
	  return 1;
	}


      /****************check GAMMA ************************/

      strcpy(thisModel, "PROTGAMMA");
      strcat(thisModel, protModels[i]);

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTGAMMA;
	  adef->proteinMatrix = i;
	  adef->useInvariant = FALSE;
	  return 1;
	}

      /* check GAMMA FLOAT */

      strcpy(thisModel, "PROTGAMMA");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "_FLOAT");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTGAMMA;
	  adef->proteinMatrix = i;
	  adef->useFloat = TRUE;
	  adef->useInvariant = FALSE;
	  return 1;
	}


      /*check GAMMAI*/

      strcpy(thisModel, "PROTGAMMAI");
      strcat(thisModel, protModels[i]);

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTGAMMA;
	  adef->proteinMatrix = i;
	  adef->useInvariant = TRUE;
	  return 1;
	}


      /* check GAMMAmodelF */

      strcpy(thisModel, "PROTGAMMA");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "F");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTGAMMA;
	  adef->proteinMatrix = i;
	  adef->protEmpiricalFreqs = 1;
	  adef->useInvariant = FALSE;
	  return 1;
	}

      /* check GAMMAmodelF FLOAT*/

      strcpy(thisModel, "PROTGAMMA");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "F");
      strcat(thisModel, "_FLOAT");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTGAMMA;
	  adef->proteinMatrix = i;
	  adef->protEmpiricalFreqs = 1;
	  adef->useInvariant = FALSE;
	  adef->useFloat = TRUE;
	  return 1;
	}

      /* check GAMMAImodelF */

      strcpy(thisModel, "PROTGAMMAI");
      strcat(thisModel, protModels[i]);
      strcat(thisModel, "F");

      if(strcmp(model, thisModel) == 0)
	{
	  adef->model = M_PROTGAMMA;
	  adef->proteinMatrix = i;
	  adef->protEmpiricalFreqs = 1;
	  adef->useInvariant = TRUE;
	  return 1;
	}

    }

  /*********************************************************************************/



  return 0;
}



static int mygetopt(int argc, char **argv, char *opts, int *optind, char **optarg)
{
  static int sp = 1;
  register int c;
  register char *cp;

  if(sp == 1)
    {
      if(*optind >= argc || argv[*optind][0] != '-' || argv[*optind][1] == '\0')
	return -1;
    }
  else
    {
      if(strcmp(argv[*optind], "--") == 0)
	{
	  *optind =  *optind + 1;
	  return -1;
	}
    }

  c = argv[*optind][sp];
  if(c == ':' || (cp=strchr(opts, c)) == 0)
    {
      printf(": illegal option -- %c \n", c);
      if(argv[*optind][++sp] == '\0')
	{
	  *optind =  *optind + 1;
	  sp = 1;
	}
      return('?');
    }
  if(*++cp == ':')
    {
      if(argv[*optind][sp+1] != '\0')
	{
	  *optarg = &argv[*optind][sp+1];
	  *optind =  *optind + 1;
	}
      else
	{
	  *optind =  *optind + 1;
	  if(*optind >= argc)
	    {
	      printf(": option requires an argument -- %c\n", c);
	      sp = 1;
	      return('?');
	    }
	  else
	    {
	      *optarg = argv[*optind];
	      *optind =  *optind + 1;
	    }
	}
      sp = 1;
    }
  else
    {
      if(argv[*optind][++sp] == '\0')
	{
	  sp = 1;
	  *optind =  *optind + 1;
	}
      *optarg = 0;
    }
  return(c);
  }

static void checkOutgroups(tree *tr, analdef *adef)
{
  if(adef->outgroup)
    {
      boolean found;
      int i, j;

      for(j = 0; j < tr->numberOfOutgroups; j++)
	{
	  found = FALSE;
	  for(i = 1; (i <= tr->mxtips) && !found; i++)
	    {
	      if(strcmp(tr->nameList[i], tr->outgroups[j]) == 0)
		{
		  tr->outgroupNums[j] = i;
		  found = TRUE;
		}
	    }
	  if(!found)
	    {
	      printf("Error, the outgroup name \"%s\" you specified can not be found in the alignment, exiting ....\n", tr->outgroups[j]);
	      errorExit(-1);
	    }
	}
    }

}

static void parseOutgroups(char *outgr, tree *tr)
{
  int count = 1, i, k;
  char name[nmlngth];

  i = 0;
  while(outgr[i] != '\0')
    {
      if(outgr[i] == ',')
	count++;
      i++;
    }

  tr->numberOfOutgroups = count;

  tr->outgroups = (char **)malloc(sizeof(char *) * count);

  for(i = 0; i < tr->numberOfOutgroups; i++)
    tr->outgroups[i] = (char *)malloc(sizeof(char) * nmlngth);

  tr->outgroupNums = (int *)malloc(sizeof(int) * count);

  i = 0;
  k = 0;
  count = 0;
  while(outgr[i] != '\0')
    {
      if(outgr[i] == ',')
	{
	  name[k] = '\0';
	  strcpy(tr->outgroups[count], name);
	  count++;
	  k = 0;
	}
      else
	{
	  name[k] = outgr[i];
	  k++;
	}
      i++;
    }

  name[k] = '\0';
  strcpy(tr->outgroups[count], name);

  /*for(i = 0; i < tr->numberOfOutgroups; i++)
    printf("%d %s \n", i, tr->outgroups[i]);*/


  /*printf("%s \n", name);*/
}


/*********************************** OUTGROUP STUFF END *********************************************************/


static void printVersionInfo(void)
{
  printf("\n\nThis is %s version %s released by Alexandros Stamatakis in %s.\n\n",  programName, programVersion, programDate);
  printf("With greatly appreciated code contributions by:\n");
  printf("Andre Aberer (TUM)\n");     
  printf("Simon Berger (TUM)\n");
  printf("John Cazes (TACC)\n");
  printf("Michael Ott (TUM)\n"); 
  printf("Nick Pattengale (UNM)\n"); 
  printf("Wayne Pfeiffer (SDSC)\n\n");
}

static void printMinusFUsage(void)
{
  printf("\n");
  printf("              \"-f a\": rapid Bootstrap analysis and search for best-scoring ML tree in one program run\n");  

  printf("              \"-f b\": draw bipartition information on a tree provided with \"-t\" based on multiple trees\n");
  printf("                      (e.g., from a bootstrap) in a file specifed by \"-z\"\n");

  printf("              \"-f c\": check if the alignment can be properly read by RAxML\n");

  printf("              \"-f d\": new rapid hill-climbing \n");
  printf("                      DEFAULT: ON\n");

  printf("              \"-f e\": optimize model+branch lengths for given input tree under GAMMA/GAMMAI only\n");

  printf("              \"-f E\": execute very fast experimental tree search, at present only for testing\n");

  printf("              \"-f F\": execute fast experimental tree search, at present only for testing\n");

  printf("              \"-f g\": compute per site log Likelihoods for one ore more trees passed via\n");
  printf("                      \"-z\" and write them to a file that can be read by CONSEL\n");
  printf("                        WARNING: does not print likelihoods in the original column order\n");
  printf("              \"-f h\": compute log likelihood test (SH-test) between best tree passed via \"-t\"\n");
  printf("                      and a bunch of other trees passed via \"-z\" \n");  

  printf("              \"-f i\": EXPERIMENTAL do not use for real tree inferences: conducts a single cycle of fast lazy SPR moves\n");
  printf("                      on a given input tree, to be used in combination with -C and -M \n");
  
  printf("              \"-f I\": EXPERIMENTAL do not use for real tree inferences: conducts a single cycle of thorough lazy SPR moves\n");
  printf("                      on a given input tree, to be used in combination with -C and -M \n");

  printf("              \"-f j\": generate a bunch of bootstrapped alignment files from an original alignemnt file.\n");
  printf("                      You need to specify a seed with \"-b\" and the number of replicates with \"-#\" \n"); 

  printf("              \"-f m\": compare bipartitions between two bunches of trees passed via \"-t\" and \"-z\" \n");
  printf("                      respectively. This will return the Pearson correlation between all bipartitions found\n");
  printf("                      in the two tree files. A file called RAxML_bipartitionFrequencies.outpuFileName\n");
  printf("                      will be printed that contains the pair-wise bipartition frequencies of the two sets\n");

  printf("              \"-f n\": compute the log likelihood score of all trees contained in a tree file provided by\n");
  printf("                      \"-z\" under GAMMA or GAMMA+P-Invar\n");

  printf("              \"-f o\": old and slower rapid hill-climbing without heuristic cutoff\n");

  printf("              \"-f p\": perform pure stepwise MP addition of new sequences to an incomplete starting tree and exit\n");

  printf("              \"-f r\": compute pairwise Robinson-Foulds (RF) distances between all pairs of trees in a tree file passed via \"-z\" \n");
  printf("                      if the trees have node labales represented as integer support values the program will also compute two flavors of\n");
  printf("                      the weighted Robinson-Foulds (WRF) distance\n");

  printf("              \"-f s\": split up a multi-gene partitioned alignment into the respective subalignments \n");

  printf("              \"-f t\": do randomized tree searches on one fixed starting tree\n");

  printf("              \"-f u\": execute morphological weight calibration using maximum likelihood, this will return a weight vector.\n");
  printf("                      you need to provide a morphological alignment and a reference tree via \"-t\" \n");
  
  printf("              \"-f U\": execute morphological wieght calibration using parsimony, this will return a weight vector.\n");
  printf("                      you need to provide a morphological alignment and a reference tree via \"-t\" \n");

  printf("              \"-f v\": classify a bunch of environmental sequences into a reference tree using the slow heuristics without dynamic alignment\n");
  printf("                      you will need to start RAxML with a non-comprehensive reference tree and an alignment containing all sequences (reference + query)\n");

  printf("              \"-f w\": compute ELW test on a bunch of trees passed via \"-z\" \n");

  printf("              \"-f x\": compute pair-wise ML distances, ML model parameters will be estimated on an MP \n");
  printf("                      starting tree or a user-defined tree passed via \"-t\", only allowed for GAMMA-based\n");
  printf("                      models of rate heterogeneity\n");

  printf("              \"-f y\": classify a bunch of environmental sequences into a reference tree using the fast heuristics without dynamic alignment\n");
  printf("                      you will need to start RAxML with a non-comprehensive reference tree and an alignment containing all sequences (reference + query)\n");
  
  printf("\n");
  printf("              DEFAULT for \"-f\": new rapid hill climbing\n");

  printf("\n");
}


static void printREADME(void)
{
  printVersionInfo();
  printf("\n");
  printf("Please also consult the RAxML-manual\n");
  printf("\nTo report bugs send an email to stamatak@cs.tum.edu\n");
  printf("Please send me all input files, the exact invocation, details of the HW and operating system,\n");
  printf("as well as all error messages printed to screen.\n\n\n");

  printf("raxmlHPC[-SSE3|-PTHREADS|-PTHREADS-SSE3|-HYBRID|-HYBRID-SSE3]\n");
  printf("      -s sequenceFileName -n outputFileName -m substitutionModel\n");
  printf("      [-a weightFileName] [-A secondaryStructureSubstModel]\n");
  printf("      [-b bootstrapRandomNumberSeed] [-B wcCriterionThreshold]\n");
  printf("      [-c numberOfCategories] [-C] [-d] [-D]\n");
  printf("      [-e likelihoodEpsilon] [-E excludeFileName]\n");
  printf("      [-f a|b|c|d|e|E|F|g|h|i|I|j|m|n|o|p|r|s|t|u|U|v|w|x|y] [-F]\n");
  printf("      [-g groupingFileName] [-G placementThreshold] [-h] [-H placementThreshold]\n");
  printf("      [-i initialRearrangementSetting] [-I autoFC|autoMR|autoMRE|autoMRE_IGN]\n");
  printf("      [-j] [-J MR|MRE|STRICT] [-k] [-K] [-M]\n");
  printf("      [-o outGroupName1[,outGroupName2[,...]]] [-O checkPointInterval]\n");
  printf("      [-p parsimonyRandomSeed] [-P proteinModel]\n");
  printf("      [-q multipleModelFileName] [-Q] [-r binaryConstraintTree]\n");
  printf("      [-S secondaryStructureFile] [-t userStartingTree]\n");
  printf("      [-T numberOfThreads] [-v] [-w workingDirectory]\n");
  printf("      [-x rapidBootstrapRandomNumberSeed] [-y] [-Y]\n");
  printf("      [-z multipleTreesFile] [-#|-N numberOfRuns|autoFC|autoMR|autoMRE|autoMRE_IGN]\n");
  printf("\n");
  printf("      -a      Specify a column weight file name to assign individual weights to each column of \n");
  printf("              the alignment. Those weights must be integers separated by any type and number \n");
  printf("              of whitespaces whithin a separate file, see file \"example_weights\" for an example.\n");
  printf("\n");
  printf("      -A      Specify one of the secondary structure substitution models implemented in RAxML.\n");
  printf("              The same nomenclature as in the PHASE manual is used, available models: \n");
  printf("              S6A, S6B, S6C, S6D, S6E, S7A, S7B, S7C, S7D, S7E, S7F, S16, S16A, S16B\n");
  printf("\n");
  printf("              DEFAULT: 16-state GTR model (S16)\n");
  printf("\n");
  printf("      -b      Specify an integer number (random seed) and turn on bootstrapping\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -B      specify a floating point number between 0.0 and 1.0 that will be used as cutoff threshold \n");
  printf("              for the MR-based bootstopping criteria. The recommended setting is 0.03.\n");
  printf("\n");
  printf("              DEFAULT: 0.03 (recommended empirically determined setting)\n");
  printf("\n");
  printf("      -c      Specify number of distinct rate catgories for RAxML when modelOfEvolution\n");
  printf("              is set to GTRCAT or GTRMIX\n");
  printf("              Individual per-site rates are categorized into numberOfCategories rate \n");
  printf("              categories to accelerate computations. \n");
  printf("\n");
  printf("              DEFAULT: 25\n");
  printf("\n");
  printf("      -C      Conduct model parameter optimization on gappy, partitioned multi-gene alignments with per-partition\n");
  printf("              branch length estimates (-M enabled) using the fast method with pointer meshes described in:\n");
  printf("              Stamatakis and Ott: \"Efficient computation of the phylogenetic likelihood function on multi-gene alignments and multi-core processors\"\n");
  printf("              WARNING: We can not conduct useful tree searches using this method yet! Does not work with Pthreads version.\n");
  printf("\n");
  printf("      -d      start ML optimization from random starting tree \n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -D      ML search convergence criterion. This will break off ML searches if the relative \n");
  printf("              Robinson-Foulds distance between the trees obtained from two consecutive lazy SPR cycles\n");
  printf("              is smaller or equal to 1%s. Usage recommended for very large datasets in terms of taxa.\n", "%");
  printf("              On trees with more than 500 taxa this will yield execution time improvements of approximately 50%s\n",  "%");
  printf("              While yielding only slightly worse trees.\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -e      set model optimization precision in log likelihood units for final\n");
  printf("              optimization of tree topology under MIX/MIXI or GAMMA/GAMMAI\n");
  printf("\n");
  printf("              DEFAULT: 0.1   for models not using proportion of invariant sites estimate\n");
  printf("                       0.001 for models using proportion of invariant sites estimate\n");
  printf("\n");
  printf("      -E      specify an exclude file name, that contains a specification of alignment positions you wish to exclude.\n");
  printf("              Format is similar to Nexus, the file shall contain entries like \"100-200 300-400\", to exclude a\n");
  printf("              single column write, e.g., \"100-100\", if you use a mixed model, an appropriatly adapted model file\n");
  printf("              will be written.\n");
  printf("\n");
  printf("      -f      select algorithm:\n");

  printMinusFUsage();

  printf("\n");
  printf("      -F      enable ML tree searches under CAT model for very large trees without switching to \n");
  printf("              GAMMA in the end (saves memory).\n");
  printf("              This option can also be used with the GAMMA models in order to avoid the thorough optimization \n");
  printf("              of the best-scoring ML tree in the end.\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -g      specify the file name of a multifurcating constraint tree\n");
  printf("              this tree does not need to be comprehensive, i.e. must not contain all taxa\n");
  printf("\n");
  printf("      -G      enable the ML-based evolutionary placement algorithm heuristics\n");
  printf("              by specifiyng a threshold value (fraction of insertion branches to be evaluated\n");
  printf("              using slow insertions under ML).\n");
  printf("\n");
  printf("      -h      Display this help message.\n");
  printf("\n");
  printf("      -H      enable the MP-based evolutionary placement algorithm heuristics\n");
  printf("              by specifiyng a threshold value (fraction of insertion branches to be evaluated\n");
  printf("              using slow insertions under ML).\n");
  printf("\n");
  printf("      -i      Initial rearrangement setting for the subsequent application of topological \n");
  printf("              changes phase\n");
  printf("\n");
  printf("      -I      a posteriori bootstopping analysis. Use:\n");
  printf("             \"-I autoFC\" for the frequency-based criterion\n");
  printf("             \"-I autoMR\" for the majority-rule consensus tree criterion\n");
  printf("             \"-I autoMRE\" for the extended majority-rule consensus tree criterion\n");
  printf("             \"-I autoMRE_IGN\" for metrics similar to MRE, but include bipartitions under the threshold whether they are compatible\n");
  printf("                              or not. This emulates MRE but is faster to compute.\n");
  printf("              You also need to pass a tree file containg several bootstrap replicates via \"-z\" \n"); 
  printf("\n");
  printf("      -j      Specifies that intermediate tree files shall be written to file during the standard ML and BS tree searches.\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -J      Compute majority rule consensus tree with \"-J MR\" or extended majority rule consensus tree with \"-J MRE\"\n");
  printf("              or strict consensus tree with \"-J STRICT\"\n");
  printf("              You will need to provide a tree file containing several UNROOTED trees via \"-z\"\n");
  printf("\n");
  printf("      -k      Specifies that bootstrapped trees should be printed with branch lengths.\n");
  printf("              The bootstraps will run a bit longer, because model parameters will be optimized\n");
  printf("              at the end of each run under GAMMA or GAMMA+P-Invar respectively.\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");  
  printf("      -K      Specify one of the multi-state substitution models (max 32 states) implemented in RAxML.\n");
  printf("              Available models are: ORDERED, MK, GTR\n");
  printf("\n");
  printf("              DEFAULT: GTR model \n");
  printf("\n");
  printf("      -m      Model of Binary (Morphological), Nucleotide, Multi-State, or Amino Acid Substitution: \n");
  printf("\n");
  printf("              BINARY:\n\n");
  printf("                \"-m BINCAT\"         : Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency. Final tree might be evaluated\n");
  printf("                                      automatically under BINGAMMA, depending on the tree search option\n");
  printf("                \"-m BINCATI\"        : Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency. Final tree might be evaluated\n");
  printf("                                      automatically under BINGAMMAI, depending on the tree search option \n");
  printf("                \"-m BINGAMMA\"       : GAMMA model of rate \n");
  printf("                                      heterogeneity (alpha parameter will be estimated)\n");
  printf("                \"-m BINGAMMAI\"      : Same as BINGAMMA, but with estimate of proportion of invariable sites\n");
  printf("\n");
  printf("              NUCLEOTIDES:\n\n");
  printf("                \"-m GTRCAT\"         : GTR + Optimization of substitution rates + Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency.  Final tree might be evaluated\n");
  printf("                                      under GTRGAMMA, depending on the tree search option\n");
  printf("                \"-m GTRCAT_FLOAT\"   : Same as above but uses single-precision floating point arithemtics instead of double-precision\n");
  printf("                                      Usage only recommened for testing, the code will run slower, but can save almost 50%s of memory.\n", "%");
  printf("                                      If you have problems with phylogenomic datasets and large memory requirements you may give it a shot.\n");
  printf("                                      Keep in mind that numerical stability seems to be okay but needs further testing.\n");  
  printf("                \"-m GTRCATI\"        : GTR + Optimization of substitution rates + Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency.  Final tree might be evaluated\n");
  printf("                                      under GTRGAMMAI, depending on the tree search option\n");
  printf("                \"-m GTRGAMMA\"       : GTR + Optimization of substitution rates + GAMMA model of rate \n");
  printf("                                      heterogeneity (alpha parameter will be estimated)\n");
  printf("                \"-m GTRGAMMA_FLOAT\" : Same as GTRGAMMA, but also with single-precision arithmetics, same cautionary notes as for  \n");
  printf("                                      GTRCAT_FLOAT apply.\n");
  printf("                \"-m GTRGAMMAI\"      : Same as GTRGAMMA, but with estimate of proportion of invariable sites \n");
  printf("\n");
  printf("              MULTI-STATE:\n\n");
  printf("                \"-m MULTICAT\"         : Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency. Final tree might be evaluated\n");
  printf("                                      automatically under MULTIGAMMA, depending on the tree search option\n");
  printf("                \"-m MULTICATI\"        : Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency. Final tree might be evaluated\n");
  printf("                                      automatically under MULTIGAMMAI, depending on the tree search option \n");
  printf("                \"-m MULTIGAMMA\"       : GAMMA model of rate \n");
  printf("                                      heterogeneity (alpha parameter will be estimated)\n");
  printf("                \"-m MULTIGAMMAI\"      : Same as MULTIGAMMA, but with estimate of proportion of invariable sites\n");
  printf("\n");
  printf("                You can use up to 32 distinct character states to encode multi-state regions, they must be used in the following order:\n");
  printf("                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V\n");
  printf("                i.e., if you have 6 distinct character states you would use 0, 1, 2, 3, 4, 5 to encode these.\n");
  printf("                The substitution model for the multi-state regions can be selected via the \"-K\" option\n");
  printf("\n");
  printf("              AMINO ACIDS:\n\n");
  printf("                \"-m PROTCATmatrixName[F]\"         : specified AA matrix + Optimization of substitution rates + Optimization of site-specific\n");
  printf("                                                    evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                                    rate categories for greater computational efficiency.   Final tree might be evaluated\n");
  printf("                                                    automatically under PROTGAMMAmatrixName[f], depending on the tree search option\n");
  printf("                \"-m PROTCATmatrixName[F]_FLOAT\"   : PROTCAT with single precision arithmetics, same cautionary notes as for GTRCAT_FLOAT apply\n");
  printf("                \"-m PROTCATImatrixName[F]\"        : specified AA matrix + Optimization of substitution rates + Optimization of site-specific\n");
  printf("                                                    evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                                    rate categories for greater computational efficiency.   Final tree might be evaluated\n");
  printf("                                                    automatically under PROTGAMMAImatrixName[f], depending on the tree search option\n");
  printf("                \"-m PROTGAMMAmatrixName[F]\"       : specified AA matrix + Optimization of substitution rates + GAMMA model of rate \n");
  printf("                                                    heterogeneity (alpha parameter will be estimated)\n");
  printf("                \"-m PROTGAMMAmatrixName[F]_FLOAT\" : PROTGAMMA with single precision arithmetics, same cautionary notes as for GTRCAT_FLOAT apply\n");
  printf("                \"-m PROTGAMMAImatrixName[F]\"      : Same as PROTGAMMAmatrixName[F], but with estimate of proportion of invariable sites \n");
  printf("\n");
  printf("                Available AA substitution models: DAYHOFF, DCMUT, JTT, MTREV, WAG, RTREV, CPREV, VT, BLOSUM62, MTMAM, LG, GTR\n");
  printf("                With the optional \"F\" appendix you can specify if you want to use empirical base frequencies\n");
  printf("                Please note that for mixed models you can in addition specify the per-gene AA model in\n");
  printf("                the mixed model file (see manual for details). Also note that if you estimate AA GTR parameters on a partitioned\n");
  printf("                dataset, they will be linked (estimated jointly) across all partitions to avoid over-parametrization\n");
  printf("\n");
  printf("      -M      Switch on estimation of individual per-partition branch lengths. Only has effect when used in combination with \"-q\"\n");
  printf("              Branch lengths for individual partitions will be printed to separate files\n");
  printf("              A weighted average of the branch lengths is computed by using the respective partition lengths\n");
  printf("\n"),
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -n      Specifies the name of the output file.\n");
  printf("\n");
  printf("      -o      Specify the name of a single outgrpoup or a comma-separated list of outgroups, eg \"-o Rat\" \n");
  printf("              or \"-o Rat,Mouse\", in case that multiple outgroups are not monophyletic the first name \n");
  printf("              in the list will be selected as outgroup, don't leave spaces between taxon names!\n");
  printf("\n");  
  printf("      -O      Enable checkpointing using the dmtcp library available at http://dmtcp.sourceforge.net/\n");
  printf("              This only works if you call the program by preceded by the command \"dmtcp_checkpoint\"\n");
  printf("              and if you compile a dedicated binary using the appropriate Makefile.\n");
  printf("              With \"-O\" you can specify the interval between checkpoints in seconds.\n");
  printf("\n");
  printf("              DEFAULT: 3600.0 seconds\n");
  printf("\n");
  printf("      -p      Specify a random number seed for the parsimony inferences. This allows you to reproduce your results\n");
  printf("              and will help me debug the program.\n");
  printf("\n");
  printf("      -P      Specify the file name of a user-defined AA (Protein) substitution model. This file must contain\n");
  printf("              420 entries, the first 400 being the AA substitution rates (this must be a symmetric matrix) and the\n");
  printf("              last 20 are the empirical base frequencies\n");
  printf("\n");
  printf("      -q      Specify the file name which contains the assignment of models to alignment\n");
  printf("              partitions for multiple models of substitution. For the syntax of this file\n");
  printf("              please consult the manual.\n");
  printf("\n");
  printf("      -Q      Turn on computation of SH-like support values on tree.\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -r      Specify the file name of a binary constraint tree.\n");
  printf("              this tree does not need to be comprehensive, i.e. must not contain all taxa\n");
  printf("\n");
  printf("      -s      Specify the name of the alignment data file in PHYLIP format\n");
  printf("\n");
  printf("      -S      Specify the name of a secondary structure file. The file can contain \".\" for \n");
  printf("              alignment columns that do not form part of a stem and characters \"()<>[]{}\" to define \n");
  printf("              stem regions and pseudoknots\n");
  printf("\n");
  printf("      -t      Specify a user starting tree file name in Newick format\n");
  printf("\n");
  printf("      -T      PTHREADS VERSION ONLY! Specify the number of threads you want to run.\n");
  printf("              Make sure to set \"-T\" to at most the number of CPUs you have on your machine,\n");
  printf("              otherwise, there will be a huge performance decrease!\n");
  printf("\n");
  printf("      -v      Display version information\n");
  printf("\n");
  printf("      -w      Name of the working directory where RAxML will write its output files\n");
  printf("\n");
  printf("              DEFAULT: current directory\n");
  printf("\n");
  printf("      -x      Specify an integer number (random seed) and turn on rapid bootstrapping\n");
  printf("              CAUTION: unlike in version 7.0.4 RAxML will conduct rapid BS replicates under \n");
  printf("              the model of rate heterogeneity you specified via \"-m\" and not by default under CAT\n");
  printf("\n");
  printf("      -y      If you want to only compute a parsimony starting tree with RAxML specify \"-y\",\n");
  printf("              the program will exit after computation of the starting tree\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -Y      Do a more thorough parsimony tree search using a parsimony ratchet and exit. \n");
  printf("              specify the number of ratchet searches via \"-#\" or \"-N\"\n");
  printf("              This has just been implemented for completeness, if you want a fast MP implementation use TNT\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -z      Specify the file name of a file containing multiple trees e.g. from a bootstrap\n");
  printf("              that shall be used to draw bipartition values onto a tree provided with \"-t\",\n");
  printf("              It can also be used to compute per site log likelihoods in combination with \"-f g\"\n");
  printf("              and to read a bunch of trees for a couple of other options (\"-f h\", \"-f m\", \"-f n\").\n");
  printf("\n");
  printf("      -#|-N   Specify the number of alternative runs on distinct starting trees\n");
  printf("              In combination with the \"-b\" option, this will invoke a multiple boostrap analysis\n");
  printf("              Note that \"-N\" has been added as an alternative since \"-#\" sometimes caused problems\n");
  printf("              with certain MPI job submission systems, since \"-#\" is often used to start comments.\n");
  printf("              If you want to use the bootstopping criteria specify \"-# autoMR\" or \"-# autoMRE\" or \"-# autoMRE_IGN\"\n");
  printf("              for the majority-rule tree based criteria (see -I option) or \"-# autoFC\" for the frequency-based criterion.\n");
  printf("              Bootstopping will only work in combination with \"-x\" or \"-b\"\n");
  printf("\n");
  printf("              DEFAULT: 1 single analysis\n");
  printf("\n\n\n\n");

}








static void get_args(int argc, char *argv[], analdef *adef, tree *tr)
{
  boolean
    bad_opt    =FALSE,
    workDirSet = FALSE;

  char       
    aut[256],       
    buf[2048],
    *optarg,
    model[2048] = "",
    secondaryModel[2048] = "",
    multiStateModel[2048] = "",
    modelChar;

  double 
    likelihoodEpsilon,    
    wcThreshold,
    fastEPAthreshold;
  
  int  
    optind = 1,        
    c,
    nameSet = 0,
    alignmentSet = 0,
    multipleRuns = 0,
    constraintSet = 0,
    treeSet = 0,
    groupSet = 0,
    modelSet = 0,
    treesSet  = 0;

  boolean
    bSeedSet = FALSE,
    xSeedSet = FALSE,
    multipleRunsSet = FALSE;

  run_id[0] = 0;
  workdir[0] = 0;
  seq_file[0] = 0;
  tree_file[0] = 0;
  model[0] = 0;
  weightFileName[0] = 0;
  modelFileName[0] = 0;

  /*********** tr inits **************/

#ifdef _USE_PTHREADS
  NumberOfThreads = 0;
#endif
  

  tr->useFastScaling = TRUE;

  tr->useFloat = FALSE;
  tr->bootStopCriterion = -1;
  tr->wcThreshold = 0.03;
  tr->doCutoff = TRUE;
  tr->secondaryStructureModel = SEC_16; /* default setting */
  tr->searchConvergenceCriterion = FALSE;
  tr->catOnly = FALSE;
  tr->multiGene = 0;
  tr->fastEPA_ML = FALSE;
  tr->fastEPA_MP = FALSE;
  tr->fastEPAthreshold = -1.0;
  tr->multiStateModel  = GTR_MULTI_STATE;
  
  /********* tr inits end*************/


  while(!bad_opt &&
	((c = mygetopt(argc,argv,"O:T:E:N:B:L:P:S:A:G:H:I:J:K:l:x:z:g:r:e:a:b:c:f:i:m:t:w:s:n:o:q:#:p:vdyjhkMYDFCQ", &optind, &optarg))!=-1))
    {
    switch(c)
      {
      case 'Q':
	adef->shSupports = TRUE;
	break;
      case 'O':
	#ifndef _IPTOL
	printf("Using the \"-O\" option to enable checkpointing does not have an effect with this binary, please use the appropriate\n");
	printf("Makefile to produce the respective binary. You will also need to install the dmtcp library http://dmtcp.sourceforge.net/\n");	
	printf("If you are a user that is afraid of the command line, don't try this at home.\n");
	#else
	sscanf(optarg,"%lf", &checkPointInterval);
	if(checkPointInterval < 1800.0)
	  printf("\n\nYou have set the checkpointing interval to %f seconds, are you sure that you want to checkpoint that frequntly?\n", checkPointInterval);
	if(checkPointInterval > 86400.0)
	  printf("\n\nYou have set the checkpointing interval to %f seconds/%f hours/%fdays are you sure that you want to checkpoint that infrequntly?\n",
		 checkPointInterval, checkPointInterval / 3600.0, checkPointInterval / 86400.0);	  	
	#endif
	break;
      case 'K':
	{
	  const char *modelList[3] = { "ORDERED", "MK", "GTR"};
	  const int states[3] = {ORDERED_MULTI_STATE, MK_MULTI_STATE, GTR_MULTI_STATE};
	  int i;

	  sscanf(optarg, "%s", multiStateModel);

	  for(i = 0; i < 3; i++)
	    if(strcmp(multiStateModel, modelList[i]) == 0)
	      break;

	  if(i < 3)
	    tr->multiStateModel = states[i];
	  else
	    {
	      printf("The multi-state model %s you want to use does not exist, exiting .... \n", multiStateModel);
	      errorExit(0);
	    }
	  

	}
	break;
      case 'A':
	{
	  const char *modelList[21] = { "S6A", "S6B", "S6C", "S6D", "S6E", "S7A", "S7B", "S7C", "S7D", "S7E", "S7F", "S16", "S16A", "S16B", "S16C",
				      "S16D", "S16E", "S16F", "S16I", "S16J", "S16K"};
	  int i;

	  sscanf(optarg, "%s", secondaryModel);

	  for(i = 0; i < 21; i++)
	    if(strcmp(secondaryModel, modelList[i]) == 0)
	      break;

	  if(i < 21)
	    tr->secondaryStructureModel = i;
	  else
	    {
	      printf("The secondary structure model %s you want to use does not exist, exiting .... \n", secondaryModel);
	      errorExit(0);
	    }
	}
	break;
      case 'B':
	sscanf(optarg,"%lf", &wcThreshold);
	tr->wcThreshold = wcThreshold;
	if(wcThreshold <= 0.0 || wcThreshold >= 1.0)
	  {
	    printf("\nBootstrap threshold must be set to values between 0.0 and 1.0, you just set it to %f\n", wcThreshold);
	    exit(-1);
	  }
	if(wcThreshold < 0.01 || wcThreshold > 0.05)
	  {
	    printf("\n\nWARNING, reasonable settings for Bootstopping threshold with MR-based criteria range between 0.01 and 0.05.\n");
	    printf("You are just setting it to %f, the most reasonable empirically determined setting is 0.03 \n\n", wcThreshold);
	  }
	break;
      case 'C':
	tr->multiGene = 1;
	break;
      case 'D':
	tr->searchConvergenceCriterion = TRUE;
	break;
      case 'E':
	strcpy(excludeFileName, optarg);
	adef->useExcludeFile = TRUE;
	break;
      case 'F':
	tr->catOnly = TRUE;
	break;
      case 'G':
	tr->fastEPA_ML = TRUE;
	sscanf(optarg,"%lf", &fastEPAthreshold);
	tr->fastEPAthreshold = fastEPAthreshold;
	
	if(fastEPAthreshold <= 0.0 || fastEPAthreshold >= 1.0)
	  {
	    printf("\nHeuristic EPA threshold must be set to values between 0.0 and 1.0, you just set it to %f\n", fastEPAthreshold);
	    exit(-1);
	  }
	if(fastEPAthreshold < 0.015625 || fastEPAthreshold > 0.5)
	  {
	    printf("\n\nWARNING, reasonable settings for heuristic EPA threshold range between 0.015625 (1/64) and 0.5 (1/2).\n");
	    printf("You are just setting it to %f\n\n", fastEPAthreshold);
	  }	
#ifdef _USE_PTHREADS
	tr->useFastScaling = FALSE;
#endif	
	break;	
      case 'H':
	tr->fastEPA_MP = TRUE;
	sscanf(optarg,"%lf", &fastEPAthreshold);
	tr->fastEPAthreshold = fastEPAthreshold;	
	if(fastEPAthreshold <= 0.0 || fastEPAthreshold >= 1.0)
	  {
	    printf("\nHeuristic EPA threshold must be set to values between 0.0 and 1.0, you just set it to %f\n", fastEPAthreshold);
	    exit(-1);
	  }
	if(fastEPAthreshold < 0.015625 || fastEPAthreshold > 0.5)
	  {
	    printf("\n\nWARNING, reasonable settings for heuristic EPA threshold range between 0.015625 (1/64) and 0.5 (1/2).\n");
	    printf("You are just setting it to %f\n\n", fastEPAthreshold);
	  }
#ifdef _USE_PTHREADS
	tr->useFastScaling = FALSE;
#endif		
	break;	
      case 'I':     
	adef->readTaxaOnly = TRUE;
	adef->mode = BOOTSTOP_ONLY;
	if((sscanf(optarg,"%s", aut) > 0) && ((strcmp(aut, "autoFC") == 0) || (strcmp(aut, "autoMR") == 0) || 
					      (strcmp(aut, "autoMRE") == 0) || (strcmp(aut, "autoMRE_IGN") == 0)))
	  {
	    if((strcmp(aut, "autoFC") == 0))	   
	      tr->bootStopCriterion = FREQUENCY_STOP;
	    if((strcmp(aut, "autoMR") == 0))		  	    
	      tr->bootStopCriterion = MR_STOP;	   
	    if((strcmp(aut, "autoMRE") == 0))	   
	      tr->bootStopCriterion = MRE_STOP;
	    if((strcmp(aut, "autoMRE_IGN") == 0))
	      tr->bootStopCriterion = MRE_IGN_STOP;
	  }
	else
	  {
	    if(processID == 0)	      
	      printf("Use -I a posteriori bootstop option either as \"-I autoFC\" or \"-I autoMR\" or \"-I autoMRE\" or \"-I autoMRE_IGN\"\n");	       	      
	    errorExit(0);
	  }
	break;     
      case 'J':	
	adef->readTaxaOnly = TRUE;
	adef->mode = CONSENSUS_ONLY;
	
	if((sscanf(optarg,"%s", aut) > 0) && ((strcmp(aut, "MR") == 0) || (strcmp(aut, "MRE") == 0) || (strcmp(aut, "STRICT") == 0)))
	  {
	    if((strcmp(aut, "MR") == 0))	   
	      tr->consensusType = MR_CONSENSUS;
	    if((strcmp(aut, "MRE") == 0))		  	    
	      tr->consensusType = MRE_CONSENSUS;	   	    
	    if((strcmp(aut, "STRICT") == 0))		  	    
	      tr->consensusType = STRICT_CONSENSUS;	
	  }
	else
	  {
	    if(processID == 0)	      
	      printf("Use -J consensus tree option either as \"-J MR\" or \"-J MRE\" or \"-J STRICT\"\n");	       	      
	    errorExit(0);
	  }	
	      
	break;
      case 'M':
	adef->perGeneBranchLengths = TRUE;
	break;
      case 'P':
	strcpy(proteinModelFileName, optarg);
	adef->userProteinModel = TRUE;
	parseProteinModel(adef);
	break;
      case 'S':
	adef->useSecondaryStructure = TRUE;
	strcpy(secondaryStructureFileName, optarg);
	break;
      case 'T':
#ifdef _USE_PTHREADS
	sscanf(optarg,"%d", &NumberOfThreads);
#else
	if(processID == 0)
	  {
	    printf("Option -T does not have any effect with the sequential or parallel MPI version.\n");
	    printf("It is used to specify the number of threads for the Pthreads-based parallelization\n");
	  }
#endif
	break;                  
      case 'o':
	{
	  char *outgroups;
	  outgroups = (char*)malloc(sizeof(char) * (strlen(optarg) + 1));
	  strcpy(outgroups, optarg);
	  parseOutgroups(outgroups, tr);
	  free(outgroups);
	  adef->outgroup = TRUE;
	}
	break;
      case 'k':
	adef->bootstrapBranchLengths = TRUE;
	break;
      case 'z':
	strcpy(bootStrapFile, optarg);
	treesSet = 1;
	break;
      case 'd':
	adef->randomStartingTree = TRUE;
	break;
      case 'g':
	strcpy(tree_file, optarg);
	adef->grouping = TRUE;
	adef->restart  = TRUE;
	groupSet = 1;
	break;
      case 'r':
	strcpy(tree_file, optarg);
	adef->restart = TRUE;
	adef->constraint = TRUE;
	constraintSet = 1;
	break;
      case 'e':
	sscanf(optarg,"%lf", &likelihoodEpsilon);
	adef->likelihoodEpsilon = likelihoodEpsilon;
	break;
      case 'q':
	strcpy(modelFileName,optarg);
	adef->useMultipleModel = TRUE;
        break;
      case 'p':
	sscanf(optarg,"%ld", &(adef->parsimonySeed));	
	if(adef->parsimonySeed <= 0)
	  {
	    printf("Parsimony seed specified via -p must be greater than zero\n");
	    errorExit(-1);
	  }
	break;
      case 'N':
      case '#':
	if(sscanf(optarg,"%d", &multipleRuns) > 0)
	  {
	    adef->multipleRuns = multipleRuns;
	  }
	else
	  {
	    if((sscanf(optarg,"%s", aut) > 0) && ((strcmp(aut, "autoFC") == 0) || (strcmp(aut, "autoMR") == 0) || 
						  (strcmp(aut, "autoMRE") == 0) || (strcmp(aut, "autoMRE_IGN") == 0)))
						  
	      {
		adef->bootStopping = TRUE;
		adef->multipleRuns = 1000;

		if((strcmp(aut, "autoFC") == 0))	   
		  tr->bootStopCriterion = FREQUENCY_STOP;
		if((strcmp(aut, "autoMR") == 0))		  	    
		  tr->bootStopCriterion = MR_STOP;	   
		if((strcmp(aut, "autoMRE") == 0))	   
		  tr->bootStopCriterion = MRE_STOP;
		if((strcmp(aut, "autoMRE_IGN") == 0))
		  tr->bootStopCriterion = MRE_IGN_STOP;
	      }
	    else
	      {
		if(processID == 0)
		  {
		    printf("Use -# or -N option either with an integer, e.g., -# 100 or with -# autoFC or -# autoMR or -# autoMRE or -# autoMRE_IGN\n");
		    printf("or -N 100 or  -N autoFC or -N autoMR or -N autoMRE or -N autoMRE_IGN respectively, note that auto will not work for the\n");
		    printf("MPI-based parallel version\n");
		  }
		errorExit(0);
	      }
	  }
	multipleRunsSet = TRUE;
	break;
      case 'v':
	printVersionInfo();
	errorExit(0);
      case 'y':
	adef->startingTreeOnly = 1;
	break;
      case 'Y':
	adef->mode = THOROUGH_PARSIMONY;
	break;
      case 'h':
	printREADME();
	errorExit(0);
      case 'j':
	adef->checkpoints = 1;
	break;
      case 'a':
	strcpy(weightFileName,optarg);
	adef->useWeightFile = TRUE;
        break;
      case 'b':
	sscanf(optarg,"%ld", &adef->boot);
	if(adef->boot <= 0)
	  {
	    printf("Bootstrap seed specified via -b must be greater than zero\n");
	    errorExit(-1);
	  }
	bSeedSet = TRUE;
	break;
      case 'x':
	sscanf(optarg,"%ld", &adef->rapidBoot);
	if(adef->rapidBoot <= 0)
	  {
	    printf("Bootstrap seed specified via -x must be greater than zero\n");
	    errorExit(-1);
	  }
	xSeedSet = TRUE;
	break;
      case 'c':
	sscanf(optarg, "%d", &adef->categories);
	break;     
      case 'f':
	sscanf(optarg, "%c", &modelChar);
	switch(modelChar)
	  {
	  case 'a':
	    adef->allInOne = TRUE;
	    adef->mode = BIG_RAPID_MODE;
	    tr->doCutoff = TRUE;
	    break;
	  case 'b':
	    adef->readTaxaOnly = TRUE;
	    adef->mode = CALC_BIPARTITIONS;
	    break;
	  case 'c':
	    adef->mode = CHECK_ALIGNMENT;
	    break;
	  case 'd':
	    adef->mode = BIG_RAPID_MODE;
	    tr->doCutoff = TRUE;
	    break;
	  case 'e':
	    adef->mode = TREE_EVALUATION;
	    break;
	  case 'F':
	    adef->mode = FAST_SEARCH;
	    adef->veryFast = FALSE;
	    break;
	  case 'E':
	    adef->mode = FAST_SEARCH;
	    adef->veryFast = TRUE;
	    break;
	  case 'g':
	    tr->useFastScaling = FALSE;
	    adef->mode = PER_SITE_LL;
	    break;
	  case 'h':
	    adef->mode = TREE_EVALUATION;
	    adef->likelihoodTest = TRUE;
	    tr->useFastScaling = FALSE;
	    break;
	  case 'i':
	    adef->mode = MESH_TREE_SEARCH;
	    adef->meshSearch = 0;
	    break;
	  case 'I':
	    adef->mode = MESH_TREE_SEARCH;
	    adef->meshSearch = 1;
	    break;
	  case 'j':
	    adef->mode = GENERATE_BS;
	    adef->generateBS = TRUE;
	    break;	 
	  case 'm': 
	    adef->readTaxaOnly = TRUE;	    
	    adef->mode = COMPUTE_BIPARTITION_CORRELATION;
	    break;
	  case 'n':
	    adef->mode = COMPUTE_LHS;
	    break;
	  case 'o':
	    adef->mode = BIG_RAPID_MODE;
	    tr->doCutoff = FALSE;
	    break;
	  case 'p':
	    adef->mode =  PARSIMONY_ADDITION;
	    break;	 
	  case 'r':
	    adef->readTaxaOnly = TRUE;
	    adef->mode = COMPUTE_RF_DISTANCE;
	    break;
	  case 's':
	    adef->mode = SPLIT_MULTI_GENE;
	    break;
	  case 't':
	    adef->mode = BIG_RAPID_MODE;
	    tr->doCutoff = TRUE;
	    adef->permuteTreeoptimize = TRUE;
	    break;
	  case 'u':
	    adef->mode = MORPH_CALIBRATOR;
	    tr->useFastScaling = FALSE;
	    adef->compressPatterns  = FALSE;	    
	    break;
	  case 'U':
	    adef->mode = MORPH_CALIBRATOR_PARSIMONY;
	    tr->useFastScaling = FALSE;
	    adef->compressPatterns  = FALSE;	    
	    break;
	  case 'v':	    
	    adef->mode = CLASSIFY_ML;	   
	    adef->thoroughInsertion = TRUE;	   
#ifdef _USE_PTHREADS
	    tr->useFastScaling = FALSE;
#endif
	    break;
	  case 'w':
	    workDirSet = TRUE;
	    adef->mode = COMPUTE_ELW;
	    adef->computeELW = TRUE;
	    break;
	  case 'x':
	    adef->mode = DISTANCE_MODE;
	    adef->computeDistance = TRUE;
	    break;	  	  
	  case 'y':
	    adef->mode = CLASSIFY_ML;	    
	    adef->thoroughInsertion = FALSE;	   
#ifdef _USE_PTHREADS
	    tr->useFastScaling = FALSE;
#endif
	    break;	  	  	  	     
	  default:
	    {
	      if(processID == 0)
		{
		  printf("Error select one of the following algorithms via -f :\n");
		  printMinusFUsage();
		}
	      errorExit(-1);
	    }
	  }
	break;
      case 'i':
	sscanf(optarg, "%d", &adef->initial);
	adef->initialSet = TRUE;
	break;
      case 'n':
        strcpy(run_id,optarg);
	nameSet = 1;
        break;
      case 'w':
        strcpy(workdir,optarg);
        break;
      case 't':
	strcpy(tree_file, optarg);
	adef->restart = TRUE;
	treeSet = 1;
	break;
      case 's':
	strcpy(seq_file, optarg);
	alignmentSet = 1;
	break;
      case 'm':
	strcpy(model,optarg);
	if(modelExists(model, adef) == 0)
	  {
	    if(processID == 0)
	      {
		printf("Model %s does not exist\n\n", model);
                printf("For BINARY data use: BINCAT                or BINGAMMA                or\n");
		printf("                     BINCATI               or BINGAMMAI                 \n");
		printf("For DNA data use:    GTRCAT                or GTRGAMMA                or\n");
		printf("                     GTRCATI               or GTRGAMMAI                 \n");
		printf("For AA data use:     PROTCATmatrixName[F]  or PROTGAMMAmatrixName[F]  or\n");
		printf("                     PROTCATImatrixName[F] or PROTGAMMAImatrixName[F]   \n");
		printf("The AA substitution matrix can be one of the following: \n");
		printf("DAYHOFF, DCMUT, JTT, MTREV, WAG, RTREV, CPREV, VT, BLOSUM62, MTMAM, LG, GTR\n\n");
		printf("With the optional \"F\" appendix you can specify if you want to use empirical base frequencies\n");
		printf("Please note that for mixed models you can in addition specify the per-gene model in\n");
		printf("the mixed model file (see manual for details)\n");
	      }
	    errorExit(-1);
	  }
	else
	  modelSet = 1;
	break;
      default:
	errorExit(-1);
    }
  }

  if(adef->shSupports)
    tr->useFastScaling = FALSE;

#ifdef _USE_PTHREADS
  if(NumberOfThreads < 2)
    {
      printf("\nThe number of threads is currently set to %d\n", NumberOfThreads);
      printf("Specify the number of threads to run via -T numberOfThreads\n");
      printf("NumberOfThreads must be set to an integer value greater than 1\n\n");
      errorExit(-1);
    }
#endif

  tr->useFloat = adef->useFloat;

  if(bSeedSet && xSeedSet)
    {
      printf("Error, you can't seed random seeds by using -x and -b at the same time\n");
      printf("use either -x or -b, exiting ......\n");
      errorExit(-1);
    }

  if(bSeedSet || xSeedSet)
    {
      if(!multipleRunsSet)
	{
	  printf("Error, you have specified a random number seed via -x or -b for some sort of bootstrapping,\n");
	  printf("but you have not specified a number of replicates via -N or -#, exiting ....\n");
	  errorExit(-1);
	}
      
      if(adef->multipleRuns == 1)
	{
	  printf("WARNING, you have specified a random number seed via -x or -b for some sort of bootstrapping,\n");
	  printf("but you have specified a number of replicates via -N or -# euqal to one\n");
	  printf("Are you really sure that this is what you want to do?\n");
	}

      if(tr->fastEPA_MP || tr->fastEPA_ML)
	{
	  printf("Error, you can's use the additional MP and ML heuristics for rapid evolutionary placement in\n");
	  printf("combination with bootstrapping\n");
	  errorExit(-1);
	}
    }


  if(adef->computeELW)
    {
      if(processID == 0)
	{
	  if(adef->boot == 0)
	    {
	      printf("Error, you must specify a bootstrap seed via \"-b\" to compute ELW statistics\n");
	      errorExit(-1);
	    }

	  if(adef->multipleRuns < 2)
	    {
	      printf("Error, you must specify the number of BS replicates via \"-#\" or \"-N\" to compute ELW statistics\n");
	      printf("it should be larger than 1, recommended setting is 100\n");
	      errorExit(-1);
	    }

	  if(!treesSet)
	    {
	      printf("Error, you must specify an input file containing several candidate trees\n");
	      printf("via \"-z\" to compute ELW statistics.\n");
	      errorExit(-1);
	    }

	  if(!isGamma(adef))
	    {
	      printf("Error ELW test can only be conducted undetr GAMMA or GAMMA+P-Invar models\n");
	      errorExit(-1);
	    }
	}
    }





  if(((!adef->boot) && (!adef->rapidBoot)) && adef->bootStopping)
    {
      if(processID == 0)
	{
	  printf("Can't use automatic bootstopping without actually doing a Bootstrap\n");
	  printf("Specify either -x randomNumberSeed (rapid) or -b randomNumberSeed (standard)\n");
	  errorExit(-1);
	}
    }

  if(adef->boot && adef->rapidBoot)
    {
      if(processID == 0)
	{
	  printf("Can't use standard and rapid BOOTSTRAP simultaneously\n");
	  errorExit(-1);
	}
    }

  if(adef->rapidBoot && !(adef->mode == CLASSIFY_ML))
    {
      if(processID == 0 && (adef->restart || treesSet) && !(groupSet || constraintSet))
	{
	  printf("Error, starting tree(s) will be ignored by rapid Bootstrapping\n");
	  errorExit(-1);
	}
    }

  if(adef->allInOne && (adef->rapidBoot == 0))
    {
      if(processID == 0)
	{
	  printf("Error, to carry out an ML search after a rapid BS inference you must specify a random number seed with -x\n");
	  errorExit(-1);
	}
    }

#ifdef _WAYNE_MPI
  if(adef->bootStopping && processes > 1)
    {
      if(processID == 0)
        printf("Error, MPI with processes > 1 does not work with bootstopping\n");
      MPI_Finalize();
      exit(-1);
    }
#endif
  

  if(adef->mode == PER_SITE_LL)
    {
      if(!isGamma(adef))
	{
	  if(processID == 0)
	    printf("\n ERROR: Computation of per-site log LHs is only allowed under GAMMA model of rate heterogeneity!\n");
	  errorExit(-1);
	}

      if(!treesSet)
	{
	  if(processID == 0)
	    printf("\n ERROR: For Computation of per-site log LHs you need to specify several input trees with \"-z\"\n");
	  errorExit(-1);
	}
    }



  if(adef->mode == SPLIT_MULTI_GENE && (!adef->useMultipleModel))
    {
      if(processID == 0)
	{
	  printf("\n  Error, you are trying to split a multi-gene alignment into individual genes with the \"-f s\" option\n");
	  printf("Without specifying a multiple model file with \"-q modelFileName\" \n");
	}
      errorExit(-1);
    }

  if(adef->mode == CALC_BIPARTITIONS && !treesSet)
    {
      if(processID == 0)
	printf("\n  Error, in bipartition computation mode you must specify a file containing multiple trees with the \"-z\" option\n");
      errorExit(-1);
    }

  if(adef->mode == CALC_BIPARTITIONS && !adef->restart)
    {
      if(processID == 0)
	printf("\n  Error, in bipartition computation mode you must specify a tree on which bipartition information will be drawn with the \"-t\" option\n");
      errorExit(-1);
    }

  if(!modelSet)
    {
      if(processID == 0)
	printf("\n Error, you must specify a model of substitution with the \"-m\" option\n");
      errorExit(-1);
    }

  if(adef->computeDistance)
    {
      if(isCat(adef))
	{
	  if(processID == 0)
	    printf("\n Error pairwise distance computation only allowed for GAMMA-based models of rate heterogeneity\n");
	  errorExit(-1);
	}

      if(adef->restart)
	{
	  if(adef->randomStartingTree)
	    {
	      if(processID == 0)
		printf("\n Error pairwise distance computation not allowed for random starting trees\n");
	      errorExit(-1);
	    }

	  if(adef->constraint)
	    {
	      if(processID == 0)
		printf("\n Error pairwise distance computation not allowed for binary backbone  constraint tree\n");
	      errorExit(-1);
	    }

	  if(adef->grouping)
	    {
	      if(processID == 0)
		printf("\n Error pairwise distance computation not allowed for constraint tree\n");
	      errorExit(-1);
	    }

	}

      if(adef->boot || adef->rapidBoot)
	{
	  if(processID == 0)
	    printf("\n Bootstrapping not implemented for pairwise distance computation\n");
	  errorExit(-1);
	}
    }








  if(!adef->restart && adef->mode == PARSIMONY_ADDITION)
    {
       if(processID == 0)
	 {
	   printf("\n You need to specify an incomplete binary input tree with \"-t\" to execute \n");
	   printf(" RAxML MP stepwise addition with \"-f p\"\n");
	 }
      errorExit(-1);
    }



  if(adef->restart && adef->randomStartingTree)
    {
      if(processID == 0)
	{
	  if(adef->constraint)
	    {
	      printf("\n Error you specified a binary constraint tree with -r AND the computation\n");
	      printf("of a random starting tree with -d for the same run\n");
	    }
	  else
	    {
	      if(adef->grouping)
		{
		  printf("\n Error you specified a multifurcating constraint tree with -g AND the computation\n");
		  printf("of a random starting tree with -d for the same run\n");
		}
	      else
		{
		  printf("\n Error you specified a starting tree with -t AND the computation\n");
		  printf("of a random starting tree with -d for the same run\n");
		}
	    }
	}
      errorExit(-1);
    }

  if(treeSet && constraintSet)
    {
      if(processID == 0)
	printf("\n Error you specified a binary constraint tree AND a starting tree for the same run\n");
      errorExit(-1);
    }


  if(treeSet && groupSet)
    {
      if(processID == 0)
	printf("\n Error you specified a multifurcating constraint tree AND a starting tree for the same run\n");
      errorExit(-1);
    }


  if(groupSet && constraintSet)
    {
      if(processID == 0)
	printf("\n Error you specified a bifurcating constraint tree AND a multifurcating constraint tree for the same run\n");
      errorExit(-1);
    }

  if(adef->restart && adef->startingTreeOnly)
    {
      if(processID == 0)
	{
	  printf("\n Error conflicting options: you want to compute only a parsimony starting tree with -y\n");
	  printf(" while you actually specified a starting tree with -t %s\n", tree_file);
	}
      errorExit(-1);
    }

  if((adef->mode == TREE_EVALUATION) && (!adef->restart))
    {
      if(processID == 0)
	printf("\n Error: please specify a treefile for the tree you want to evaluate with -t\n");
      errorExit(-1);
    }

#if defined PARALLEL || defined _WAYNE_MPI

  if(adef->mode == SPLIT_MULTI_GENE)
    {
      if(processID == 0)
	printf("Multi gene alignment splitting (-f s) not implemented for the MPI-Version\n");
      errorExit(-1);
    }

  if(adef->mode == TREE_EVALUATION)
    {
      if(processID == 0)
	printf("Tree Evaluation mode (-f e) not implemented for the MPI-Version\n");
      errorExit(-1);
    }

  if(adef->mode == CALC_BIPARTITIONS)
    {
      if(processID == 0)
	 printf("Computation of bipartitions (-f b) not implemented for the MPI-Version\n");
      errorExit(-1);
    }

  if(adef->multipleRuns == 1)
    {
      if(processID == 0)
	{
	  printf("Error: you are running the parallel MPI program but only want to compute one tree\n");
	  printf("For the MPI version you must specify a number of trees greater than 1 with the -# or -N option\n");
	}
      errorExit(-1);
    }

#endif

   if((adef->mode == TREE_EVALUATION) && (isCat(adef)))
     {
       if(processID == 0)
	 {
	   printf("\n Error: No tree evaluation with GTRCAT/PROTCAT possible\n");
	   printf("the GTRCAT likelihood values are instable at present and should not\n");
	   printf("be used to compare trees based on ML values\n");
	 }
       errorExit(-1);
     }

  if(!nameSet)
    {
      if(processID == 0)
	printf("\n Error: please specify a name for this run with -n\n");
      errorExit(-1);
    }

  if(! alignmentSet && !adef->readTaxaOnly)
    {
      if(processID == 0)
	printf("\n Error: please specify an alignment for this run with -s\n");
      errorExit(-1);
    }


  /*if(!workDirSet)*/
    {
#ifdef WIN32
    if(workdir[0]==0 || !(workdir[0] == '\\' || (workdir[0] != 0 && workdir[1] == ':')))
	{
	  getcwd(buf,sizeof(buf));
	  if( buf[strlen(buf)-1] != '\\') strcat(buf,"\\");
	  strcat(buf,workdir);
	  if( buf[strlen(buf)-1] != '\\') strcat(buf,"\\");
	  strcpy(workdir,buf);
	}
#else
      if(workdir[0]==0 || workdir[0] != '/')
	{
	  getcwd(buf,sizeof(buf));
	  if( buf[strlen(buf)-1] != '/') strcat(buf,"/");
	  strcat(buf,workdir);
	  if( buf[strlen(buf)-1] != '/') strcat(buf,"/");
	  strcpy(workdir,buf);
	}
#endif
    }


  return;
}




void errorExit(int e)
{
#ifdef PARALLEL
  MPI_Status msgStatus;
  int i, dummy;

  if(processID == 0)
    {
      for(i = 1; i < numOfWorkers; i++)
	MPI_Send(&dummy, 1, MPI_INT, i, FINALIZE, MPI_COMM_WORLD);

      MPI_Finalize();
      exit(e);
    }
  else
    {
      MPI_Recv(&dummy, 1, MPI_INT, 0, FINALIZE, MPI_COMM_WORLD, &msgStatus);
      MPI_Finalize();
      exit(e);
    }
#else
#ifdef _WAYNE_MPI
  MPI_Finalize();
#endif

  exit(e);
#endif
}



static void makeFileNames(void)
{
  int infoFileExists = 0;
#ifdef PARALLEL
  MPI_Status msgStatus;
#endif

  strcpy(permFileName,         workdir);
  strcpy(resultFileName,       workdir);
  strcpy(logFileName,          workdir);
  strcpy(checkpointFileName,   workdir);
  strcpy(infoFileName,         workdir);
  strcpy(randomFileName,       workdir);
  strcpy(bootstrapFileName,    workdir);
  strcpy(bipartitionsFileName, workdir);
  strcpy(bipartitionsFileNameBranchLabels, workdir);
  strcpy(ratesFileName,        workdir);
  strcpy(lengthFileName,       workdir);
  strcpy(lengthFileNameModel,  workdir);
  strcpy( perSiteLLsFileName,  workdir);

  strcat(permFileName,         "RAxML_parsimonyTree.");
  strcat(resultFileName,       "RAxML_result.");
  strcat(logFileName,          "RAxML_log.");
  strcat(checkpointFileName,   "RAxML_checkpoint.");
  strcat(infoFileName,         "RAxML_info.");
  strcat(randomFileName,       "RAxML_randomTree.");
  strcat(bootstrapFileName,    "RAxML_bootstrap.");
  strcat(bipartitionsFileName, "RAxML_bipartitions.");
  strcat(bipartitionsFileNameBranchLabels, "RAxML_bipartitionsBranchLabels.");
  strcat(ratesFileName,        "RAxML_perSiteRates.");
  strcat(lengthFileName,       "RAxML_treeLength.");
  strcat(lengthFileNameModel,  "RAxML_treeLengthModel.");
  strcat(perSiteLLsFileName,   "RAxML_perSiteLLs.");

  strcat(permFileName,         run_id);
  strcat(resultFileName,       run_id);
  strcat(logFileName,          run_id);
  strcat(checkpointFileName,   run_id);
  strcat(infoFileName,         run_id);
  strcat(randomFileName,       run_id);
  strcat(bootstrapFileName,    run_id);
  strcat(bipartitionsFileName, run_id);
  strcat(bipartitionsFileNameBranchLabels, run_id);  
  strcat(ratesFileName,        run_id);
  strcat(lengthFileName,       run_id);
  strcat(lengthFileNameModel,  run_id);
  strcat(perSiteLLsFileName,   run_id);

  if(processID == 0)
    {
      infoFileExists = filexists(infoFileName);

#ifdef PARALLEL
      {
	int i;

	for(i = 1; i < numOfWorkers; i++)
	  MPI_Send(&infoFileExists, 1, MPI_INT, i, FINALIZE, MPI_COMM_WORLD);
      }
#endif

      if(infoFileExists)
	{
	  printf("RAxML output files with the run ID <%s> already exist \n", run_id);
	  printf("in directory %s ...... exiting\n", workdir);
#ifdef PARALLEL
	  MPI_Finalize();
	  exit(-1);
#else
	  //exit(-1);
#endif
	}
    }
#ifdef PARALLEL
  else
    {
      MPI_Recv(&infoFileExists, 1, MPI_INT, 0, FINALIZE, MPI_COMM_WORLD, &msgStatus);
      if(infoFileExists)
	{
	  MPI_Finalize();
	  exit(-1);
	}
    }
#endif
}




 




/***********************reading and initializing input ******************/


/********************PRINTING various INFO **************************************/


static void printModelAndProgramInfo(tree *tr, analdef *adef, int argc, char *argv[])
{
    const partitionLengths *pl;
  if(processID == 0)
    {
      int i, model;
      FILE *infoFile = myfopen(infoFileName, "a");
      char modelType[128];

      if(!adef->readTaxaOnly)
	{
	  if(adef->useInvariant)
	    strcpy(modelType, "GAMMA+P-Invar");
	  else
	    strcpy(modelType, "GAMMA");
	}
     
      printBoth(infoFile, "\n\nThis is %s version %s released by Alexandros Stamatakis in %s.\n\n",  programName, programVersion, programDate);
      printBoth(infoFile, "With greatly appreciated code contributions by:\n");
      printBoth(infoFile, "Andre Aberer (TUM)\n");     
      printBoth(infoFile, "Simon Berger (TUM)\n");
      printBoth(infoFile, "John Cazes (TACC)\n");
      printBoth(infoFile, "Michael Ott (TUM)\n"); 
      printBoth(infoFile, "Nick Pattengale (UNM)\n"); 
      printBoth(infoFile, "Wayne Pfeiffer (SDSC)\n\n");
      
      if(!adef->readTaxaOnly)
	{
	  if(!adef->compressPatterns)
	    printBoth(infoFile, "\nAlignment has %d columns\n\n",  tr->cdta->endsite);
	  else
	    printBoth(infoFile, "\nAlignment has %d distinct alignment patterns\n\n",  tr->cdta->endsite);
	  
	  if(adef->useInvariant)
	    printBoth(infoFile, "Found %d invariant alignment patterns that correspond to %d columns \n", tr->numberOfInvariableColumns, tr->weightOfInvariableColumns);

	  printBoth(infoFile, "Proportion of gaps and completely undetermined characters in this alignment: %3.2f%s\n", 100.0 * adef->gapyness, "%");
	}

      switch(adef->mode)
	{
	case THOROUGH_PARSIMONY:
	  printBoth(infoFile, "\nRAxML more exhaustive parsimony search with a ratchet.\n");
	  printBoth(infoFile, "For a faster and better implementation of MP searches please use TNT by Pablo Goloboff.\n\n");
	  break;
	case DISTANCE_MODE:
	  printBoth(infoFile, "\nRAxML Computation of pairwise distances\n\n");
	  break;
	case TREE_EVALUATION :
	  printBoth(infoFile, "\nRAxML Model Optimization up to an accuracy of %f log likelihood units\n\n", adef->likelihoodEpsilon);
	  break;
	case  BIG_RAPID_MODE:
	  if(adef->rapidBoot)
	    {
	      if(adef->allInOne)
		printBoth(infoFile, "\nRAxML rapid bootstrapping and subsequent ML search\n\n");
	      else
		printBoth(infoFile,  "\nRAxML rapid bootstrapping algorithm\n\n");
	    }
	  else
	    printBoth(infoFile, "\nRAxML rapid hill-climbing mode\n\n");
	  break;
	case CALC_BIPARTITIONS:
	  printBoth(infoFile, "\nRAxML Bipartition Computation: Drawing support values from trees in file %s onto tree in file %s\n\n",
		    bootStrapFile, tree_file);
	  break;
	case PER_SITE_LL:
	  printBoth(infoFile, "\nRAxML computation of per-site log likelihoods\n");
	  break;
	case PARSIMONY_ADDITION:
	  printBoth(infoFile, "\nRAxML stepwise MP addition to incomplete starting tree\n\n");
	  break;
	case CLASSIFY_ML:
	  printBoth(infoFile, "\nRAxML classification algorithm\n\n");
	  break;
	case GENERATE_BS:
	  printBoth(infoFile, "\nRAxML BS replicate generation\n\n");
	  break;
	case COMPUTE_ELW:
	  printBoth(infoFile, "\nRAxML ELW test\n\n");
	  break;
	case BOOTSTOP_ONLY:
	  printBoth(infoFile, "\nRAxML a posteriori Bootstrap convergence assessment\n\n");
	  break;
	case CONSENSUS_ONLY:
	  printBoth(infoFile, "\nRAxML consensus tree computation\n\n");
	  break;
	case COMPUTE_LHS:
	  printBoth(infoFile, "\nRAxML computation of likelihoods for a set of trees\n\n");
	  break;
	case COMPUTE_BIPARTITION_CORRELATION:
	  printBoth(infoFile, "\nRAxML computation of bipartition support correlation on two sets of trees\n\n");
	  break;
	case COMPUTE_RF_DISTANCE:
	  printBoth(infoFile, "\nRAxML computation of RF distances for all pairs of trees in a set of trees\n\n");
	  break;
	case MORPH_CALIBRATOR:
	  printBoth(infoFile, "\nRAxML morphological calibrator using Maximum Likelihood\n\n");
	  break;
	case MORPH_CALIBRATOR_PARSIMONY:
	  printBoth(infoFile, "\nRAxML morphological calibrator using Parsimony\n\n");
	  break;	  
	case MESH_TREE_SEARCH:
	  printBoth(infoFile, "\nRAxML experimental mesh tree search\n\n");
	  break;
	case FAST_SEARCH:
	  printBoth(infoFile, "\nRAxML experimental very fast tree search\n\n");
	  break;
	default:
	  assert(0);
	}

      if(adef->mode != THOROUGH_PARSIMONY)
	{ 
	  if(!adef->readTaxaOnly)
	    {
	      if(adef->perGeneBranchLengths)
		printBoth(infoFile, "Using %d distinct models/data partitions with individual per partition branch length optimization\n\n\n", tr->NumberOfModels);
	      else
		printBoth(infoFile, "Using %d distinct models/data partitions with joint branch length optimization\n\n\n", tr->NumberOfModels);
	    }
	}

      if(adef->mode == BIG_RAPID_MODE)
	{
	  if(adef->rapidBoot)
	    {
	      if(adef->allInOne)
		printBoth(infoFile, "\nExecuting %d rapid bootstrap inferences and thereafter a thorough ML search \n\n", adef->multipleRuns);
	      else
		printBoth(infoFile, "\nExecuting %d rapid bootstrap inferences\n\n", adef->multipleRuns);
	    }
	  else
	    {
	      if(adef->boot)
		printBoth(infoFile, "Executing %d non-parametric bootstrap inferences\n\n", adef->multipleRuns);
	      else
		{
		  char treeType[1024];

		  if(adef->restart)
		    strcpy(treeType, "user-specifed");
		  else
		    {
		      if(adef->randomStartingTree)
			strcpy(treeType, "distinct complete random");
		      else
			strcpy(treeType, "distinct randomized MP");
		    }

		  printBoth(infoFile, "Executing %d inferences on the original alignment using %d %s trees\n\n",
			    adef->multipleRuns, adef->multipleRuns, treeType);
		}
	    }
	}


      if(!adef->readTaxaOnly)
	{
	  if(adef->mode != THOROUGH_PARSIMONY)
	    printBoth(infoFile, "All free model parameters will be estimated by RAxML\n");

	  if(adef->mode != THOROUGH_PARSIMONY)
	    {
	      if(tr->rateHetModel == GAMMA || tr->rateHetModel == GAMMA_I)
		printBoth(infoFile, "%s model of rate heteorgeneity, ML estimate of alpha-parameter\n\n", modelType);
	      else
		{
		  printBoth(infoFile, "ML estimate of %d per site rate categories\n\n", adef->categories);
		  if(adef->mode != CLASSIFY_ML)
		    printBoth(infoFile, "Likelihood of final tree will be evaluated and optimized under %s\n\n", modelType);
		}
	      
	      if(adef->mode != CLASSIFY_ML)
		printBoth(infoFile, "%s Model parameters will be estimated up to an accuracy of %2.10f Log Likelihood units\n\n",
			  modelType, adef->likelihoodEpsilon);
	    }

	  for(model = 0; model < tr->NumberOfModels; model++)
	    {
	      printBoth(infoFile, "Partition: %d\n", model);
	      printBoth(infoFile, "Alignment Patterns: %d\n", tr->partitionData[model].upper - tr->partitionData[model].lower);
	      printBoth(infoFile, "Name: %s\n", tr->partitionData[model].partitionName);
              
	      assert(tr->mxtips == nofSpecies);
              assert(tr->multiGene == 0);
              
              pl = getPartitionLengths(&(tr->partitionData[0]));
              assert(pl->eiLength == 12);
              assert(pl->eignLength == 3);
              assert(pl->evLength == 16);
              assert(pl->tipVectorLength == 64);
              assert(pl->leftLength == 16);
              assert(pl->rightLength == 16);
              assert(tr->maxCategories == 25);
              
              assert(tr->discreteRateCategories == 4);
              assert(tr->partitionData[0].states == 4);
              
              assert(tr->partitionData[0].upper == alignLength);
              assert(tr->partitionData[0].lower == 0);
              assert(tr->innerNodes == nofSpecies);
              
              
              
	      switch(tr->partitionData[model].dataType)
		{
		case DNA_DATA:
		  printBoth(infoFile, "DataType: DNA\n");
		  if(adef->mode != THOROUGH_PARSIMONY)
		    printBoth(infoFile, "Substitution Matrix: GTR\n");
		  break;
		case AA_DATA:
		  assert(tr->partitionData[model].protModels >= 0 && tr->partitionData[model].protModels < NUM_PROT_MODELS);
		  printBoth(infoFile, "DataType: AA\n");
		  if(adef->mode != THOROUGH_PARSIMONY)
		    {
		      printBoth(infoFile, "Substitution Matrix: %s\n", (adef->userProteinModel)?"External user-specified model":protModels[tr->partitionData[model].protModels]);
		      printBoth(infoFile, "%s Base Frequencies:\n", (tr->partitionData[model].protFreqs == 1)?"Empirical":"Fixed");
		    }
		  break;
		case BINARY_DATA:
		  printBoth(infoFile, "DataType: BINARY/MORPHOLOGICAL\n");
		  if(adef->mode != THOROUGH_PARSIMONY)
		    printBoth(infoFile, "Substitution Matrix: Uncorrected\n");
		  break;
		case SECONDARY_DATA:
		  printBoth(infoFile, "DataType: SECONDARY STRUCTURE\n");
		  if(adef->mode != THOROUGH_PARSIMONY)
		    printBoth(infoFile, "Substitution Matrix: %s\n", secondaryModelList[tr->secondaryStructureModel]);
		  break;
		case SECONDARY_DATA_6:
		  printBoth(infoFile, "DataType: SECONDARY STRUCTURE 6 STATE\n");
		  if(adef->mode != THOROUGH_PARSIMONY)
		    printBoth(infoFile, "Substitution Matrix: %s\n", secondaryModelList[tr->secondaryStructureModel]);
		  break;
		case SECONDARY_DATA_7:
		  printBoth(infoFile, "DataType: SECONDARY STRUCTURE 7 STATE\n");
		  if(adef->mode != THOROUGH_PARSIMONY)
		    printBoth(infoFile, "Substitution Matrix: %s\n", secondaryModelList[tr->secondaryStructureModel]);
		  break;
		case GENERIC_32:
		  printBoth(infoFile, "DataType: Multi-State with %d distinct states in use (maximum 32)\n",tr->partitionData[model].states);		  
		  switch(tr->multiStateModel)
		    {
		    case ORDERED_MULTI_STATE:
		      printBoth(infoFile, "Substitution Matrix: Ordered Likelihood\n");
		      break;
		    case MK_MULTI_STATE:
		      printBoth(infoFile, "Substitution Matrix: MK model\n");
		      break;
		    case GTR_MULTI_STATE:
		      printBoth(infoFile, "Substitution Matrix: GTR\n");
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case GENERIC_64:
		  printBoth(infoFile, "DataType: Codon\n");		  
		  break;		
		default:
		  assert(0);
		}
	      printBoth(infoFile, "\n\n\n");
	    }
	}

      printBoth(infoFile, "\n");

      printBoth(infoFile, "RAxML was called as follows:\n\n");
      for(i = 0; i < argc; i++)
	printBoth(infoFile,"%s ", argv[i]);
      printBoth(infoFile,"\n\n\n");

      fclose(infoFile);
    }
}

void printResult(tree *tr, analdef *adef, boolean finalPrint)
{
  FILE *logFile;
  char temporaryFileName[1024] = "", treeID[64] = "";

  strcpy(temporaryFileName, resultFileName);

  switch(adef->mode)
    {
    case MORPH_CALIBRATOR_PARSIMONY:
    case MESH_TREE_SEARCH:    
    case MORPH_CALIBRATOR:
      break;
    case TREE_EVALUATION:


      Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, finalPrint, adef, SUMMARIZE_LH, FALSE);

      logFile = myfopen(temporaryFileName, "w");
      fprintf(logFile, "%s", tr->tree_string);
      fclose(logFile);

      if(adef->perGeneBranchLengths)
	printTreePerGene(tr, adef, temporaryFileName, "w");


      break;
    case BIG_RAPID_MODE:
      if(!adef->boot)
	{
	  if(adef->multipleRuns > 1)
	    {
	      sprintf(treeID, "%d", tr->treeID);
	      strcat(temporaryFileName, ".RUN.");
	      strcat(temporaryFileName, treeID);
	    }


	  if(finalPrint)
	    {
	      switch(tr->rateHetModel)
		{
		case GAMMA:
		case GAMMA_I:
		  Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, finalPrint, adef,
			      SUMMARIZE_LH, FALSE);

		  logFile = myfopen(temporaryFileName, "w");
		  fprintf(logFile, "%s", tr->tree_string);
		  fclose(logFile);

		  if(adef->perGeneBranchLengths)
		    printTreePerGene(tr, adef, temporaryFileName, "w");
		  break;
		case CAT:
		  Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef,
			      NO_BRANCHES, FALSE);

		  logFile = myfopen(temporaryFileName, "w");
		  fprintf(logFile, "%s", tr->tree_string);
		  fclose(logFile);

		  break;
		default:
		  assert(0);
		}
	    }
	  else
	    {
	      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef,
			  NO_BRANCHES, FALSE);
	      logFile = myfopen(temporaryFileName, "w");
	      fprintf(logFile, "%s", tr->tree_string);
	      fclose(logFile);
	    }
	}
      break;
    default:
      printf("FATAL ERROR call to printResult from undefined STATE %d\n", adef->mode);
      exit(-1);
      break;
    }
}

void printBootstrapResult(tree *tr, analdef *adef, boolean finalPrint)
{
#ifdef PARALLEL
  if(processID == 0)
#endif
    {
      FILE *logFile;

      if(adef->mode == BIG_RAPID_MODE && (adef->boot || adef->rapidBoot))
	{
#ifndef PARALLEL
	  if(adef->bootstrapBranchLengths)
	    {
	      Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, finalPrint, adef, SUMMARIZE_LH, FALSE);
	      logFile = myfopen(bootstrapFileName, "a");
	      fprintf(logFile, "%s", tr->tree_string);
	      fclose(logFile);
	      if(adef->perGeneBranchLengths)
		printTreePerGene(tr, adef, bootstrapFileName, "a");
	    }
	  else
	    {
	      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, FALSE);
	      logFile = myfopen(bootstrapFileName, "a");
	      fprintf(logFile, "%s", tr->tree_string);
	      fclose(logFile);
	    }
#else
	  logFile = myfopen(bootstrapFileName, "a");
	  fprintf(logFile, "%s", tr->tree_string);
	  fclose(logFile);
#endif
	}
      else
	{
	  printf("FATAL ERROR in  printBootstrapResult\n");
	  exit(-1);
	}
    }
}



void printBipartitionResult(tree *tr, analdef *adef, boolean finalPrint)
{
  if(processID == 0 || adef->allInOne)
    {
      FILE *logFile;

      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, TRUE, finalPrint, adef, NO_BRANCHES, FALSE);
      logFile = myfopen(bipartitionsFileName, "a");
      fprintf(logFile, "%s", tr->tree_string);
      fclose(logFile);

      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, TRUE);

      logFile = myfopen(bipartitionsFileNameBranchLabels, "a");
      fprintf(logFile, "%s", tr->tree_string);
      fclose(logFile);

    }
}



void printLog(tree *tr, analdef *adef, boolean finalPrint)
{
  FILE *logFile;
  char temporaryFileName[1024] = "", checkPoints[1024] = "", treeID[64] = "";
  double lh, t;

  lh = tr->likelihood;
  t = gettime() - masterTime;

  strcpy(temporaryFileName, logFileName);
  strcpy(checkPoints,       checkpointFileName);

  switch(adef->mode)
    {
    case TREE_EVALUATION:
      logFile = myfopen(temporaryFileName, "a");

      printf("%f %f\n", t, lh);
      fprintf(logFile, "%f %f\n", t, lh);

      fclose(logFile);
      break;
    case BIG_RAPID_MODE:
      if(adef->boot || adef->rapidBoot)
	{
	  /* testing only printf("%f %f\n", t, lh);*/
	  /* NOTHING PRINTED so far */
	}
      else
	{
	  if(adef->multipleRuns > 1)
	    {
	      sprintf(treeID, "%d", tr->treeID);
	      strcat(temporaryFileName, ".RUN.");
	      strcat(temporaryFileName, treeID);

	      strcat(checkPoints, ".RUN.");
	      strcat(checkPoints, treeID);
	    }


	  if(!adef->checkpoints)
	    {
	      logFile = myfopen(temporaryFileName, "a");
#ifndef PARALLEL
	      /*printf("%f %1.20f\n", t, lh);*/
#endif
	      fprintf(logFile, "%f %f\n", t, lh);

	      fclose(logFile);
	    }
	  else
	    {
	      logFile = myfopen(temporaryFileName, "a");
#ifndef PARALLEL
	      /*printf("%f %f %d\n", t, lh, tr->checkPointCounter);*/
#endif
	      fprintf(logFile, "%f %f %d\n", t, lh, tr->checkPointCounter);

	      fclose(logFile);

	      strcat(checkPoints, ".");

	      sprintf(treeID, "%d", tr->checkPointCounter);
	      strcat(checkPoints, treeID);

	      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, FALSE);

	      logFile = myfopen(checkPoints, "a");
	      fprintf(logFile, "%s", tr->tree_string);
	      fclose(logFile);

	      tr->checkPointCounter++;
	    }
	}
      break;
    case MORPH_CALIBRATOR_PARSIMONY:
    case MORPH_CALIBRATOR:
      break;
    default:
      assert(0);
    }
}



void printStartingTree(tree *tr, analdef *adef, boolean finalPrint)
{
  if(adef->boot)
    {
      /* not printing starting trees for bootstrap */
    }
  else
    {
      FILE *treeFile;
      char temporaryFileName[1024] = "", treeID[64] = "";

      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, FALSE);

      if(adef->randomStartingTree)
	strcpy(temporaryFileName, randomFileName);
      else
	strcpy(temporaryFileName, permFileName);

      if(adef->multipleRuns > 1)
	{
	  sprintf(treeID, "%d", tr->treeID);
	  strcat(temporaryFileName, ".RUN.");
	  strcat(temporaryFileName, treeID);
	}

      treeFile = myfopen(temporaryFileName, "a");
      fprintf(treeFile, "%s", tr->tree_string);
      fclose(treeFile);
    }
}

void writeInfoFile(analdef *adef, tree *tr, double t)
{
#ifdef PARALLEL
  if(processID == 0)
#endif
    {      
      switch(adef->mode)
	{
	case MESH_TREE_SEARCH:
	  break;
	case TREE_EVALUATION:
	  break;
	case BIG_RAPID_MODE:
	  if(adef->boot || adef->rapidBoot)
	    {
	      if(!adef->initialSet)	
		printBothOpen("Bootstrap[%d]: Time %f seconds, bootstrap likelihood %f, best rearrangement setting %d\n", tr->treeID, t, tr->likelihood,  adef->bestTrav);		
	      else	
		printBothOpen("Bootstrap[%d]: Time %f seconds, bootstrap likelihood %f\n", tr->treeID, t, tr->likelihood);		
	    }
	  else
	    {
	      int model;
	      char modelType[128];

	      switch(tr->rateHetModel)
		{
		case GAMMA_I:
		  strcpy(modelType, "GAMMA+P-Invar");
		  break;
		case GAMMA:
		  strcpy(modelType, "GAMMA");
		  break;
		case CAT:
		  strcpy(modelType, "CAT");
		  break;
		default:
		  assert(0);
		}

	      if(!adef->initialSet)		
		printBothOpen("Inference[%d]: Time %f %s-based likelihood %f, best rearrangement setting %d\n",
			      tr->treeID, t, modelType, tr->likelihood,  adef->bestTrav);		 
	      else		
		printBothOpen("Inference[%d]: Time %f %s-based likelihood %f\n",
			      tr->treeID, t, modelType, tr->likelihood);		 

	      {
		FILE *infoFile = myfopen(infoFileName, "a");

		for(model = 0; model < tr->NumberOfModels; model++)
		  {
		    fprintf(infoFile, "alpha[%d]: %f ", model, tr->partitionData[model].alpha);
		    if(adef->useInvariant)
		      fprintf(infoFile, "invar[%d]: %f ", model, tr->partitionData[model].propInvariant);
#ifndef PARALLEL
		    if(tr->partitionData[model].dataType == DNA_DATA)
		      {
			int 
			  k,
			  states = tr->partitionData[model].states,
			  rates = ((states * states - states) / 2);
			
			fprintf(infoFile, "rates[%d] ac ag at cg ct gt: ", model);
			for(k = 0; k < rates; k++)
			  fprintf(infoFile, "%f ", tr->partitionData[model].substRates[k]);
		      }		 
#endif
		  }

		fprintf(infoFile, "\n");
		fclose(infoFile);
	      }
	    }
	  break;
	default:
	  assert(0);
	}      
    }
}

static void printFreqs(int n, double *f, char **names)
{
  int k;

  for(k = 0; k < n; k++)
    printBothOpen("freq pi(%s): %f\n", names[k], f[k]);
}

static void printRatesDNA_BIN(int n, double *r, char **names)
{
  int i, j, c;

  for(i = 0, c = 0; i < n; i++)
    {
      for(j = i + 1; j < n; j++)
	{
	  if(i == n - 2 && j == n - 1)
	    printBothOpen("rate %s <-> %s: %f\n", names[i], names[j], 1.0);
	  else
	    printBothOpen("rate %s <-> %s: %f\n", names[i], names[j], r[c]);
	  c++;
	}
    }
}

static void printRatesRest(int n, double *r, char **names)
{
  int i, j, c;

  for(i = 0, c = 0; i < n; i++)
    {
      for(j = i + 1; j < n; j++)
	{
	  printBothOpen("rate %s <-> %s: %f\n", names[i], names[j], r[c]);
	  c++;
	}
    }
}


void printModelParams(tree *tr, analdef *adef)
{
  int
    model;

  double
    *f = (double*)NULL,
    *r = (double*)NULL;

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      double tl;
      char typeOfData[1024];

      switch(tr->partitionData[model].dataType)
	{
	case AA_DATA:
	  strcpy(typeOfData,"AA");
	  break;
	case DNA_DATA:
	  strcpy(typeOfData,"DNA");
	  break;
	case BINARY_DATA:
	  strcpy(typeOfData,"BINARY/MORPHOLOGICAL");
	  break;
	case SECONDARY_DATA:
	  strcpy(typeOfData,"SECONDARY 16 STATE MODEL USING ");
	  strcat(typeOfData, secondaryModelList[tr->secondaryStructureModel]);
	  break;
	case SECONDARY_DATA_6:
	  strcpy(typeOfData,"SECONDARY 6 STATE MODEL USING ");
	  strcat(typeOfData, secondaryModelList[tr->secondaryStructureModel]);
	  break;
	case SECONDARY_DATA_7:
	  strcpy(typeOfData,"SECONDARY 7 STATE MODEL USING ");
	  strcat(typeOfData, secondaryModelList[tr->secondaryStructureModel]);
	  break;
	case GENERIC_32:
	  strcpy(typeOfData,"Multi-State");
	  break;
	case GENERIC_64:
	  strcpy(typeOfData,"Codon"); 
	  break;
	default:
	  assert(0);
	}

      printBothOpen("Model Parameters of Partition %d, Name: %s, Type of Data: %s\n",
		    model, tr->partitionData[model].partitionName, typeOfData);
      printBothOpen("alpha: %f\n", tr->partitionData[model].alpha);

      if(adef->useInvariant)
	printBothOpen("invar: %f\n", tr->partitionData[model].propInvariant);

      if(adef->perGeneBranchLengths)
	tl = treeLength(tr, model);
      else
	tl = treeLength(tr, 0);

      printBothOpen("Tree-Length: %f\n", tl);

      f = tr->partitionData[model].frequencies;
      r = tr->partitionData[model].substRates;

      switch(tr->partitionData[model].dataType)
	{
	case AA_DATA:
	  {
	    char *freqNames[20] = {"A", "R", "N ","D", "C", "Q", "E", "G",
				   "H", "I", "L", "K", "M", "F", "P", "S",
				   "T", "W", "Y", "V"};

	    printRatesRest(20, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(20, f, freqNames);
	  }
	  break;
	case GENERIC_32:
	  {
	    char *freqNames[32] = {"0", "1", "2", "3", "4", "5", "6", "7", 
				   "8", "9", "A", "B", "C", "D", "E", "F",
				   "G", "H", "I", "J", "K", "L", "M", "N",
				   "O", "P", "Q", "R", "S", "T", "U", "V"}; 

	    printRatesRest(32, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(32, f, freqNames);
	  }
	  break;
	case GENERIC_64:
	  assert(0);
	  break;
	case DNA_DATA:
	  {
	    char *freqNames[4] = {"A", "C", "G", "T"};

	    printRatesDNA_BIN(4, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(4, f, freqNames);
	  }
	  break;
	case SECONDARY_DATA_6:
	   {
	    char *freqNames[6] = {"AU", "CG", "GC", "GU", "UA", "UG"};

	    printRatesRest(6, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(6, f, freqNames);
	  }
	  break;
	case SECONDARY_DATA_7:
	  {
	    char *freqNames[7] = {"AU", "CG", "GC", "GU", "UA", "UG", "REST"};

	    printRatesRest(7, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(7, f, freqNames);
	  }
	  break;
	case SECONDARY_DATA:
	  {
	    char *freqNames[16] = {"AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU",
				   "GA", "GC", "GG", "GU", "UA", "UC", "UG", "UU"};

	    printRatesRest(16, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(16, f, freqNames);
	  }
	  break;
	case BINARY_DATA:
	  {
	    char *freqNames[2] = {"0", "1"};

	    printRatesDNA_BIN(2, r, freqNames);
	    printBothOpen("\n");
	    printFreqs(2, f, freqNames);
	  }
	  break;
	default:
	  assert(0);
	}

      printBothOpen("\n");
    }
}

static void finalizeInfoFile(tree *tr, analdef *adef)
{
  if(processID == 0)
    {
      double t;

      t = gettime() - masterTime;

      switch(adef->mode)
	{
	case MESH_TREE_SEARCH:
	  break;
	case TREE_EVALUATION :
	  printBothOpen("\n\nOverall Time for Tree Evaluation %f\n", t);
	  printBothOpen("Final GAMMA  likelihood: %f\n", tr->likelihood);

	  {
	    int
	      params,
	      paramsBrLen;

	    if(tr->NumberOfModels == 1)
	      {
		if(adef->useInvariant)
		  {
		    params      = 1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */;
		    paramsBrLen = 1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
		      (2 * tr->mxtips - 3);
		  }
		else
		  {
		    params      = 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */;
		    paramsBrLen = 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
		      (2 * tr->mxtips - 3);
		  }
	      }
	    else
	      {
		if(tr->multiBranch)
		  {
		    if(adef->useInvariant)
		      {
			params      = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
			paramsBrLen = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
							    (2 * tr->mxtips - 3));
		      }
		    else
		      {
			params      = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
			paramsBrLen = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
							    (2 * tr->mxtips - 3));
		      }
		  }
		else
		  {
		    if(adef->useInvariant)
		      {
			params      = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
			paramsBrLen = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */)
			  + (2 * tr->mxtips - 3);
		      }
		    else
		      {
			params      = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
			paramsBrLen = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */)
			  + (2 * tr->mxtips - 3);
		      }

		  }
	      }

	    if(tr->partitionData[0].dataType == DNA_DATA)
	      {
		printBothOpen("Number of free parameters for AIC-TEST(BR-LEN): %d\n",    paramsBrLen);
		printBothOpen("Number of free parameters for AIC-TEST(NO-BR-LEN): %d\n", params);
	      }

	  }

	  printBothOpen("\n\n");

	  printModelParams(tr, adef);

	  printBothOpen("Final tree written to:                 %s\n", resultFileName);
	  printBothOpen("Execution Log File written to:         %s\n", logFileName);

	  break;
	case  BIG_RAPID_MODE:
	  if(adef->boot)
	    {
	      printBothOpen("\n\nOverall Time for %d Bootstraps %f\n", adef->multipleRuns, t);
	      printBothOpen("\n\nAverage Time per Bootstrap %f\n", (double)(t/((double)adef->multipleRuns)));
	      printBothOpen("All %d bootstrapped trees written to: %s\n", adef->multipleRuns, bootstrapFileName);
	    }
	  else
	    {
	      if(adef->multipleRuns > 1)
		{
		  double avgLH = 0;
		  double bestLH = unlikely;
		  int i, bestI  = 0;

		  for(i = 0; i < adef->multipleRuns; i++)
		    {
		      avgLH   += tr->likelihoods[i];
		      if(tr->likelihoods[i] > bestLH)
			{
			  bestLH = tr->likelihoods[i];
			  bestI  = i;
			}
		    }
		  avgLH /= ((double)adef->multipleRuns);

		  printBothOpen("\n\nOverall Time for %d Inferences %f\n", adef->multipleRuns, t);
		  printBothOpen("Average Time per Inference %f\n", (double)(t/((double)adef->multipleRuns)));
		  printBothOpen("Average Likelihood   : %f\n", avgLH);
		  printBothOpen("\n");
		  printBothOpen("Best Likelihood in run number %d: likelihood %f\n\n", bestI, bestLH);

		  if(adef->checkpoints)
		    printBothOpen("Checkpoints written to:                 %s.RUN.%d.* to %d.*\n", checkpointFileName, 0, adef->multipleRuns - 1);
		  if(!adef->restart)
		    {
		      if(adef->randomStartingTree)
			printBothOpen("Random starting trees written to:       %s.RUN.%d to %d\n", randomFileName, 0, adef->multipleRuns - 1);
		      else
			printBothOpen("Parsimony starting trees written to:    %s.RUN.%d to %d\n", permFileName, 0, adef->multipleRuns - 1);
		    }
		  printBothOpen("Final trees written to:                 %s.RUN.%d to %d\n", resultFileName,  0, adef->multipleRuns - 1);
		  printBothOpen("Execution Log Files written to:         %s.RUN.%d to %d\n", logFileName, 0, adef->multipleRuns - 1);
		  printBothOpen("Execution information file written to:  %s\n", infoFileName);
		}
	      else
		{
		  printBothOpen("\n\nOverall Time for 1 Inference %f\n", t);
		  printBothOpen("Likelihood   : %f\n", tr->likelihood);
		  printBothOpen("\n\n");

		  if(adef->checkpoints)
		  printBothOpen("Checkpoints written to:                %s.*\n", checkpointFileName);
		  if(!adef->restart)
		    {
		      if(adef->randomStartingTree)
			printBothOpen("Random starting tree written to:       %s\n", randomFileName);
		      else
			printBothOpen("Parsimony starting tree written to:    %s\n", permFileName);
		    }
		  printBothOpen("Final tree written to:                 %s\n", resultFileName);
		  printBothOpen("Execution Log File written to:         %s\n", logFileName);
		  printBothOpen("Execution information file written to: %s\n",infoFileName);
		}
	    }

	  break;
	case CALC_BIPARTITIONS:
	  printBothOpen("\n\nTime for Computation of Bipartitions %f\n", t);
	  printBothOpen("Tree with bipartitions written to file:  %s\n", bipartitionsFileName);
	  printBothOpen("Tree with bipartitions as branch labels written to file:  %s\n", bipartitionsFileNameBranchLabels);	  
	  printBothOpen("Execution information file written to :  %s\n",infoFileName);
	  break;
	case PER_SITE_LL:
	  printBothOpen("\n\nTime for Optimization of per-site log likelihoods %f\n", t);
	  printBothOpen("Per-site Log Likelihoods written to File %s in Tree-Puzzle format\n",  perSiteLLsFileName);
	  printBothOpen("Execution information file written to :  %s\n",infoFileName);

	  break;
	case PARSIMONY_ADDITION:
	  printBothOpen("\n\nTime for MP stepwise addition %f\n", t);
	  printBothOpen("Execution information file written to :  %s\n",infoFileName);
	  printBothOpen("Complete parsimony tree written to:      %s\n", permFileName);
	  break;
	default:
	  assert(0);
	}
    }

}


/************************************************************************************/


#ifdef _USE_PTHREADS






static void computeFraction(tree *localTree, int tid, int n)
{
  int
    i,
    model;

  for(model = 0; model < localTree->NumberOfModels; model++)
    {
      int width = 0;

      for(i = localTree->partitionData[model].lower; i < localTree->partitionData[model].upper; i++)
	if(i % n == tid)
	      width++;

      localTree->partitionData[model].width = width;
    }
}



static void threadFixModelIndices(tree *tr, tree *localTree, int tid, int n)
{
  size_t
    model,
    j,
    i,
    globalCounter = 0,
    localCounter  = 0,
    offset,
    countOffset,
    myLength = 0,
    memoryRequirements = 0;

  for(model = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      localTree->partitionData[model].lower      = tr->partitionData[model].lower;
      localTree->partitionData[model].upper      = tr->partitionData[model].upper;
    }

  computeFraction(localTree, tid, n);

  for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      
      if(!tr->useFloat)
	localTree->partitionData[model].sumBuffer       = &localTree->sumBuffer[offset];
      else
	localTree->partitionData[model].sumBuffer_FLOAT = &localTree->sumBuffer_FLOAT[offset];

      localTree->partitionData[model].perSiteLL    = &localTree->perSiteLLPtr[countOffset];
      localTree->partitionData[model].wr           = &localTree->wrPtr[countOffset];
      localTree->partitionData[model].wr2          = &localTree->wr2Ptr[countOffset];

      if(localTree->useFloat)
	{
	  localTree->partitionData[model].wr_FLOAT           = &localTree->wrPtr_FLOAT[countOffset];
	  localTree->partitionData[model].wr2_FLOAT          = &localTree->wr2Ptr_FLOAT[countOffset]; 
	}

      localTree->partitionData[model].wgt          = &localTree->wgtPtr[countOffset];
      localTree->partitionData[model].invariant    = &localTree->invariantPtr[countOffset];
      localTree->partitionData[model].rateCategory = &localTree->rateCategoryPtr[countOffset];     

      countOffset += localTree->partitionData[model].width;

      offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * (size_t)(localTree->partitionData[model].width);      
    }

  myLength           = countOffset;
  memoryRequirements = offset;


  /* figure in data */   

  for(i = 0; i < (size_t)localTree->mxtips; i++)
    {
      for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
	{
	  localTree->partitionData[model].yVector[i+1]   = &localTree->y_ptr[i * myLength + countOffset];
	  countOffset +=  localTree->partitionData[model].width;
	}
      assert(countOffset == myLength);
    }

  for(i = 0; i < (size_t)localTree->innerNodes; i++)
    {
      for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
	{
	  size_t width = localTree->partitionData[model].width;

	  if(!tr->useFastScaling)	  
	    localTree->partitionData[model].expVector[i] = &localTree->expArray[i * myLength + countOffset];

	  /*localTree->partitionData[model].yVector[i+1]   = &localTree->y_ptr[i * myLength + countOffset];*/

	  if(!tr->useFloat)
	    {
	      localTree->partitionData[model].xVector[i]   = &localTree->likelihoodArray[i * memoryRequirements + offset];
	      localTree->partitionData[model].pVector[i]   = (parsimonyVector *)localTree->partitionData[model].xVector[i];
	    }
	  else
	    {
	      localTree->partitionData[model].xVector_FLOAT[i]   = &localTree->likelihoodArray_FLOAT[i * memoryRequirements + offset];
	      localTree->partitionData[model].pVector[i]         = (parsimonyVector *)localTree->partitionData[model].xVector_FLOAT[i];
	    }

	  countOffset += width;

	  offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;
	  
	}
      assert(countOffset == myLength);
    }

  for(model = 0, globalCounter = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      for(localCounter = 0, i = (size_t)localTree->partitionData[model].lower;  i < (size_t)localTree->partitionData[model].upper; i++)
	{
	  if(i % (size_t)n == (size_t)tid)
	    {
	      localTree->partitionData[model].wgt[localCounter]          = tr->cdta->aliaswgt[globalCounter];
	      localTree->partitionData[model].wr[localCounter]           = tr->cdta->wr[globalCounter];
	      localTree->partitionData[model].wr2[localCounter]          = tr->cdta->wr2[globalCounter];
	      localTree->partitionData[model].invariant[localCounter]    = tr->invariant[globalCounter];
	      localTree->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[globalCounter];

	      if(localTree->useFloat)
		{
		  localTree->partitionData[model].wr_FLOAT[localCounter]           = tr->cdta->wr_FLOAT[globalCounter];
		  localTree->partitionData[model].wr2_FLOAT[localCounter]          = tr->cdta->wr2_FLOAT[globalCounter]; 
		}


	      for(j = 1; j <= (size_t)localTree->mxtips; j++)
	       localTree->partitionData[model].yVector[j][localCounter] = tr->yVector[j][globalCounter];

	      localCounter++;
	    }
	  globalCounter++;
	}
    }

}


static void initPartition(tree *tr, tree *localTree, int tid)
{
  int model;

  localTree->threadID = tid; 

  if(tid > 0)
    {
      int totalLength = 0;

      localTree->innerNodes              = tr->innerNodes;
      localTree->useFastScaling          = tr->useFastScaling;
      localTree->maxCategories           = tr->maxCategories;
      localTree->useFloat                = tr->useFloat;
      localTree->originalCrunchedLength  = tr->originalCrunchedLength;
      localTree->NumberOfModels          = tr->NumberOfModels;
      localTree->mxtips                  = tr->mxtips;
      localTree->multiBranch             = tr->multiBranch;
      localTree->multiGene               = tr->multiGene;
      assert(localTree->multiGene == 0);
      localTree->numBranches             = tr->numBranches;
      localTree->lhs                     = (double*)malloc(sizeof(double)   * localTree->originalCrunchedLength);
      localTree->executeModel            = (boolean*)malloc(sizeof(boolean) * localTree->NumberOfModels);
      localTree->perPartitionLH          = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);
      localTree->storedPerPartitionLH    = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);

      localTree->fracchanges = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);
      localTree->partitionContributions = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);

      localTree->partitionData = (pInfo*)malloc(sizeof(pInfo) * localTree->NumberOfModels);

      /* extend for multi-branch */
      localTree->td[0].count = 0;
      localTree->td[0].ti    = (traversalInfo *)malloc(sizeof(traversalInfo) * localTree->mxtips);

      localTree->cdta               = (cruncheddata*)malloc(sizeof(cruncheddata));
      localTree->cdta->patrat       = (double*)malloc(sizeof(double) * localTree->originalCrunchedLength);
      localTree->cdta->patratStored = (double*)malloc(sizeof(double) * localTree->originalCrunchedLength);

      localTree->NumberOfCategories = tr->NumberOfCategories;

      localTree->discreteRateCategories = tr->discreteRateCategories;     

      for(model = 0; model < localTree->NumberOfModels; model++)
	{
	  localTree->partitionData[model].states     = tr->partitionData[model].states;
	  localTree->partitionData[model].maxTipStates    = tr->partitionData[model].maxTipStates;
	  localTree->partitionData[model].dataType   = tr->partitionData[model].dataType;
	  localTree->partitionData[model].protModels = tr->partitionData[model].protModels;
	  localTree->partitionData[model].protFreqs  = tr->partitionData[model].protFreqs;
	  localTree->partitionData[model].mxtips     = tr->partitionData[model].mxtips;
	  localTree->partitionData[model].lower      = tr->partitionData[model].lower;
	  localTree->partitionData[model].upper      = tr->partitionData[model].upper;
	  localTree->executeModel[model]             = TRUE;
	  localTree->perPartitionLH[model]           = 0.0;
	  localTree->storedPerPartitionLH[model]     = 0.0;
	  totalLength += (localTree->partitionData[model].upper -  localTree->partitionData[model].lower);
	}

      assert(totalLength == localTree->originalCrunchedLength);
    }

  for(model = 0; model < localTree->NumberOfModels; model++)
    localTree->partitionData[model].width        = 0;
}




static void allocNodex(tree *tr, int tid, int n)
{
  size_t   
    model,
    memoryRequirements = 0,
    myLength = 0;

  computeFraction(tr, tid, n);

  allocPartitions(tr);

  for(model = 0; model < (size_t)tr->NumberOfModels; model++)
    {
      size_t width = tr->partitionData[model].width;

      myLength += width;

      memoryRequirements += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;      
    }

  if(tid == 0)
    {
      tr->perSiteLL       = (double *)malloc((size_t)tr->cdta->endsite * sizeof(double));
      assert(tr->perSiteLL != NULL);
    }

  if(!tr->useFloat)
    {
      tr->likelihoodArray = (double *)malloc_aligned(tr->innerNodes * memoryRequirements * sizeof(double));
      assert(tr->likelihoodArray != NULL);
    }
  else
    {
      tr->likelihoodArray_FLOAT = (float *)malloc_aligned(tr->innerNodes * memoryRequirements * sizeof(float));
      assert(tr->likelihoodArray_FLOAT != NULL);      
    }

  if(!tr->useFastScaling)
    {
      tr->expArray = (int *)malloc(myLength * tr->innerNodes * sizeof(int));
      assert(tr->expArray != NULL);
    }

  if(!tr->useFloat)
    {
      tr->sumBuffer  = (double *)malloc_aligned(memoryRequirements * sizeof(double));
      assert(tr->sumBuffer != NULL);
    }
  else
    {
      tr->sumBuffer_FLOAT  = (float *)malloc_aligned(memoryRequirements * sizeof(float));
      assert(tr->sumBuffer_FLOAT != NULL);
    }

  tr->y_ptr = (unsigned char *)malloc(myLength * (size_t)(tr->mxtips) * sizeof(unsigned char));
  assert(tr->y_ptr != NULL);

  assert(4 * sizeof(double) > sizeof(parsimonyVector));

  tr->perSiteLLPtr     = (double*) malloc(myLength * sizeof(double));

  tr->wrPtr            = (double*) malloc(myLength * sizeof(double));
  assert(tr->wrPtr != NULL);

  tr->wr2Ptr           = (double*) malloc(myLength * sizeof(double));
  assert(tr->wr2Ptr != NULL);

  if(tr->useFloat)
    {
      tr->wrPtr_FLOAT            = (float*) malloc(myLength * sizeof(float));
      assert(tr->wrPtr_FLOAT != NULL);
      
      tr->wr2Ptr_FLOAT           = (float*) malloc(myLength * sizeof(float));
      assert(tr->wr2Ptr_FLOAT != NULL);
    }

  tr->wgtPtr           = (int*)    malloc(myLength * sizeof(int));
  assert(tr->wgtPtr != NULL);  

  tr->invariantPtr     = (int*)    malloc(myLength * sizeof(int));
  assert(tr->invariantPtr != NULL);

  tr->rateCategoryPtr  = (int*)    malloc(myLength * sizeof(int));
  assert(tr->rateCategoryPtr != NULL);
}








inline static void sendTraversalInfo(tree *localTree, tree *tr)
{
  /* the one below is a hack we are re-assigning the local pointer to the global one
     the memcpy version below is just for testing and preparing the
     fine-grained MPI BlueGene version */

  if(1)
    {
      localTree->td[0] = tr->td[0];
    }
  else
    {
      localTree->td[0].count = tr->td[0].count;
      memcpy(localTree->td[0].ti, tr->td[0].ti, localTree->td[0].count * sizeof(traversalInfo));
    }
}


static void collectDouble(double *dst, double *src, tree *tr, int n, int tid)
{
  int model, i;

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      for(i = tr->partitionData[model].lower; i < tr->partitionData[model].upper; i++)
	{
	  if(i % n == tid)
	    dst[i] = src[i];
	}
    }
}


static void execFunction(tree *tr, tree *localTree, int tid, int n)
{
  double volatile result;
  int
    i,
    currentJob,
    parsimonyResult,
    model,
    localCounter,
    globalCounter;

  currentJob = threadJob >> 16;

  switch(currentJob)
    {
      /* initialization only */
    case THREAD_INIT_PARTITION:
      initPartition(tr, localTree, tid);
      break;
    case THREAD_ALLOC_LIKELIHOOD:
      allocNodex(localTree, tid, n);
      threadFixModelIndices(tr, localTree, tid, n);
      break;
    case THREAD_FIX_MODEL_INDICES:
      threadFixModelIndices(tr, localTree, tid, n);
      break;
    case THREAD_EVALUATE_PARSIMONY:
      sendTraversalInfo(localTree, tr);
      parsimonyResult = evaluateParsimonyIterative(localTree);
      reductionBufferParsimony[tid] = parsimonyResult;
      break;
    case THREAD_NEWVIEW_PARSIMONY:
      sendTraversalInfo(localTree, tr);
      newviewParsimonyIterative(localTree);
      break;
    case THREAD_EVALUATE:
      sendTraversalInfo(localTree, tr);
      result = evaluateIterative(localTree, FALSE);

      if(localTree->NumberOfModels > 1)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    reductionBuffer[tid * localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
	}
      else
	reductionBuffer[tid] = result;

      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->executeModel[model] = TRUE;
	}
      break;
    case THREAD_NEWVIEW_MASKED:
      sendTraversalInfo(localTree, tr);
      memcpy(localTree->executeModel, tr->executeModel, sizeof(boolean) * localTree->NumberOfModels);
      newviewIterative(localTree);
      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->executeModel[model] = TRUE;
	}
      break;
    case THREAD_NEWVIEW:
      sendTraversalInfo(localTree, tr);
      newviewIterative(localTree);
      break;
    case THREAD_MAKENEWZ_FIRST:
      {
	volatile double
	  dlnLdlz[NUM_BRANCHES],
	  d2lnLdlz2[NUM_BRANCHES];

	sendTraversalInfo(localTree, tr);
	if(tid > 0)
	  {
	    memcpy(localTree->coreLZ,   tr->coreLZ,   sizeof(double) *  localTree->numBranches);
	    memcpy(localTree->executeModel, tr->executeModel, sizeof(boolean) * localTree->NumberOfModels);
	  }

	makenewzIterative(localTree);	
	execCore(localTree, dlnLdlz, d2lnLdlz2);

	if(!tr->multiBranch)
	  {
	    reductionBuffer[tid]    = dlnLdlz[0];
	    reductionBufferTwo[tid] = d2lnLdlz2[0];
	  }
	else
	  {
	    for(i = 0; i < localTree->NumberOfModels; i++)
	      {
		reductionBuffer[tid * localTree->NumberOfModels + i]    = dlnLdlz[i];
		reductionBufferTwo[tid * localTree->NumberOfModels + i] = d2lnLdlz2[i];
	      }
	  }

	if(tid > 0)
	  {
	    for(model = 0; model < localTree->NumberOfModels; model++)
	      localTree->executeModel[model] = TRUE;
	  }
      }
      break;
    case THREAD_MAKENEWZ:
      {
	volatile double
	  dlnLdlz[NUM_BRANCHES],
	  d2lnLdlz2[NUM_BRANCHES];

	memcpy(localTree->coreLZ,   tr->coreLZ,   sizeof(double) *  localTree->numBranches);
	memcpy(localTree->executeModel, tr->executeModel, sizeof(boolean) * localTree->NumberOfModels);
	
	execCore(localTree, dlnLdlz, d2lnLdlz2);

	if(!tr->multiBranch)
	  {
	    reductionBuffer[tid]    = dlnLdlz[0];
	    reductionBufferTwo[tid] = d2lnLdlz2[0];
	  }
	else
	  {
	    for(i = 0; i < localTree->NumberOfModels; i++)
	      {
		reductionBuffer[tid * localTree->NumberOfModels + i]    = dlnLdlz[i];
		reductionBufferTwo[tid * localTree->NumberOfModels + i] = d2lnLdlz2[i];
	      }
	  }
	if(tid > 0)
	  {
	    for(model = 0; model < localTree->NumberOfModels; model++)
	      localTree->executeModel[model] = TRUE;
	  }
      }
      break;
    case THREAD_COPY_RATES:
      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    {	      
	      const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	      
	      memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
	      memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	      memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	      
	      if(localTree->useFloat)
		{		     
		  memcpy(localTree->partitionData[model].EV_FLOAT,          tr->partitionData[model].EV_FLOAT,          pl->evLength * sizeof(float));
		  memcpy(localTree->partitionData[model].tipVector_FLOAT,   tr->partitionData[model].tipVector_FLOAT,   pl->tipVectorLength * sizeof(float));
		}	
	    }
	}
      break;
    case THREAD_OPT_RATE:
      if(tid > 0)
	{
	  memcpy(localTree->executeModel, tr->executeModel, localTree->NumberOfModels * sizeof(boolean));

	  for(model = 0; model < localTree->NumberOfModels; model++)
	    {
	      const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	      
	      memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
	      memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	      memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	      
	      if(localTree->useFloat)
		{		     
		  memcpy(localTree->partitionData[model].EV_FLOAT,          tr->partitionData[model].EV_FLOAT,          pl->evLength * sizeof(float));
		  memcpy(localTree->partitionData[model].tipVector_FLOAT,   tr->partitionData[model].tipVector_FLOAT,   pl->tipVectorLength * sizeof(float));
		}	     
	    }
	}

      result = evaluateIterative(localTree, FALSE);


      if(localTree->NumberOfModels > 1)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    reductionBuffer[tid * localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
	}
      else
	reductionBuffer[tid] = result;


      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->executeModel[model] = TRUE;
	}
      break;
    case THREAD_COPY_INVAR:
      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->partitionData[model].propInvariant = tr->partitionData[model].propInvariant;
	}
      break;
    case THREAD_OPT_INVAR:
      if(tid > 0)
	{
	  memcpy(localTree->executeModel, tr->executeModel, localTree->NumberOfModels * sizeof(boolean));
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->partitionData[model].propInvariant = tr->partitionData[model].propInvariant;
	}

      result = evaluateIterative(localTree, FALSE);

      if(localTree->NumberOfModels > 1)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    reductionBuffer[tid * localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
	}
      else
	reductionBuffer[tid] = result;

      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->executeModel[model] = TRUE;
	}
      break;
    case THREAD_COPY_ALPHA:
      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    {
	      memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
	      localTree->partitionData[model].alpha = tr->partitionData[model].alpha;
	    }
	}
      break;
    case THREAD_OPT_ALPHA:
      if(tid > 0)
	{
	  memcpy(localTree->executeModel, tr->executeModel, localTree->NumberOfModels * sizeof(boolean));
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
	}

      result = evaluateIterative(localTree, FALSE);


      if(localTree->NumberOfModels > 1)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    reductionBuffer[tid *  localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
	}
      else
	reductionBuffer[tid] = result;

      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->executeModel[model] = TRUE;
	}
      break;
    case THREAD_RESET_MODEL:
      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    {
	      const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

	      memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	      memcpy(localTree->partitionData[model].substRates,  tr->partitionData[model].substRates,  pl->substRatesLength * sizeof(double));
	      memcpy(localTree->partitionData[model].frequencies, tr->partitionData[model].frequencies, pl->frequenciesLength * sizeof(double));
	      memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	      
	      if(localTree->useFloat)
		{		     
		  memcpy(localTree->partitionData[model].EV_FLOAT,          tr->partitionData[model].EV_FLOAT,          pl->evLength * sizeof(float));
		  memcpy(localTree->partitionData[model].tipVector_FLOAT,   tr->partitionData[model].tipVector_FLOAT,   pl->tipVectorLength * sizeof(float));
		}	    	      	      

	      memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
	      localTree->partitionData[model].alpha = tr->partitionData[model].alpha;
	      localTree->partitionData[model].propInvariant = tr->partitionData[model].propInvariant;
	    }
	}
      break;
     

    case THREAD_COPY_INIT_MODEL:
      if(tid > 0)
	{
	  localTree->NumberOfCategories = tr->NumberOfCategories;
	  localTree->rateHetModel       = tr->rateHetModel;

	  for(model = 0; model < localTree->NumberOfModels; model++)
	    {
	      const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

	      memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	      memcpy(localTree->partitionData[model].substRates,  tr->partitionData[model].substRates,  pl->substRatesLength * sizeof(double));
	      memcpy(localTree->partitionData[model].frequencies, tr->partitionData[model].frequencies, pl->frequenciesLength * sizeof(double));
	      memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	      
	      if(localTree->useFloat)
		{		     
		  memcpy(localTree->partitionData[model].EV_FLOAT,          tr->partitionData[model].EV_FLOAT,          pl->evLength * sizeof(float));
		  memcpy(localTree->partitionData[model].tipVector_FLOAT,   tr->partitionData[model].tipVector_FLOAT,   pl->tipVectorLength * sizeof(float));
		}	       

	       memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
	       localTree->partitionData[model].alpha = tr->partitionData[model].alpha;
	       localTree->partitionData[model].propInvariant = tr->partitionData[model].propInvariant;
	       localTree->partitionData[model].lower      = tr->partitionData[model].lower;
	       localTree->partitionData[model].upper      = tr->partitionData[model].upper;
	    }

	  memcpy(localTree->cdta->patrat,        tr->cdta->patrat,      localTree->originalCrunchedLength * sizeof(double));
	  memcpy(localTree->cdta->patratStored, tr->cdta->patratStored, localTree->originalCrunchedLength * sizeof(double));
	}     

       for(model = 0; model < localTree->NumberOfModels; model++)
	 {
	   int localIndex;
	   for(i = localTree->partitionData[model].lower, localIndex = 0; i <  localTree->partitionData[model].upper; i++)
	     if(i % n == tid)
	       {
		 localTree->partitionData[model].wgt[localIndex]          = tr->cdta->aliaswgt[i];
		 localTree->partitionData[model].wr[localIndex]           = tr->cdta->wr[i];
		 localTree->partitionData[model].wr2[localIndex]          = tr->cdta->wr2[i];
		 
		 if(localTree->useFloat)
		   {
		     localTree->partitionData[model].wr_FLOAT[localIndex]           = tr->cdta->wr_FLOAT[i];
		     localTree->partitionData[model].wr2_FLOAT[localIndex]          = tr->cdta->wr2_FLOAT[i]; 
		   }

		 localTree->partitionData[model].invariant[localIndex]    = tr->invariant[i];
		 localTree->partitionData[model].rateCategory[localIndex] = tr->cdta->rateCategory[i];
		 localIndex++;
	       }	  
	 }

      break;

    case THREAD_PARSIMONY_RATCHET:
      for(model = 0; model < localTree->NumberOfModels; model++)
	{
	  int localIndex;
	  for(i = localTree->partitionData[model].lower, localIndex = 0; i <  localTree->partitionData[model].upper; i++)
	    if(i % n == tid)
	      {
		localTree->partitionData[model].wgt[localIndex]          = tr->cdta->aliaswgt[i];
		localIndex++;
	      }
	}
      break;
    case THREAD_RATE_CATS:
      sendTraversalInfo(localTree, tr);
      if(tid > 0)
	{
	  localTree->lower_spacing = tr->lower_spacing;
	  localTree->upper_spacing = tr->upper_spacing;
	}

      optRateCatPthreads(localTree, localTree->lower_spacing, localTree->upper_spacing, localTree->lhs, n, tid);

      if(tid > 0)
	{
	  collectDouble(tr->cdta->patrat,       localTree->cdta->patrat,         localTree, n, tid);
	  collectDouble(tr->cdta->patratStored, localTree->cdta->patratStored,   localTree, n, tid);
	  collectDouble(tr->lhs,                localTree->lhs,                  localTree, n, tid);
	}
      break;
    case THREAD_COPY_RATE_CATS:
      if(tid > 0)
	{
	  localTree->NumberOfCategories = tr->NumberOfCategories;
	  memcpy(localTree->cdta->patrat,       tr->cdta->patrat,         localTree->originalCrunchedLength * sizeof(double));
	  memcpy(localTree->cdta->patratStored, tr->cdta->patratStored,   localTree->originalCrunchedLength * sizeof(double));
	}


      for(model = 0; model < localTree->NumberOfModels; model++)
	{
	  for(localCounter = 0, i = localTree->partitionData[model].lower;  i < localTree->partitionData[model].upper; i++)
	    {
	      if(i % n == tid)
		{
		  localTree->partitionData[model].wr[localCounter]           = tr->cdta->wr[i];
		  localTree->partitionData[model].wr2[localCounter]          = tr->cdta->wr2[i];

		  if(localTree->useFloat)
		   {
		     localTree->partitionData[model].wr_FLOAT[localCounter]           = tr->cdta->wr_FLOAT[i];
		     localTree->partitionData[model].wr2_FLOAT[localCounter]          = tr->cdta->wr2_FLOAT[i]; 
		   }

		  localTree->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[i];
		  localCounter++;
		}
	    }
	}
      break;
    case THREAD_CAT_TO_GAMMA:
      if(tid > 0)
	localTree->rateHetModel = tr->rateHetModel;
      break;
    case THREAD_GAMMA_TO_CAT:
      if(tid > 0)
	localTree->rateHetModel = tr->rateHetModel;
      break;
    case THREAD_EVALUATE_VECTOR:
      sendTraversalInfo(localTree, tr);
      evaluateIterative(localTree, TRUE);

      if(localTree->NumberOfModels > 1)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    reductionBuffer[tid * localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
	}
      else
	reductionBuffer[tid] = result;

      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    localTree->executeModel[model] = TRUE;
	}

      for(model = 0, globalCounter = 0; model < localTree->NumberOfModels; model++)
	{
	  for(localCounter = 0, i = localTree->partitionData[model].lower;  i < localTree->partitionData[model].upper; i++)
	    {
	      if(i % n == tid)
		{
		  tr->perSiteLL[globalCounter] =  localTree->partitionData[model].perSiteLL[localCounter];
		  localCounter++;
		}
	      globalCounter++;
	    }
	}
      break;    
    case THREAD_COPY_PARAMS:
      if(tid > 0)
	{
	  for(model = 0; model < localTree->NumberOfModels; model++)
	    {
	      const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	      
	      memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));
	      memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	      memcpy(localTree->partitionData[model].substRates,  tr->partitionData[model].substRates,  pl->substRatesLength * sizeof(double));
	      memcpy(localTree->partitionData[model].frequencies, tr->partitionData[model].frequencies, pl->frequenciesLength * sizeof(double));
	      memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	      
	      if(localTree->useFloat)
		{		    
		  memcpy(localTree->partitionData[model].EV_FLOAT,          tr->partitionData[model].EV_FLOAT,          pl->evLength * sizeof(float));
		  memcpy(localTree->partitionData[model].tipVector_FLOAT,   tr->partitionData[model].tipVector_FLOAT,   pl->tipVectorLength * sizeof(float));
		}	     
	    }
	}
      break;
    case THREAD_INIT_EPA:     
      if(tid > 0)
	{
	  localTree->bInf                     = tr->bInf;	 
	  localTree->numberOfBranches         = tr->numberOfBranches;
	  localTree->contiguousVectorLength   = tr->contiguousVectorLength;
	  localTree->contiguousScalingLength  = tr->contiguousScalingLength;
	  localTree->inserts                  = tr->inserts;
	  localTree->numberOfTipsForInsertion = tr->numberOfTipsForInsertion;	
	  localTree->fracchange = tr->fracchange;
	  memcpy(localTree->partitionContributions, tr->partitionContributions, sizeof(double) * localTree->NumberOfModels);
	  memcpy(localTree->fracchanges, tr->fracchanges, sizeof(double) * localTree->NumberOfModels);
	}                                                

      localTree->temporarySumBuffer = (double *)malloc_aligned(sizeof(double) * localTree->contiguousVectorLength);
      localTree->temporaryVector  = (double *)malloc_aligned(sizeof(double) * localTree->contiguousVectorLength);
      localTree->temporaryParsimonyVector = (parsimonyVector*)localTree->temporaryVector;

      localTree->temporaryScaling = (int *)malloc(sizeof(int) * localTree->contiguousScalingLength);
                 
      localTree->contiguousRateCategory = (int*)malloc(sizeof(int) * localTree->contiguousScalingLength);
      localTree->contiguousWgt          = (int*)malloc(sizeof(int) * localTree->contiguousScalingLength);
      localTree->contiguousInvariant    = (int*)malloc(sizeof(int) * localTree->contiguousScalingLength);	  
      
      memcpy(localTree->contiguousRateCategory, tr->cdta->rateCategory, sizeof(int) * localTree->contiguousScalingLength);
      memcpy(localTree->contiguousWgt         , tr->cdta->aliaswgt,     sizeof(int) * localTree->contiguousScalingLength);
      memcpy(localTree->contiguousInvariant   , tr->invariant,          sizeof(int) * localTree->contiguousScalingLength);
      
      localTree->contiguousWR   = (double*)malloc(sizeof(double) * localTree->contiguousScalingLength);
      localTree->contiguousWR2  = (double*)malloc(sizeof(double) * localTree->contiguousScalingLength);
      localTree->contiguousPATRAT  = (double*)malloc(sizeof(double) * localTree->contiguousScalingLength);
      
      memcpy(localTree->contiguousWR, tr->cdta->wr, sizeof(double) * localTree->contiguousScalingLength);
      memcpy(localTree->contiguousWR2, tr->cdta->wr2, sizeof(double) * localTree->contiguousScalingLength);     
      memcpy(localTree->contiguousPATRAT, tr->cdta->patrat, sizeof(double) * localTree->contiguousScalingLength);
      
     
      localTree->contiguousTips = tr->yVector;	  	
	 
      break;       
    case THREAD_GATHER_PARSIMONY:
      {	
	int 
	  branchCounter = tr->branchCounter;

	parsimonyVector
	  *leftContigousVector = localTree->bInf[branchCounter].epa->leftParsimony,
	  *rightContigousVector = localTree->bInf[branchCounter].epa->rightParsimony;
      
	int	 		  
	  globalCount       = 0,
	  rightNumber = localTree->bInf[branchCounter].epa->rightNodeNumber,
	  leftNumber  = localTree->bInf[branchCounter].epa->leftNodeNumber;	

	for(model = 0; model < localTree->NumberOfModels; model++)
	  {	    
	    parsimonyVector
	      *leftStridedVector  =  (parsimonyVector *)NULL,
	      *rightStridedVector =  (parsimonyVector *)NULL;

	    int	      	     	      
	      localCount = 0;	   	    	   

	    if(!isTip(leftNumber, localTree->mxtips))	      
	      leftStridedVector        = localTree->partitionData[model].pVector[leftNumber - localTree->mxtips - 1];	       	    
	   
	    if(!isTip(rightNumber, localTree->mxtips))	      
	      rightStridedVector        = localTree->partitionData[model].pVector[rightNumber - localTree->mxtips - 1];	   
	   
	    assert(!(isTip(leftNumber, localTree->mxtips) && isTip(rightNumber, localTree->mxtips)));	   	    	    

	    for(globalCount = localTree->partitionData[model].lower; globalCount < localTree->partitionData[model].upper; globalCount++)
	      {	
	
		if(globalCount % n == tid)
		  {		    		   
		    if(leftStridedVector)
		      memcpy(&leftContigousVector[globalCount], 
			     &leftStridedVector[localCount], sizeof(parsimonyVector));		      		    
		    
		    if(rightStridedVector)		      
		      memcpy(&rightContigousVector[globalCount], 
			     &rightStridedVector[localCount], sizeof(parsimonyVector));		  
		   
		    localCount++;		    
		  }	       
	      }	    	    
	    
	  }
      }
      break;       
    case THREAD_GATHER_LIKELIHOOD:
      {	
	int 
	  branchCounter = tr->branchCounter;

	double
	  *leftContigousVector = localTree->bInf[branchCounter].epa->left,
	  *rightContigousVector = localTree->bInf[branchCounter].epa->right;
      
	int
	  *leftContigousScalingVector = localTree->bInf[branchCounter].epa->leftScaling,
	  *rightContigousScalingVector = localTree->bInf[branchCounter].epa->rightScaling,	
	  globalColumnCount = 0,
	  globalCount       = 0,
	  rightNumber = localTree->bInf[branchCounter].epa->rightNodeNumber,
	  leftNumber  = localTree->bInf[branchCounter].epa->leftNodeNumber;	


	for(model = 0; model < localTree->NumberOfModels; model++)
	  {
	    size_t
	      blockRequirements;

	    double
	      *leftStridedVector  =  (double *)NULL,
	      *rightStridedVector =  (double *)NULL;

	    int
	      *leftStridedScalingVector  =  (int *)NULL,
	      *rightStridedScalingVector =  (int *)NULL,
	     
	      localColumnCount = 0,
	      localCount = 0;	   

	    if(!isTip(leftNumber, localTree->mxtips))
	      {
		leftStridedVector        = localTree->partitionData[model].xVector[leftNumber - localTree->mxtips - 1];
		leftStridedScalingVector = localTree->partitionData[model].expVector[leftNumber - localTree->mxtips - 1];
	      }	   

	    if(!isTip(rightNumber, localTree->mxtips))
	      {
		rightStridedVector        = localTree->partitionData[model].xVector[rightNumber - localTree->mxtips - 1];
		rightStridedScalingVector = localTree->partitionData[model].expVector[rightNumber - localTree->mxtips - 1];
	      }	    

	    assert(!(isTip(leftNumber, localTree->mxtips) && isTip(rightNumber, localTree->mxtips)));	   

	    blockRequirements = (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states);	   

	    for(globalColumnCount = localTree->partitionData[model].lower; globalColumnCount < localTree->partitionData[model].upper; globalColumnCount++)
	      {	
		if(globalColumnCount % n == tid)
		  {		    
		    if(leftStridedVector)
		      {
			memcpy(&leftContigousVector[globalCount], &leftStridedVector[localCount], sizeof(double) * blockRequirements);
			leftContigousScalingVector[globalColumnCount] = leftStridedScalingVector[localColumnCount];
		      }
		    
		    if(rightStridedVector)
		      {
			memcpy(&rightContigousVector[globalCount], &rightStridedVector[localCount], sizeof(double) * blockRequirements);
			rightContigousScalingVector[globalColumnCount] = rightStridedScalingVector[localColumnCount];
		      }
		   
		    localColumnCount++;
		    localCount += blockRequirements;
		  }

	

		globalCount += blockRequirements;
	      }	    

	    assert(localColumnCount == localTree->partitionData[model].width);
	    assert(localCount == (localTree->partitionData[model].width * (int)blockRequirements));

	  }
      }
      break;    
    case THREAD_PREPARE_EPA_PARSIMONY:              	             
      memcpy(localTree->contiguousWgt         , tr->cdta->aliaswgt,     sizeof(int) * localTree->contiguousScalingLength);                                 	  	           
      break;
    case THREAD_CLEANUP_EPA_PARSIMONY:
      memcpy(localTree->contiguousWgt         , tr->cdta->aliaswgt,     sizeof(int) * localTree->contiguousScalingLength);                            	  	           
      break;
    case THREAD_CONTIGUOUS_REPLICATE:      	        
      memcpy(localTree->contiguousRateCategory, tr->cdta->rateCategory, sizeof(int) * localTree->contiguousScalingLength);
      memcpy(localTree->contiguousWgt         , tr->cdta->aliaswgt,     sizeof(int) * localTree->contiguousScalingLength);
      memcpy(localTree->contiguousInvariant   , tr->invariant,          sizeof(int) * localTree->contiguousScalingLength);            
      
      memcpy(localTree->contiguousWR, tr->cdta->wr, sizeof(double) * localTree->contiguousScalingLength);
      memcpy(localTree->contiguousWR2, tr->cdta->wr2, sizeof(double) * localTree->contiguousScalingLength);     
      memcpy(localTree->contiguousPATRAT, tr->cdta->patrat, sizeof(double) * localTree->contiguousScalingLength);                	  	
     
      break;        
    case THREAD_INSERT_CLASSIFY:          
    case THREAD_INSERT_CLASSIFY_THOROUGH:     
    case THREAD_PARSIMONY_INSERTIONS:    
    case THREAD_INSERT_CLASSIFY_THOROUGH_BS:
      { 
	int
	  branchNumber;
	
	boolean 
	  done = FALSE;	       

	while(!done)
	  {	      	      	    		    
	    pthread_mutex_lock(&mutex);
	      
	    if(NumberOfJobs == 0)
	      done = TRUE;
	    else
	      {		  
		branchNumber = localTree->numberOfBranches - NumberOfJobs;		 
		NumberOfJobs--;		 
	      }
	      	   
	    pthread_mutex_unlock(&mutex);
	      
	    if(!done)
	      {		 		 		 		 	 		      
		switch(currentJob)
		  {
		  case THREAD_INSERT_CLASSIFY:
		    addTraverseRobIterative(localTree, branchNumber);
		    break;		  
		  case  THREAD_INSERT_CLASSIFY_THOROUGH:		    
		    testInsertThoroughIterative(localTree, branchNumber, FALSE);		   
		    break;   
		  case THREAD_PARSIMONY_INSERTIONS:
		    insertionsParsimonyIterative(localTree, branchNumber);
		    break; 
		  case THREAD_INSERT_CLASSIFY_THOROUGH_BS:
		    testInsertThoroughIterative(localTree, branchNumber, TRUE);
		    break;
		  default:
		    assert(0);
		  }
		  		  		
	      }	    
	    }
      }
      break;
    case THREAD_DRAW_BIPARTITIONS:
      {
	int 
	  i;          

	if(tid > 0)
	  {
	    int 
	      tips  = localTree->mxtips,
	      inter = localTree->mxtips - 1;

	    nodeptr p0, p, q;	   

	    localTree->nameList = tr->nameList;
	    localTree->nameHash = tr->nameHash;	   
	    localTree->h        = tr->h;

	    localTree->outgroups = tr->outgroups;
	    localTree->outgroupNums = tr->outgroupNums;
	    localTree->grouped      = tr->grouped;
	    localTree->constraintVector = tr->constraintVector;
	    
	    localTree->numberOfTrees = tr->numberOfTrees;	
	    localTree->treeStarts    = tr->treeStarts;	    	    
	    
	    localTree->nodep_scratch = (nodeptr) malloc((tips + 3 * inter) * sizeof(node));
	    p0 = localTree->nodep_scratch;
	    assert(p0);
	    
	    localTree->nodep = (nodeptr *)malloc((2 * localTree->mxtips) * sizeof(nodeptr));
	    assert(localTree->nodep);
	    
	    localTree->nodep[0] = (node *) NULL;

	    for(i = 1; i <= tips; i++)
	      {			
		p = p0++;

		p->hash   =  tr->nodep[i]->hash;
		p->x      =  0;
		p->number =  i;
		
		p->next   =  p;
		p->back   = (node *)NULL;
		p->bInf   = (branchInfo *)NULL;								
		
		localTree->nodep[i] = p;
	      }

	    for(i = tips + 1; i <= tips + inter; i++)
	      {
		int j;

		q = (node *) NULL;
		for (j = 1; j <= 3; j++)
		  {	 
		    p = p0++;
		    if(j == 1)
		      p->x = 1;
		    else
		      p->x =  0;
		    p->number = i;
		    p->next   = q;
		    
		    p->bInf   = (branchInfo *)NULL;
		    
		    p->hash   = 0;

		    
		    q = p;
		  }
		
		p->next->next->next = p;
		localTree->nodep[i] = p;
	      }
	  }

	localTree->bitVectors = initBitVector(localTree, &(localTree->bitVectorLength));
	localTree->threadBranchInfo =  (branchInfo*)malloc(sizeof(branchInfo) * (localTree->mxtips - 3));

	/*
	  if(tid > 0)
	  localTree->h = copyHashTable(tr->h, localTree->bitVectorLength);
	*/
	
	branchInfos[tid] = localTree->threadBranchInfo;

	for(i = 0; i < (localTree->mxtips - 3); i++)
	  localTree->threadBranchInfo[i].support = 0;		
	
	

	{       	 
	  boolean done = FALSE;	
	 
	  while(!done)
	    {	    
	      	      	    		    
	      pthread_mutex_lock(&mutex);
	      
	      if(NumberOfJobs == 0)
		done = TRUE;
	      else
		{		  
		  i = localTree->numberOfTrees - NumberOfJobs;		 
		  NumberOfJobs--;		 
		}
	      
	      pthread_mutex_unlock(&mutex);
	     

	      if(!done)
		{		 		 		 		 	 		      
		  int 
		    bCount = 0,
		    position = 0;
		 
		  treeReadTopologyString(localTree->treeStarts[i], localTree, FALSE, FALSE, &position, TRUE, (analdef*)NULL);
		  
		  assert(localTree->ntips == localTree->mxtips);
		  		 
		  bitVectorInitravSpecial(localTree->bitVectors, localTree->nodep[1]->back, localTree->mxtips, 
					  localTree->bitVectorLength, localTree->h, 0, DRAW_BIPARTITIONS_BEST, localTree->threadBranchInfo, &bCount, 0, FALSE, FALSE);
		  
		}
	      
	    }
	}		
      }
      break;
    case THREAD_FREE_DRAW_BIPARTITIONS:
      free(localTree->threadBranchInfo);
      freeBitVectors(localTree->bitVectors, localTree->bitVectorLength);
      if(tid > 0)
	{
	  free(localTree->nodep);
	  free(localTree->nodep_scratch);
	}
      break;
      /*
	Pthreads parallelization doesn't really scale with the new optimizations ....

	case THREAD_INIT_FAST_PARSIMONY:
	if(tid > 0)
	{
	localTree->ti = tr->ti;
	localTree->parsimonyScore = (unsigned int*)malloc_aligned(sizeof(unsigned int) * 2 * localTree->mxtips);
	
	localTree->parsimonyState_A = tr->parsimonyState_A;
	localTree->parsimonyState_C = tr->parsimonyState_C;
	localTree->parsimonyState_G = tr->parsimonyState_G;
	localTree->parsimonyState_T = tr->parsimonyState_T;
	
	localTree->compressedWidth = tr->compressedWidth;	  
	}
	break;
	case  THREAD_FAST_NEWVIEW_PARSIMONY:
	newviewParsimonyIterativeFast(localTree);
	break;
	case THREAD_FAST_EVALUATE_PARSIMONY: 
	parsimonyResult = evaluateParsimonyIterativeFast(localTree);
	reductionBufferParsimony[tid] = parsimonyResult;
	break;
      */
    default:
      printf("Job %d\n", currentJob);
      assert(0);
    }
}



void masterBarrier(int jobType, tree *tr)
{
  const int n = NumberOfThreads;
  int i, sum;

  jobCycle = !jobCycle;
  threadJob = (jobType << 16) + jobCycle;

  execFunction(tr, tr, 0, n);

  do
    {
      for(i = 1, sum = 1; i < n; i++)
	sum += barrierBuffer[i];
    }
  while(sum < n);

  for(i = 1; i < n; i++)
    barrierBuffer[i] = 0;
}




static void *likelihoodThread(void *tData)
{
  threadData *td = (threadData*)tData;
  tree
    *tr = td->tr,
    *localTree = (tree *)malloc(sizeof(tree));
  int
    myCycle = 0;

  const int n = NumberOfThreads;
  const int tid             = td->threadNumber;

  printf("\nThis is RAxML Worker Pthread Number: %d\n", tid);

  while(1)
    {
      while (myCycle == threadJob);
      myCycle = threadJob;
      //printf("\nthread: %d, Job: %d", tid, myCycle);
      execFunction(tr, localTree, tid, n);

      barrierBuffer[tid] = 1;
    }

  return (void*)NULL;
}

static void startPthreads(tree *tr)
{
  pthread_t *threads;
  pthread_attr_t attr;
  int rc, t;
  threadData *tData;

  jobCycle        = 0;
  threadJob       = 0;

  printf("\nThis is the RAxML Master Pthread\n");

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  pthread_mutex_init(&mutex , (pthread_mutexattr_t *)NULL);

  threads    = (pthread_t *)malloc(NumberOfThreads * sizeof(pthread_t));
  tData      = (threadData *)malloc(NumberOfThreads * sizeof(threadData));
  reductionBuffer          = (double *)malloc(sizeof(double) *  NumberOfThreads * tr->NumberOfModels);
  reductionBufferTwo       = (double *)malloc(sizeof(double) *  NumberOfThreads * tr->NumberOfModels);
  reductionBufferThree     = (double *)malloc(sizeof(double) *  NumberOfThreads * tr->NumberOfModels);
  reductionBufferParsimony = (int *)malloc(sizeof(int) *  NumberOfThreads);
  barrierBuffer            = (int *)malloc(sizeof(int) *  NumberOfThreads);
 
  branchInfos              = (volatile branchInfo **)malloc(sizeof(volatile branchInfo *) * NumberOfThreads);

  for(t = 0; t < NumberOfThreads; t++)
    barrierBuffer[t] = 0;

  for(t = 1; t < NumberOfThreads; t++)
    {
      tData[t].tr  = tr;
      tData[t].threadNumber = t;
      rc = pthread_create(&threads[t], &attr, likelihoodThread, (void *)(&tData[t]));
      if(rc)
	{
	  printf("ERROR; return code from pthread_create() is %d\n", rc);
	  exit(-1);
	}
    }
}



#endif


/*************************************************************************************************************************************************************/

static int elwCompare(const void *p1, const void *p2)
{
  elw *rc1 = (elw *)p1;
  elw *rc2 = (elw *)p2;

  double i = rc1->weight;
  double j = rc2->weight;

  if (i > j)
    return (-1);
  if (i < j)
    return (1);
  return (0);
}

static int elwCompareLikelihood(const void *p1, const void *p2)
{
  elw *rc1 = (elw *)p1;
  elw *rc2 = (elw *)p2;

  double i = rc1->lh;
  double j = rc2->lh;

  if (i > j)
    return (-1);
  if (i < j)
    return (1);
  return (0);
}

static void computeLHTest(tree *tr, analdef *adef, char *bootStrapFileName)
{
  int
    numberOfTrees = 0,
    i;
  
  double 
    bestLH, 
    currentLH, 
    weightSum = 0.0;
  
  double 
    *bestVector = (double*)malloc(sizeof(double) * tr->cdta->endsite);

  for(i = 0; i < tr->cdta->endsite; i++)
    weightSum += (double)(tr->cdta->aliaswgt[i]);

  modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
  printBothOpen("Model optimization, best Tree: %f\n", tr->likelihood);
  bestLH = tr->likelihood;

  evaluateGenericInitrav(tr, tr->start);

  evaluateGenericVector(tr, tr->start);
  memcpy(bestVector, tr->perSiteLL, tr->cdta->endsite * sizeof(double));

  INFILE = myfopen(bootStrapFileName, "r");
  numberOfTrees = countTrees(INFILE);

  printBothOpen("Found %d trees in File %s\n", numberOfTrees, bootStrapFileName);

  for(i = 0; i < numberOfTrees; i++)
    {
      int 
	j;
	 
      double 
	temp, 
	wtemp, 
	sum = 0.0, 
	sum2 = 0.0, 
	sd;
      
      treeReadLen(INFILE, tr, adef);
      treeEvaluate(tr, 2);
      tr->start = tr->nodep[1];

      evaluateGenericInitrav(tr, tr->start);

      currentLH = tr->likelihood;
      if(currentLH > bestLH)	
	printBothOpen("Better tree found %d at %f\n", i, currentLH);	 

      evaluateGenericVector(tr, tr->start);         

      sum = 0.0;
      sum2 = 0.0;

      for (j = 0; j < tr->cdta->endsite; j++)
	{
	  temp  = bestVector[j] - tr->perSiteLL[j];
	  wtemp = tr->cdta->aliaswgt[j] * temp;
	  sum  += wtemp;
	  sum2 += wtemp * temp;
	}

      sd = sqrt( weightSum * (sum2 - sum*sum / weightSum) / (weightSum - 1) );
      /* this is for a 5% p level */
	 
      printBothOpen("Tree: %d Likelihood: %f D(LH): %f SD: %f Significantly Worse: %s (5%s), %s (2%s), %s (1%s)\n", 
		    i, currentLH, currentLH - bestLH, sd, 
		    (sum > 1.95996 * sd) ? "Yes" : " No", "%",
		    (sum > 2.326 * sd) ? "Yes" : " No", "%",
		    (sum > 2.57583 * sd) ? "Yes" : " No", "%");	       
    }

  fclose(INFILE);
  free(bestVector);

  exit(0);
}

static void computePerSiteLLs(tree *tr, analdef *adef, char *bootStrapFileName)
{
  int
    numberOfTrees = 0,
    i;
  
  FILE 
    *tlf = myfopen(perSiteLLsFileName, "w");

  double 
    *unsortedSites = (double*)malloc(sizeof(double) * tr->rdta->sites);

  INFILE = myfopen(bootStrapFileName, "r");
  numberOfTrees = countTrees(INFILE);

  printBothOpen("Found %d trees in File %s\n", numberOfTrees, bootStrapFileName);

  fprintf(tlf, "  %d  %d\n", numberOfTrees, tr->rdta->sites);

  for(i = 0; i < numberOfTrees; i++)
    {      
      int 
	k, 
	j;

      treeReadLen(INFILE, tr, adef);
      
      if(i == 0)
	modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
      else
	treeEvaluate(tr, 2);

      printBothOpen("Tree %d: %f\n", i, tr->likelihood);

      tr->start = tr->nodep[1];

      evaluateGenericInitrav(tr, tr->start);

      evaluateGenericVector(tr, tr->start);

      fprintf(tlf, "tr%d\t", i + 1);     

      for(j = 0; j < tr->cdta->endsite; j++)
	{
	  for(k = 0; k < tr->rdta->sites; k++)
	    {
	      if(j == tr->patternPosition[k])
		unsortedSites[tr->columnPosition[k] - 1] = tr->perSiteLL[j];
	    }
	}

      for(j = 0; j < tr->rdta->sites; j++)	  
	fprintf(tlf, "%f ", unsortedSites[j]);	   	             
     
      fprintf(tlf, "\n");
    }

  free(unsortedSites);
  fclose(INFILE);
  fclose(tlf);  
}


static void computeAllLHs(tree *tr, analdef *adef, char *bootStrapFileName)
{
  int
    numberOfTrees = 0,
    i;
  double
    bestLH = unlikely;
  
  bestlist 
    *bestT;
  
  FILE 
    *result = myfopen(resultFileName, "w");

  elw 
    *list;

  bestT = (bestlist *) malloc(sizeof(bestlist));
  bestT->ninit = 0;
  initBestTree(bestT, 1, tr->mxtips);

  INFILE = myfopen(bootStrapFileName, "r");
  numberOfTrees = countTrees(INFILE);

  list = (elw *)malloc(sizeof(elw) * numberOfTrees);

  printBothOpen("\n\nFound %d trees in File %s\n\n", numberOfTrees, bootStrapFileName);

  for(i = 0; i < numberOfTrees; i++)
    {
      treeReadLen(INFILE, tr, adef);

      if(i == 0)
	{
	  modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
	  printBothOpen("Model optimization on first Tree: %f\n", tr->likelihood);	  
	  bestLH = tr->likelihood;
	  resetBranches(tr);
	}
      
      treeEvaluate(tr, 2);      

      list[i].tree = i;
      list[i].lh   = tr->likelihood;

      Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE,
		  TRUE, adef, SUMMARIZE_LH, FALSE);

      fprintf(result, "%s", tr->tree_string);

      saveBestTree(bestT, tr);

      if(tr->likelihood > bestLH)
	bestLH   = tr->likelihood;
      
      printBothOpen("Tree %d Likelihood %f\n", i, tr->likelihood);    
    }

  qsort(list, numberOfTrees, sizeof(elw), elwCompareLikelihood);

  printBothOpen("\n");
  for(i = 0; i < numberOfTrees; i++)
    printBothOpen("%d %f\n", list[i].tree, list[i].lh);

  printBothOpen("\n");

 
 
  /*
    recallBestTree(bestT, 1, tr);
    evaluateGeneric(tr, tr->start);
    printf("Model optimization, %f <-> %f\n", bestLH, tr->likelihood);
    fprintf(infoFile, "Model optimization, %f <-> %f\n", bestLH, tr->likelihood);
    modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
    treeEvaluate(tr, 2);
    printf("Model optimization, %f <-> %f\n", bestLH, tr->likelihood);
    fprintf(infoFile, "Model optimization, %f <-> %f\n", bestLH, tr->likelihood);
  */

  printBothOpen("\nAll evaluated trees with branch lengths written to File: %s\n", resultFileName);
  printBothOpen("\nTotal execution time: %f\n", gettime() - masterTime);

  fclose(INFILE);
  fclose(result);
  exit(0);
}




static void computeELW(tree *tr, analdef *adef, char *bootStrapFileName)
{
  int
    cutOff95 = -1,
    cutOff99 = -1,
    bestIndex = -1,
    numberOfTrees = 0,
    i,
    k,
    *originalRateCategories = (int*)malloc(tr->cdta->endsite * sizeof(int)),
    *originalInvariant      = (int*)malloc(tr->cdta->endsite * sizeof(int)),
    *countBest;

  long 
    startSeed;

  double
    best = unlikely,
    **lhs,
    **lhweights,
    sum = 0.0;

  elw
    *bootweights,
    **rankTest;

  initModel(tr, tr->rdta, tr->cdta, adef); 

  INFILE = myfopen(bootStrapFileName, "r");

  numberOfTrees = countTrees(INFILE);

  if(numberOfTrees < 2)
    {
      printBothOpen("Error, there is only one tree in file %s which you want to use to conduct an ELW test\n", bootStrapFileName);

      exit(-1);
    }

  printBothOpen("\n\nFound %d trees in File %s\n\n", numberOfTrees, bootStrapFileName);

  bootweights = (elw *)malloc(sizeof(elw) * numberOfTrees);

  rankTest = (elw **)malloc(sizeof(elw *) * adef->multipleRuns);

  for(k = 0; k < adef->multipleRuns; k++)
    rankTest[k] = (elw *)malloc(sizeof(elw) * numberOfTrees);

  lhs = (double **)malloc(sizeof(double *) * numberOfTrees);

  for(k = 0; k < numberOfTrees; k++)
    lhs[k] = (double *)calloc(adef->multipleRuns, sizeof(double));


  lhweights = (double **)malloc(sizeof(double *) * numberOfTrees);

  for(k = 0; k < numberOfTrees; k++)
    lhweights[k] = (double *)calloc(adef->multipleRuns, sizeof(double));

  countBest = (int*)calloc(adef->multipleRuns, sizeof(int));

  /* read in the first tree and optimize ML params on it */

  treeReadLen(INFILE, tr, adef);
  modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
  rewind(INFILE);

  printBothOpen("Model optimization, first Tree: %f\n", tr->likelihood);

  memcpy(originalRateCategories, tr->cdta->rateCategory, sizeof(int) * tr->cdta->endsite);
  memcpy(originalInvariant,      tr->invariant,          sizeof(int) * tr->cdta->endsite);

  assert(adef->boot > 0);

  /* TODO this is ugly, should be passed as param to computenextreplicate() */

  startSeed = adef->boot;


  /*
     now read the trees one by one, do a couple of BS replicates and re-compute their likelihood
     for every replicate
  */

  /* loop over all trees */

  for(i = 0; i < numberOfTrees; i++)
    {
      /* read in new tree */

      treeReadLen(INFILE, tr, adef);
      treeEvaluate(tr, 2.0);
      printBothOpen("Original tree %d likelihood %f\n", i, tr->likelihood);

      if(tr->likelihood > best)
	{
	  best      = tr->likelihood;
	  bestIndex = i;
	}
      /* reset branches to default values */

      resetBranches(tr);

      /* reset BS random seed, we want to use the same replicates for every tree */

      adef->rapidBoot = startSeed;

      for(k = 0; k < adef->multipleRuns; k++)
	{
	  /* compute the next BS replicate, i.e., re-sample alignment columns */

	  computeNextReplicate(tr, &adef->rapidBoot, originalRateCategories, originalInvariant, TRUE);

	  /* if this is the first replicate for this tree do a slightly more thorough br-len opt */
	  /* we don't re-estimate ML model params (except branches) for every replicate to make things a bit faster */

	  if(k == 0)
	    treeEvaluate(tr, 2.0);
	  else
	    treeEvaluate(tr, 0.5);	  

	  /* store the likelihood of replicate k for tree i */
	  lhs[i][k] = tr->likelihood;

	  rankTest[k][i].lh   = tr->likelihood;
	  rankTest[k][i].tree = i;
	}

      /* restore the original alignment to start BS procedure for the next tree */

      reductionCleanup(tr, originalRateCategories, originalInvariant);
    }

  assert(bestIndex >= 0 && best != unlikely);

  printBothOpen("Best-Scoring tree is tree %d with score %f\n", bestIndex, best);


  /* now loop over all replicates */

  for(k = 0; k < adef->multipleRuns; k++)
    {
      /* find best score for this replicate */

      for(i = 0, best = unlikely; i < numberOfTrees; i++)
	if(lhs[i][k] > best)
	  best = lhs[i][k];

      /* compute exponential weights w.r.t. the best likelihood for replicate k */

      for(i = 0; i < numberOfTrees; i++)
	lhweights[i][k] = exp(lhs[i][k] - best);

      /* sum over all exponential weights */

      for(i = 0, sum = 0.0; i < numberOfTrees; i++)
	sum += lhweights[i][k];

      /* and normalize by the sum */

      for(i = 0; i < numberOfTrees; i++)
	lhweights[i][k] = lhweights[i][k] / sum;

    }

  /* now loop over all trees */

  for(i = 0; i < numberOfTrees; i++)
    {

      /* loop to sum over all replicate weights for tree i  */

      for(k = 0, sum = 0.0; k < adef->multipleRuns; k++)
	sum += lhweights[i][k];

      /* set the weight and the index of the respective tree */

      bootweights[i].weight = sum / ((double)adef->multipleRuns);
      bootweights[i].tree   = i;
    }

  /* now just sort the tree collection by weights */

  qsort(bootweights, numberOfTrees, sizeof(elw), elwCompare);

  printBothOpen("Tree\t Posterior Probability \t Cumulative posterior probability\n");

  /* loop over the sorted array of trees and print out statistics */

  for(i = 0, sum = 0.0; i < numberOfTrees; i++)
    {
      sum += bootweights[i].weight;

      printBothOpen("%d\t\t %f \t\t %f\n", bootweights[i].tree, bootweights[i].weight, sum);
    }


  if(0)  
    {
      /* now compute the super-duper rank test */

      printBothOpen("\n\nNow also computing the super-duper rank test, though I still don't\n");
      printBothOpen("understand what it actually means. What this thing does is to initially determine\n");
      printBothOpen("the best-scoring ML tree on the original alignment and then the scores of the input\n");
      printBothOpen("trees on the number of specified Bootstrap replicates. Then it sorts the scores of the trees\n");
      printBothOpen("for every bootstrap replicate and determines the rank of the best-scoring tree on every BS\n");
      printBothOpen("replicate. It then prints out how many positions in the sorted lists of thz BS replicates \n");
      printBothOpen("must be included in order for the best scoring tree to appear 95 and 99 times respectively.\n");
      printBothOpen("This gives some intuition about how variable the score order of the trees will be under\n");
      printBothOpen("slight alterations of the data.\n\n");

      /* sort all BS replicates accodring to likelihood scores */

      for(i = 0; i < adef->multipleRuns; i++)
	qsort(rankTest[i], numberOfTrees, sizeof(elw), elwCompareLikelihood);


      /* search for our best-scoring tree in every sorted array of likelihood scores */
      
      for(i = 0; i < adef->multipleRuns; i++)
	{
	  for(k = 0; k < numberOfTrees; k++)
	    {
	      if(rankTest[i][k].tree == bestIndex)
		countBest[k]++;
	    }
	}
      
      for(k = 0; k < numberOfTrees; k++)
	{
	  if(k > 0)
	    countBest[k] += countBest[k - 1];
	  
	  printBothOpen("Number of Occurences of best-scoring tree for %d BS replicates up to position %d in sorted list: %d\n",
			adef->multipleRuns, k, countBest[k]);
	  
	  if(cutOff95 == -1 && countBest[k] <= (int)((double)adef->multipleRuns * 0.95 + 0.5))
	    cutOff95 = k;
	  
	  if(cutOff99 == -1 && countBest[k] <= (int)((double)adef->multipleRuns * 0.99 + 0.5))
	    cutOff99 = k;
	}

      assert(countBest[k-1] == adef->multipleRuns);
      assert(cutOff95 >= 0 && cutOff99 >= 0);

      printBothOpen("\n95%s cutoff reached after including %d out of %d sorted likelihood columns\n", "%", countBest[cutOff95], adef->multipleRuns);
      
      printBothOpen("99%s cutoff reached after including %d out of %d sorted likelihood columns\n\n", "%", countBest[cutOff99], adef->multipleRuns);
    }

  printBothOpen("\nTotal execution time: %f\n\n", gettime() - masterTime);

  free(originalRateCategories);
  free(originalInvariant);

  fclose(INFILE);
  exit(0);
}



static void computeDistances(tree *tr, analdef *adef)
{
  int i, j, modelCounter;
  double z0[NUM_BRANCHES];
  double result[NUM_BRANCHES];
  double t;
  char distanceFileName[1024];

  FILE
    *out;

  strcpy(distanceFileName,         workdir);
  strcat(distanceFileName,         "RAxML_distances.");
  strcat(distanceFileName,         run_id);

  out = myfopen(distanceFileName, "w");

  modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);

  printBothOpen("\nLog Likelihood Score after parameter optimization: %f\n\n", tr->likelihood);
  printBothOpen("\nComputing pairwise ML-distances ...\n");

  for(modelCounter = 0; modelCounter < tr->NumberOfModels; modelCounter++)
    z0[modelCounter] = defaultz;

  t = gettime();

  for(i = 1; i <= tr->mxtips; i++)
    for(j = i + 1; j <= tr->mxtips; j++)
      {
	double z, x;

	makenewzGenericDistance(tr, 10, z0, result, i, j);

	if(tr->multiBranch)
	  {
	    int k;

	    for(k = 0, x = 0.0; k < tr->numBranches; k++)
	      {
		assert(tr->partitionContributions[k] != -1.0);
		assert(tr->fracchanges[k] != -1.0);
		z = result[k];
		if (z < zmin)
		  z = zmin;
		x += (-log(z) * tr->fracchanges[k]) * tr->partitionContributions[k];
	      }
	  }
	else
	  {
	    z = result[0];
	    if (z < zmin)
	      z = zmin;
	    x = -log(z) * tr->fracchange;
	  }

	/*printf("%s-%s \t %f\n", tr->nameList[i], tr->nameList[j], x);*/
	fprintf(out, "%s %s \t %f\n", tr->nameList[i], tr->nameList[j], x);
      }

  fclose(out);

  t = gettime() - t;

  printBothOpen("\nTime for pair-wise ML distance computation of %d distances: %f seconds\n",
		 (tr->mxtips * tr->mxtips - tr->mxtips) / 2, t);
  printBothOpen("\nDistances written to file: %s\n", distanceFileName);



  exit(0);
}



static void morphologicalCalibration(tree *tr, analdef *adef)
{
  int 
    replicates = adef->multipleRuns,
    i,     
    *significanceCounter = (int*)malloc(sizeof(int) * tr->cdta->endsite); 

  double 
    *reference  = (double*)malloc(sizeof(double) *  tr->cdta->endsite);

  char    
    integerFileName[1024] = "";

  FILE 
    *integerFile;

  if(replicates == 1)
    {
      printBothOpen("You did not specify the number of random trees to be generated by \"-#\" !\n");
      printBothOpen("Automatically setting it to 100.\n");
      replicates = 100;
    }      

  printBothOpen("Likelihood on Reference tree: %f\n\n", tr->likelihood);

  evaluateGenericInitrav(tr, tr->start);

  evaluateGenericVector(tr, tr->start);

  for(i = 0; i < tr->cdta->endsite; i++)    
    significanceCounter[i] = 0;             

  memcpy(reference, tr->perSiteLL, tr->cdta->endsite * sizeof(double));

  for(i = 0; i < replicates; i++)
    {    
      int k;
      
      printBothOpen("Testing Random Tree [%d]\n", i);
      makeRandomTree(tr, adef);
      evaluateGenericInitrav(tr, tr->start);
      treeEvaluate(tr, 2);
      
      /*
	don't really need modOpt here
	modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
      */
      
      evaluateGenericVector(tr, tr->start);
            
      
      for(k = 0; k < tr->cdta->endsite; k++)	
	if(tr->perSiteLL[k] <= reference[k])
	  significanceCounter[k] = significanceCounter[k] + 1;	        
    }
   
  strcpy(integerFileName,         workdir);
  strcat(integerFileName,         "RAxML_weights.");
  strcat(integerFileName,         run_id);

  integerFile = myfopen(integerFileName, "w");  

  for(i = 0; i < tr->cdta->endsite; i++)   
    fprintf(integerFile, "%d ", significanceCounter[i]);
    
  fclose(integerFile);
 
  printBothOpen("RAxML calibrated integer weight file written to: %s\n", integerFileName);

  exit(0);
}


static void minimumWeights(tree *tr, unsigned int *minWeights)
{
  int 
    i,
    j;
  
  for(i = 0; i < tr->cdta->endsite; i++)
    {
      int 
	count = -1,
	undetermined = getUndetermined(tr->dataVector[i]);

      unsigned int	
	accumulator = 0,
	nucleotide = 0;
      
      const unsigned int
	*bitVector = getBitVector(tr->dataVector[i]);          
      
      for(j = 1; j <= tr->mxtips; j++)
	{	   
	  if(tr->yVector[j][i] != undetermined)
	    {
	      nucleotide = bitVector[tr->yVector[j][i]];	
	      assert(nucleotide > 0);	      	     
	      
	      if((nucleotide & accumulator) == 0)
		{
		  count++;
		  accumulator |= nucleotide;
		}
	    }	  
	}
   
      assert(count >= 0);
      
      minWeights[i] = (unsigned int)count;
    }
}


static void morphologicalCalibrationParsimony(tree *tr)
{
  int
    i;

  unsigned int
    maximumDifference = 0,
    parsimonyScore = 0,
    *siteParsimony = (unsigned int*)calloc(tr->cdta->endsite, sizeof(unsigned int)),
    *minWeights    = (unsigned int*)calloc(tr->cdta->endsite, sizeof(unsigned int));
  
  double 
    scaler;

  char    
    integerFileName[1024] = "";

  FILE 
    *integerFile; 
  
  minimumWeights(tr, minWeights);
 
  initravParsimonyNormal(tr, tr->start);
  initravParsimonyNormal(tr, tr->start->back);  
  
  parsimonyScore = evaluatePerSiteParsimony(tr, tr->start, siteParsimony);

  /*printf("Parsimony Score: %u \n\n", parsimonyScore);*/

  for(i = 0; i < tr->cdta->endsite; i++)
    {
      assert(siteParsimony[i] >= minWeights[i]);
      
      siteParsimony[i] = siteParsimony[i] - minWeights[i];
      
      /*printf("%u ", siteParsimony[i]);*/
      
      if(siteParsimony[i] > maximumDifference)
	maximumDifference = siteParsimony[i];
    }
  
  /* printf("\nMax diff: %u\n", maximumDifference); */
 

  strcpy(integerFileName,         workdir);
  strcat(integerFileName,         "RAxML_parsimonyWeights.");
  strcat(integerFileName,         run_id);

  integerFile = myfopen(integerFileName, "w");  

  scaler = 100.0 / (double)maximumDifference;

  for(i = 0; i < tr->cdta->endsite; i++)   
    {
      unsigned int 
	value = 100 - (unsigned int)((double)siteParsimony[i] * scaler + 0.5);

      fprintf(integerFile, "%u ", value);
    }
    
  fclose(integerFile);
 
  printBothOpen("RAxML calibrated parsimony-based integer weight file written to: %s\n", integerFileName);

  exit(0);
}




static void extractTaxaFromTopology(tree *tr, rawdata *rdta, cruncheddata *cdta)
{
  FILE *f = myfopen(bootStrapFile, "r");

  char 
    **nameList,
    buffer[nmlngth + 2]; 

  int
    i = 0,
    c,
    taxaSize = 1024,
    taxaCount = 0;
   
  nameList = (char**)malloc(sizeof(char*) * taxaSize);  

  while((c = fgetc(f)) != ';')
    {
      if(c == '(' || c == ',')
	{
	  c = fgetc(f);
	  if(c ==  '(' || c == ',')
	    ungetc(c, f);
	  else
	    {	      
	      i = 0;	      	     
	      
	      do
		{
		  buffer[i++] = c;
		  c = fgetc(f);
		}
	      while(c != ':' && c != ')' && c != ',');
	      buffer[i] = '\0';	    

	      for(i = 0; i < taxaCount; i++)
		{
		  if(strcmp(buffer, nameList[i]) == 0)
		    {
		      printf("A taxon labelled by %s appears twice in the first tree of tree collection %s, exiting ...\n", buffer, bootStrapFile);
		      exit(-1);
		    }
		}	     
	     
	      if(taxaCount == taxaSize)
		{		  
		  taxaSize *= 2;
		  nameList = (char **)realloc(nameList, sizeof(char*) * taxaSize);		 
		}
	      
	      nameList[taxaCount] = (char*)malloc(sizeof(char) * (strlen(buffer) + 1));
	      strcpy(nameList[taxaCount], buffer);
	     
	      taxaCount++;
			    
	      ungetc(c, f);
	    }
	}   
    }
  
  printf("Found a total of %d taxa in first tree of tree collection %s\n", taxaCount, bootStrapFile);
  printf("Expecting all remaining trees in collection to have the same taxon set\n");

  rdta->numsp = taxaCount;

  tr->nameList = (char **)malloc(sizeof(char *) * (taxaCount + 1));  
  for(i = 1; i <= taxaCount; i++)
    tr->nameList[i] = nameList[i - 1];
  
  free(nameList);

  tr->rdta       = rdta;
  tr->cdta       = cdta;

  if (rdta->numsp < 4)
    {    
      printf("TOO FEW SPECIES, tree contains only %d species\n", rdta->numsp);
      assert(0);
    }

  tr->nameHash = initStringHashTable(10 * taxaCount);
  for(i = 1; i <= taxaCount; i++)
    addword(tr->nameList[i], tr->nameHash, i);

  fclose(f);
}



static void getInputVariables(int argc, char **argv)
{
    
 int i;
 int len;
 char nofSpecieschar[10];
 char alignLenchar[16];
 int j, k=0;

 nofSpecies=-1; alignLength=-1;  //global var

 for(i=0; i<argc; i++){
    printf("%s\n", argv[i]);

    if (!strncmp(argv[i], "-s", 2)){
	i++;
	len = strlen(argv[i]);
	//printf("%d\n", len);
	
	for(j=4; j<len; j++){
  	   //printf("%c\n", argv[i][j]);
	   while(strncmp(&argv[i][j] ,"_", 1)){
		
		nofSpecieschar[k]=argv[i][j];
		//printf("%c", nofSpecieschar[k]);
		k++;
		j++;
	   }
	   nofSpecieschar[k]='\0';
	   printf("%s\n", nofSpecieschar);
	   //printf("\n");
	   j++;
	   k=0;
	   while(strncmp(&argv[i][j] ,".", 1)){
                alignLenchar[k]=argv[i][j];
                //printf("%c", alignLenchar[k]);
                
                j++;
		k++;
           }
	   alignLenchar[k]='\0';
	   printf("%s\n", alignLenchar);


	j=len;
	}
    i=argc;
    }
	//printf("found it!\n");    
 }

 nofSpecies = atoi(nofSpecieschar);
 alignLength = atoi(alignLenchar);

 printf("final species: %d, alignLend: %d\n", nofSpecies, alignLength);    
    
}



int main (int argc, char *argv[])
{
  rawdata      *rdta;
  cruncheddata *cdta;
  tree         *tr;
  analdef      *adef;
  int
    i,
    countGTR = 0,
    countOtherModel = 0;
  
  
  getInputVariables(argc, argv);
  //assert(alignLength == 1024);
  //assert(nofSpecies == 40);
  
  
  //alignLength = 1024;
  //nofSpecies = 40;

#ifdef PARALLEL
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processID);
  MPI_Comm_size(MPI_COMM_WORLD, &numOfWorkers);
  if(processID == 0)
    printf("\nThis is the RAxML MPI Master process\n");
  else
    printf("\nThis is the RAxML MPI Worker Process Number: %d\n", processID);
#else
#ifdef _WAYNE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processID);
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  printf("\nThis is RAxML MPI Process Number: %d\n", processID);
#else
  processID = 0;
#endif
#endif

  masterTime = gettime();

#ifdef _IPTOL
  lastCheckpointTime = masterTime;
#endif


#ifdef  _USE_FPGA_LOG
  log_approx_init(12);
#endif

#ifdef _USE_FPGA_EXP
  exp_approx_init();
#endif

  _mm_setcsr( _mm_getcsr() | _MM_FLUSH_ZERO_ON );

  /*
    __builtin_ia32_ldmxcsr(__builtin_ia32_stmxcsr () | 0x8000 );
  */

  adef = (analdef *)malloc(sizeof(analdef));
  rdta = (rawdata *)malloc(sizeof(rawdata));
  cdta = (cruncheddata *)malloc(sizeof(cruncheddata));
  tr   = (tree *)malloc(sizeof(tree));
  
#ifdef MEMORG
  //tree //traversal info *tr->td[0].ti //int *tr->partitionData[0].wgt //uchar **tr->partitionData[0].yvector //uchar *yVector //float **xVector //float *xVector
  //size_t memoryRequirements = (size_t) 4*4*2048;// (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width; ////
  //size_t size_xVector = 107*memoryRequirements* sizeof(float);//tr->innerNodes * memoryRequirements * sizeof(float)////
 
  /*          tr->partitionData[i].EV_FLOAT          = (float *)malloc_aligned(pl->evLength * sizeof(float));
	  tr->partitionData[i].tipVector_FLOAT   = (float *)malloc_aligned(pl->tipVectorLength * sizeof(float));
	  tr->partitionData[i].left_FLOAT        = (float *)malloc_aligned(pl->leftLength * maxCategories * sizeof(float));
	  tr->partitionData[i].right_FLOAT       = (float *)malloc_aligned(pl->rightLength * maxCategories * sizeof(float));*/
  EV_size =  16 * sizeof(float) +16; //pl->evLength == 16
  tipVector_size =  64 * sizeof(float) +16; //pl->tipVectorLength == 64
  left_size = 16 * 25 * sizeof(float) +16; // pl->leftLength==16 maxCategories==25
  right_size = 16 * 25 * sizeof(float) +16; // pl->rightLength==16 maxCategories==25
  patrat_size = (alignLength + 1) * sizeof(double) +16; // rdta->sites + 1
  ei_size = 12 * sizeof(double) +16; // pl->eiLength == 12
  eign_size = 3 * sizeof(double) +16; // pl->eignLength == 3
  rateCategory_size = (alignLength +1) * sizeof(int) +16; //cdta->rateCategory= (int *)malloc((rdta->sites + 1) * sizeof(int));
  wgt_size = (alignLength + 1)*sizeof(int) +16; //(int *)malloc((rdta->sites + 1) * sizeof(int));
  gammaRates_size = 4*sizeof(double) +16;
  globalScaler_size = 2*nofSpecies*sizeof(int) +16; //2*tr->mxtips
  scalerThread_size = alignLength*2*nofSpecies*sizeof(int); //1.6mb
  size_t partitionLikelihood_size = alignLength*sizeof(double);
  size_t wr_size = (alignLength + 1) * sizeof(float) +16;//(rdta->sites + 1) * sizeof(float)
  size_t wr2_size = (alignLength + 1) * sizeof(float) +16;
  
  traversalInfo_size = nofSpecies*sizeof(traversalInfo) +16; //
  
  
  size_t toGPUsize;
  toGPUsize = wgt_size + patrat_size + rateCategory_size + wr_size + wr2_size + traversalInfo_size + ei_size + eign_size + EV_size + tipVector_size + gammaRates_size;
  printf("\n toGPUsize: %zu bytes\n", toGPUsize);
  //size_t yVector_size =  (107+1) * sizeof(unsigned char *) + (107) * 4*(2048/4 + 1)* sizeof(unsigned char);
  yVector_size =  (nofSpecies+1) * sizeof(unsigned char *)  +16 + nofSpecies*alignLength* sizeof(unsigned char) +16;

  memoryRequirements = (size_t) 4*4*alignLength; // (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width; ////
  xVector_size = nofSpecies*sizeof(float*) +16 + nofSpecies*memoryRequirements* sizeof(float) +16; //tr->innerNodes * memoryRequirements * sizeof(float)////
  
  dataMallocSize = sizeof(tree) +16 + wr_size + wr2_size + globalScaler_size + gammaRates_size + wgt_size + yVector_size + rateCategory_size + patrat_size + ei_size + eign_size + traversalInfo_size + xVector_size + EV_size + tipVector_size + left_size + right_size;  
  dataTransferSizeInit = sizeof(tree) + globalScaler_size + gammaRates_size + wgt_size  + yVector_size + rateCategory_size + patrat_size + ei_size + eign_size + traversalInfo_size  + EV_size + tipVector_size ; //+ xVector_size + left_size + right_size

  dataTransferSizeTestKernel = sizeof(tree) + globalScaler_size + gammaRates_size + wgt_size  + yVector_size + rateCategory_size + patrat_size + ei_size + eign_size + traversalInfo_size  + EV_size + tipVector_size + xVector_size + left_size + right_size; //
  //dataTransferSizePart = sizeof(tree) + wgt_size + rateCategory_size + patrat_size + ei_size + eign_size + traversalInfo_size  + EV_size + tipVector_size; 
  cudaError_t error;
  
  size_t stackSize;
  cudaDeviceGetLimit(&stackSize ,cudaLimitStackSize);
  printf("cudaLimitStackSize %d\n", (int)stackSize);
  cudaDeviceSetLimit (cudaLimitStackSize, 16384);
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaDeviceSetLimit : %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return 1;
	}
  cudaDeviceGetLimit(&stackSize ,cudaLimitStackSize);
  printf("cudaLimitStackSize %d\n", (int)stackSize);
  

  printf("\nGLOBAL mem req: %zu bytes\n", dataMallocSize);
  //tr->mxtips==107; rdta->sites==2048; rdta->numsp==107;
  yVectorBase = malloc(yVector_size);
  cudaMalloc((void **)&d_yVector, yVector_size);
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_yVector: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return 1;
	}
  
  
  globalp = malloc(dataMallocSize);
  //globalpStart = globalp; //misaligned version
  //add tree
  printf("\n!!!!STEP 0\n");
  tr = (tree *) (((uintptr_t)globalp + 16) & ~0x0F);//tr = (tree *)globalp; //misaligned version
  globalpStart = tr;
  globalp = tr +1;
  printf("sizeof struct tree: %zu bytes\n", sizeof(tree));
  
  h_partitionLikelihood = (double *)malloc(partitionLikelihood_size);
  cudaMalloc((void **)&d_partitionLikelihood, partitionLikelihood_size);
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_partitionLikelihood: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return 1;
	}  

  h_scalerThread = (int *) malloc(scalerThread_size);
  cudaMalloc((void **)&d_scalerThread, scalerThread_size);
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_scalerThread: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return 1;
	}
  cudaMalloc(&d_globalp, dataMallocSize);
  	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMalloc d_globalp: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    return 1;
	}
  //d_globalpStart = d_globalp; //misaligned version 
  
  d_startMisaligned = d_globalp;      
  d_tree = (tree *)(((uintptr_t)d_globalp + 16) & ~0x0F); //d_tree = d_globalp;
  d_globalpStart = d_tree;
  d_globalp = d_tree + 1;
  
  
//CUDA FREE in multiple.c
#endif
  
  initAdef(adef);
  get_args(argc,argv, adef, tr); 
  
  if(adef->readTaxaOnly)  
    extractTaxaFromTopology(tr, rdta, cdta);   
 //printf("!!!!! tr->mxtips %d\n", tr->mxtips); 0 not ready yet
  getinput(adef, rdta, cdta, tr);

  printf("!!!!! main tr->mxtips %d\n", tr->mxtips);
  assert(tr->mxtips == nofSpecies);
  
  checkOutgroups(tr, adef);
  makeFileNames();

#ifdef _WAYNE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if(adef->useInvariant && adef->likelihoodEpsilon > 0.001)
    {
      printBothOpen("\nYou are using a proportion of Invariable sites estimate, although I don't\n");
      printBothOpen("like it. The likelihood epsilon \"-f e\" will be automatically lowered to 0.001\n");
      printBothOpen("to avoid unfavorable effects caused by simultaneous optimization of alpha and P-Invar\n");

      adef->likelihoodEpsilon = 0.001;
    }


  /*
     switch back to model without secondary structure for all this
     checking stuff
  */

  if(adef->useSecondaryStructure)
    {
      tr->dataVector    = tr->initialDataVector;
      tr->partitionData = tr->initialPartitionData;
      tr->NumberOfModels--;
    }

  if(adef->useExcludeFile)
    {
      handleExcludeFile(tr, adef, rdta);
      exit(0);
    }

 
  if(!adef->readTaxaOnly && adef->mode != FAST_SEARCH)
    checkSequences(tr, rdta, adef);
  

  if(adef->mode == SPLIT_MULTI_GENE)
    {
      splitMultiGene(tr, rdta);
      exit(0);
    }

  if(adef->mode == CHECK_ALIGNMENT)
    {
      printf("Alignment format can be read by RAxML \n");
      exit(0);
    }

  /*
     switch back to model with secondary structure for all this
     checking stuff
  */

  if(adef->useSecondaryStructure && !adef->readTaxaOnly)
    {
      tr->dataVector    = tr->extendedDataVector;
      tr->partitionData = tr->extendedPartitionData;
      tr->NumberOfModels++;
      /* might as well free the initial structures here */

    }
  
  if(!adef->readTaxaOnly)
    {
      makeweights(adef, rdta, cdta, tr);
      makevalues(rdta, cdta, tr, adef);      

      for(i = 0; i < tr->NumberOfModels; i++)
	{
	  if(tr->partitionData[i].dataType == AA_DATA)
	    {
	      if(tr->partitionData[i].protModels == GTR)
		countGTR++;
	      else
		countOtherModel++;
	    }
	}

      if(countGTR > 0 && countOtherModel > 0)
	{
	  printf("Error, it is only allowed to conduct partitioned AA analyses\n");
	  printf("with a GTR model of AA substitution, if all AA partitions are assigned\n");
	  printf("the GTR model.\n\n");
	  
	  printf("The following partitions do not use GTR:\n");
	  
	  for(i = 0; i < tr->NumberOfModels; i++)
	    {
	      if(tr->partitionData[i].dataType == AA_DATA && tr->partitionData[i].protModels != GTR)
		printf("Partition %s\n", tr->partitionData[i].partitionName);
	    }
	  printf("exiting ...\n");
	  errorExit(-1);
	}

      if(countGTR > 0 && tr->NumberOfModels > 1)
	{
	  FILE *info = fopen(infoFileName, "a");

	  printBoth(info, "You are using the GTR model of AA substitution!\n");
	  printBoth(info, "GTR parameters for AA substiution will automatically be estimated\n");
	  printBoth(info, "jointly (GTR params will be linked) across all partitions to avoid over-parametrization!\n\n\n");

	  fclose(info);
	}
    }

  if(adef->mode == CLASSIFY_ML)              
    tr->innerNodes = (size_t)(countTaxaInTopology() - 1);   
  else
    tr->innerNodes = tr->mxtips;

  
  setRateHetAndDataIncrement(tr, adef);

#ifdef _USE_PTHREADS
  /* integrate !readTaxaOnly ! */
  startPthreads(tr);
  masterBarrier(THREAD_INIT_PARTITION, tr);
  masterBarrier(THREAD_ALLOC_LIKELIHOOD, tr);
#else
  if(!adef->readTaxaOnly)       
    allocNodex(tr);    
#endif

  makeMissingData(tr);

  printModelAndProgramInfo(tr, adef, argc, argv);

  
  
  /*
   * INIT kernel variables
   */
  
 
    cudaMalloc((void **)&d2_umpX1, alignLength*256*sizeof(float));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
        printf("CUDA Error AFTER cudaMalloc d2_umpX1: %s\n", cudaGetErrorString(error));
    // we can't recover from the error -- exit the program
        return 1;
    }    
    cudaMalloc((void **)&d2_umpX2, alignLength*256*sizeof(float));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
        printf("CUDA Error AFTER cudaMalloc d2_umpX2: %s\n", cudaGetErrorString(error));
    // we can't recover from the error -- exit the program
        return 1;
    }      
  
    cudaMalloc((void **)&d2_left, alignLength*400*sizeof(float));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
        printf("CUDA Error AFTER cudaMalloc d2_left: %s\n", cudaGetErrorString(error));
    // we can't recover from the error -- exit the program
        return 1;
    }        
  
    cudaMalloc((void **)&d2_right, alignLength*400*sizeof(float));
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
        printf("CUDA Error AFTER cudaMalloc d2_right: %s\n", cudaGetErrorString(error));
    // we can't recover from the error -- exit the program
        return 1;
    }   
  
  initKernel();
  
  switch(adef->mode)
    {
    case THOROUGH_PARSIMONY:
      makeParsimonyTreeThorough(tr, adef);
      break;
    case CLASSIFY_ML:
      initModel(tr, rdta, cdta, adef);
      getStartingTree(tr, adef);
      exit(0);
      break;
    case GENERATE_BS:
      generateBS(tr, adef);
      exit(0);
      break;
    case COMPUTE_ELW:
      computeELW(tr, adef, bootStrapFile);
      exit(0);
      break;
    case COMPUTE_LHS:
      initModel(tr, rdta, cdta, adef);
      computeAllLHs(tr, adef, bootStrapFile);
      exit(0);
      break;
    case COMPUTE_BIPARTITION_CORRELATION:
      compareBips(tr, bootStrapFile, adef);
      exit(0);
      break;
    case COMPUTE_RF_DISTANCE:
      computeRF(tr, bootStrapFile, adef);
      exit(0);
      break;
    case BOOTSTOP_ONLY:
      computeBootStopOnly(tr, bootStrapFile, adef);
      exit(0);
      break;
    case CONSENSUS_ONLY:      
      computeConsensusOnly(tr, bootStrapFile, adef);
      exit(0);
      break;
    case DISTANCE_MODE:
      initModel(tr, rdta, cdta, adef);
      getStartingTree(tr, adef);
      computeDistances(tr, adef);
      break;
    case  PARSIMONY_ADDITION:
      initModel(tr, rdta, cdta, adef);
      getStartingTree(tr, adef);
      printStartingTree(tr, adef, TRUE);
      break;
    case PER_SITE_LL:
      initModel(tr, rdta, cdta, adef);
      computePerSiteLLs(tr, adef, bootStrapFile);
      break;
    case TREE_EVALUATION:
      initModel(tr, rdta, cdta, adef);
      getStartingTree(tr, adef);      
      if(adef->likelihoodTest)
	computeLHTest(tr, adef, bootStrapFile);
      else
	{
	  modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
	  printLog(tr, adef, TRUE);
	  printResult(tr, adef, TRUE);
	}
      break;
    case CALC_BIPARTITIONS:
      initModel(tr, rdta, cdta, adef);
      calcBipartitions(tr, adef, tree_file, bootStrapFile);
      break;
    case BIG_RAPID_MODE:
      if(adef->boot) //adef->boot=0
	doBootstrap(tr, adef, rdta, cdta);
      else
	{

	  if(adef->rapidBoot) //adef->rapidBoot=0
	    {
	      initModel(tr, rdta, cdta, adef);

	      doAllInOne(tr, adef);
	    }
	  else
	    {
	    
	      doInference(tr, adef, rdta, cdta);
	     
	    }
	}
      break;
    case MORPH_CALIBRATOR:
      initModel(tr, rdta, cdta, adef);
      getStartingTree(tr, adef);
      modOpt(tr, adef, TRUE, adef->likelihoodEpsilon);
      morphologicalCalibration(tr, adef);
      break;
    case MORPH_CALIBRATOR_PARSIMONY:
      initModel(tr, rdta, cdta, adef);
      getStartingTree(tr, adef);     
      morphologicalCalibrationParsimony(tr);
      break;
    case MESH_TREE_SEARCH:
      initModel(tr, rdta, cdta, adef); 
      getStartingTree(tr, adef); 
      meshTreeSearch(tr, adef, adef->meshSearch);
      /* TODO */
      break;
    case FAST_SEARCH:
      fastSearch(tr, adef, rdta, cdta);
      exit(0);
    default:
      assert(0);
    }

  finalizeInfoFile(tr, adef);

#if defined PARALLEL || defined _WAYNE_MPI
  MPI_Finalize();
#endif

  return 0;
}


