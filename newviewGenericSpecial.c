/*  RAxML-VI-HPC (version 2.2) a program for sequential and parallel estimation of phylogenetic trees
 *  Copyright August 2006 by Alexandros Stamatakis
 *
 *  Partially derived from
 *  fastDNAml, a program for estimation of phylogenetic trees from sequences by Gary J. Olsen
 *
 *  and
 *
 *  Programs of the PHYLIP package by Joe Felsenstein.
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

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "axml.h"

#ifdef MEMORG
#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

//#define checkLR
extern int timesCalledNewView, 
    timesCalledNewz,
    timesCalledEvaluate;

extern float T2DnewView; 
extern float KenrelTnewView; 
extern float T2HnewView;

extern float T2Devaluate; 
extern float KenrelTevaluate; 
extern float T2Hevaluate;

extern int alignLength;
extern int nofSpecies;

extern int firstIter;

extern void *yVectorBase;


extern int gCounter;
extern void *globalpStart, *globalpEnd;
extern void *d_globalpStart, *d_globalpEnd;
extern size_t dataTransferSizeInit , memoryRequirements, dataTransferSizeTestKernel;
extern size_t yVector_size, left_size, globalScaler_size, gammaRates_size, ei_size, eign_size, EV_size, tipVector_size, wgt_size, patrat_size, rateCategory_size, traversalInfo_size;
extern size_t dataMallocSize;
extern int *testData;
extern int ydim1, ydim2;
extern double *d_eign;
extern float **d_xVector;
extern unsigned char **d_yVector;
extern double *d_gammaRates;
extern unsigned int *d_globalScaler, *d_scalerThread;
extern float *d_left, *d_right;
extern double *d_partitionLikelihood;
extern int *d_wgt;
extern traversalInfo *d_ti;

extern double *h_partitionLikelihood;
extern int *h_scalerThread;
extern float **xVector_dp;

extern double fullTimeKe, fullTimeTr;

//float *testDataTransfer;
void kernelnewViewEvaluate(int , int , int , int );
void kernelNewview(tree *);
//void kernelScale(unsigned int *d_scalerThread, unsigned int *d_globalScaler);
#endif

#ifndef WIN32
#include <unistd.h>
#endif



#ifdef __SIM_SSE3

#include <stdint.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

const union __attribute__ ((aligned (16)))
{
       uint64_t i[2];
       __m128d m;
} absMask = {{0x7fffffffffffffffULL , 0x7fffffffffffffffULL }};

const union __attribute__ ((aligned (16)))
{
       uint64_t i[2];
       __m128 m;
} absMask_FLOAT = {{0x7fffffffffffffffULL , 0x7fffffffffffffffULL }};


#endif


#ifdef _USE_PTHREADS
#include <pthread.h>
extern volatile int NumberOfThreads;
#endif


extern const int mask32[32];

static void makeP_Flex(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, double *left, double *right, const int numStates)
{
  int 
    i,
    j,
    k;
  
  const int
    rates = numStates - 1,
    statesSquare = numStates * numStates;

  double 
    lz1[64],
    lz2[64], 
    d1[64],  
    d2[64];

  assert(numStates <= 64);
     
  for(i = 0; i < rates; i++)
    {
      lz1[i] = EIGN[i] * z1;
      lz2[i] = EIGN[i] * z2;
    }

  for(i = 0; i < numberOfCategories; i++)
    {
      for(j = 0; j < rates; j++)
	{
	  d1[j] = EXP (rptr[i] * lz1[j]);
	  d2[j] = EXP (rptr[i] * lz2[j]);
	}

      for(j = 0; j < numStates; j++)
	{
	  left[statesSquare * i  + numStates * j] = 1.0;
	  right[statesSquare * i + numStates * j] = 1.0;

	  for(k = 1; k < numStates; k++)
	    {
	      left[statesSquare * i + numStates * j + k]  = d1[k-1] * EI[rates * j + (k-1)];
	      right[statesSquare * i + numStates * j + k] = d2[k-1] * EI[rates * j + (k-1)];
	    }
	}
    }  
}


static void newviewFlexCat(int tipCase, double *extEV,
			   int *cptr,
			   double *x1, double *x2, double *x3, double *tipVector,
			   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			   int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling, const int numStates)
{
  double
    *le, *ri, *v, *vl, *vr,
    ump_x1, ump_x2, x1px2;
  int i, l, j, scale, addScale = 0;

  const int 
    statesSquare = numStates * numStates;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * statesSquare];
	    ri = &right[cptr[i] * statesSquare];

	    vl = &(tipVector[numStates * tipX1[i]]);
	    vr = &(tipVector[numStates * tipX2[i]]);
	    v  = &x3[numStates * i];

	    for(l = 0; l < numStates; l++)
	      v[l] = 0.0;

	    for(l = 0; l < numStates; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < numStates; j++)
		  {
		    ump_x1 += vl[j] * le[l * numStates + j];
		    ump_x2 += vr[j] * ri[l * numStates + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < numStates; j++)
		  v[j] += x1px2 * extEV[l * numStates + j];
	      }	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * statesSquare];
	    ri = &right[cptr[i] * statesSquare];

	    vl = &(tipVector[numStates * tipX1[i]]);
	    vr = &x2[numStates * i];
	    v  = &x3[numStates * i];

	    for(l = 0; l < numStates; l++)
	      v[l] = 0.0;

	    for(l = 0; l < numStates; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < numStates; j++)
		  {
		    ump_x1 += vl[j] * le[l * numStates + j];
		    ump_x2 += vr[j] * ri[l * numStates + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < numStates; j++)
		  v[j] += x1px2 * extEV[l * numStates + j];
	      }

	    scale = 1;
	    for(l = 0; scale && (l < numStates); l++)
	      scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	    

	    if(scale)
	      {
		for(l = 0; l < numStates; l++)
		  v[l] *= twotothe256;
		
		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * statesSquare];
	  ri = &right[cptr[i] * statesSquare];

	  vl = &x1[numStates * i];
	  vr = &x2[numStates * i];
	  v = &x3[numStates * i];

	  for(l = 0; l < numStates; l++)
	    v[l] = 0.0;

	  for(l = 0; l < numStates; l++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;

	      for(j = 0; j < numStates; j++)
		{
		  ump_x1 += vl[j] * le[l * numStates + j];
		  ump_x2 += vr[j] * ri[l * numStates + j];
		}

	      x1px2 =  ump_x1 * ump_x2;

	      for(j = 0; j < numStates; j++)
		v[j] += x1px2 * extEV[l * numStates + j];
	    }

	   scale = 1;
	   for(l = 0; scale && (l < numStates); l++)
	     scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));
	  
	   if(scale)
	     {
	       for(l = 0; l < numStates; l++)
		 v[l] *= twotothe256;

	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	     
	     }
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}

static void newviewFlexGamma(int tipCase,
			     double *x1, double *x2, double *x3, double *extEV, double *tipVector,
			     int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			     int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling, 
			     const int numStates)
{
  double  *v;
  double x1px2;
  int  i, j, l, k, scale, addScale = 0;
  double *vl, *vr, al, ar;

  const int 
    statesSquare = numStates * numStates,
    gammaStates = 4 * numStates;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for(i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[numStates * tipX1[i]]);
		vr = &(tipVector[numStates * tipX2[i]]);
		v =  &(x3[gammaStates * i + numStates * k]);

		for(l = 0; l < numStates; l++)
		  v[l] = 0;

		for(l = 0; l < numStates; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < numStates; j++)
		      {
			al += vl[j] * left[k * statesSquare + l * numStates + j];
			ar += vr[j] * right[k * statesSquare + l * numStates + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < numStates; j++)
		      v[j] += x1px2 * extEV[numStates * l + j];
		  }
	      }	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[numStates * tipX1[i]]);
		vr = &(x2[gammaStates * i + numStates * k]);
		v =  &(x3[gammaStates * i + numStates * k]);

		for(l = 0; l < numStates; l++)
		  v[l] = 0;

		for(l = 0; l < numStates; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < numStates; j++)
		      {
			al += vl[j] * left[k * statesSquare + l * numStates + j];
			ar += vr[j] * right[k * statesSquare + l * numStates + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < numStates; j++)
		      v[j] += x1px2 * extEV[numStates * l + j];
		  }
	      }
	   
	    v = &x3[gammaStates * i];
	    scale = 1;
	    for(l = 0; scale && (l < gammaStates); l++)
	      scale = (ABS(v[l]) <  minlikelihood);

	    if(scale)
	      {
		for(l = 0; l < gammaStates; l++)
		  v[l] *= twotothe256;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
       {
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[gammaStates * i + numStates * k]);
	     vr = &(x2[gammaStates * i + numStates * k]);
	     v =  &(x3[gammaStates * i + numStates * k]);

	     for(l = 0; l < numStates; l++)
	       v[l] = 0;

	     for(l = 0; l < numStates; l++)
	       {
		 al = 0.0;
		 ar = 0.0;
		 for(j = 0; j < numStates; j++)
		   {
		     al += vl[j] * left[k * statesSquare + l * numStates + j];
		     ar += vr[j] * right[k * statesSquare + l * numStates + j];
		   }

		 x1px2 = al * ar;
		 for(j = 0; j < numStates; j++)
		   v[j] += x1px2 * extEV[numStates * l + j];
	       }
	   }
	 
	 v = &(x3[gammaStates * i]);
	 scale = 1;
	 for(l = 0; scale && (l < gammaStates); l++)
	   scale = ((ABS(v[l]) <  minlikelihood));

	 if (scale)
	   {
	     for(l = 0; l < gammaStates; l++)
	       v[l] *= twotothe256;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	    
	   }
       }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}


static void makeP(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, double *left, double *right, int data)
{
  int i, j, k;

  switch(data)
    {
    case BINARY_DATA:
      {
	double d1, d2;

	for(i = 0; i < numberOfCategories; i++)
	  {
	    d1 = EXP(rptr[i] * EIGN[0] * z1);
	    d2 = EXP(rptr[i] * EIGN[0] * z2);
	    
	    for(j = 0; j < 2; j++)
	      {
		left[i * 4 + j * 2] = 1.0;
		right[i * 4 + j * 2] = 1.0;

		left[i * 4 + j * 2 + 1]  = d1 * EI[j];
		right[i * 4 + j * 2 + 1] = d2 * EI[j];	
	      }
	  }
      }
      break;
    case DNA_DATA:
      {
#ifdef __SIM_SSE3
	double 
	  d1[4] __attribute__ ((aligned (16))), 
	  d2[4] __attribute__ ((aligned (16))),
	  ez1[3], 
	  ez2[3],
	  EI_16[16] __attribute__ ((aligned (16)));
	
	  	  
	for(j = 0; j < 4; j++)
	  {
	    EI_16[j * 4] = 1.0;
	    for(k = 0; k < 3; k++)
	      EI_16[j * 4 + k + 1] = EI[3 * j + k];
	  }	  

	for(j = 0; j < 3; j++)
	  {
	    ez1[j] = EIGN[j] * z1;
	    ez2[j] = EIGN[j] * z2;
	  }


	for(i = 0; i < numberOfCategories; i++)
	  {	   
	    __m128d 
	      d1_0, d1_1,
	      d2_0, d2_1;
 
	    d1[0] = 1.0;
	    d2[0] = 1.0;

	    for(j = 0; j < 3; j++)
	      {
		d1[j+1] = EXP(rptr[i] * ez1[j]);
		d2[j+1] = EXP(rptr[i] * ez2[j]);
	      }

	    d1_0 = _mm_load_pd(&d1[0]);
	    d1_1 = _mm_load_pd(&d1[2]);

	    d2_0 = _mm_load_pd(&d2[0]);
	    d2_1 = _mm_load_pd(&d2[2]);
	    

	    for(j = 0; j < 4; j++)
	      {	       
		double *ll = &left[i * 16 + j * 4];
		double *rr = &right[i * 16 + j * 4];
		double *ee = &EI_16[4 * j];

		__m128d eev = _mm_load_pd(&EI_16[4 * j]);
		
		_mm_store_pd(&ll[0], _mm_mul_pd(d1_0, eev));
		_mm_store_pd(&rr[0], _mm_mul_pd(d2_0, eev));
		
		eev = _mm_load_pd(&EI_16[4 * j + 2]);
		
		_mm_store_pd(&ll[2], _mm_mul_pd(d1_1, eev));
		_mm_store_pd(&rr[2], _mm_mul_pd(d2_1, eev));

		/*for(k = 0; k < 4; k++)
		  {
		    ll[k]  = d1[k] * ee[k];
		    rr[k]  = d2[k] * ee[k];
		    }*/
	      }
	  }	

#else
	double d1[3], d2[3];

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 3; j++)
	      {
		d1[j] = EXP(rptr[i] * EIGN[j] * z1);
		d2[j] = EXP(rptr[i] * EIGN[j] * z2);
	      }

	    for(j = 0; j < 4; j++)
	      {
		left[i * 16 + j * 4] = 1.0;
		right[i * 16 + j * 4] = 1.0;

		for(k = 0; k < 3; k++)
		  {
		    left[i * 16 + j * 4 + k + 1]  = d1[k] * EI[3 * j + k];
		    right[i * 16 + j * 4 + k + 1] = d2[k] * EI[3 * j + k];
		  }
	      }
	  }
#endif
      }
      break;
    case SECONDARY_DATA:
      {
	double lz1[15], lz2[15], d1[15], d2[15];

	for(i = 0; i < 15; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 15; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 16; j++)
	      {
		left[256 * i  + 16 * j] = 1.0;
		right[256 * i + 16 * j] = 1.0;

		for(k = 1; k < 16; k++)
		  {
		    left[256 * i + 16 * j + k]  = d1[k-1] * EI[15 * j + (k-1)];
		    right[256 * i + 16 * j + k] = d2[k-1] * EI[15 * j + (k-1)];
		  }
	      }
	  }
      }
      break;
    case SECONDARY_DATA_6:
      {
	double lz1[5], lz2[5], d1[5], d2[5];

	for(i = 0; i < 5; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 5; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 6; j++)
	      {
		left[36 * i  + 6 * j] = 1.0;
		right[36 * i + 6 * j] = 1.0;

		for(k = 1; k < 6; k++)
		  {
		    left[36 * i + 6 * j + k]  = d1[k-1] * EI[5 * j + (k-1)];
		    right[36 * i + 6 * j + k] = d2[k-1] * EI[5 * j + (k-1)];
		  }
	      }
	  }
      }
      break;
    case SECONDARY_DATA_7:
      {
	double lz1[6], lz2[6], d1[6], d2[6];

	for(i = 0; i < 6; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 6; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 7; j++)
	      {
		left[49 * i  + 7 * j] = 1.0;
		right[49 * i + 7 * j] = 1.0;

		for(k = 1; k < 7; k++)
		  {
		    left[49 * i + 7 * j + k]  = d1[k-1] * EI[6 * j + (k-1)];
		    right[49 * i + 7 * j + k] = d2[k-1] * EI[6 * j + (k-1)];
		  }
	      }
	  }
      }
      break;
    case AA_DATA:
      {
	double lz1[19], lz2[19], d1[19], d2[19];

	for(i = 0; i < 19; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 19; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 20; j++)
	      {
		left[400 * i  + 20 * j] = 1.0;
		right[400 * i + 20 * j] = 1.0;

		for(k = 1; k < 20; k++)
		  {
		    left[400 * i + 20 * j + k]  = d1[k-1] * EI[19 * j + (k-1)];
		    right[400 * i + 20 * j + k] = d2[k-1] * EI[19 * j + (k-1)];
		  }
	      }
	  }
      }
      break;
    default:
      assert(0);
    }

}
/*
 * makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);
 */


static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
{
  int i, j, k;

  switch(data)
    {
    case DNA_DATA:
       {
	double d1[3], d2[3];

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 3; j++)
	      {
		d1[j] = EXP(rptr[i] * EIGN[j] * z1);
		d2[j] = EXP(rptr[i] * EIGN[j] * z2);
	      }

	    for(j = 0; j < 4; j++)
	      {
		left[i * 16 + j * 4] = 1.0;
		right[i * 16 + j * 4] = 1.0;

		for(k = 0; k < 3; k++)
		  {
		    left[i * 16 + j * 4 + k + 1]  = ((float)(d1[k] * EI[3 * j + k]));
		    right[i * 16 + j * 4 + k + 1] = ((float)(d2[k] * EI[3 * j + k]));
		  }
	      }
	  }
       }
       break;
    case BINARY_DATA:
      {
	double d1, d2;

	for(i = 0; i < numberOfCategories; i++)
	  {
	    d1 = EXP(rptr[i] * EIGN[0] * z1);
	    d2 = EXP(rptr[i] * EIGN[0] * z2);

	    for(j = 0; j < 2; j++)
	      {
		left[i * 4 + j * 2] = 1.0;
		right[i * 4 + j * 2] = 1.0;

		left[i * 4 + j * 2 + 1]  = ((float)(d1 * EI[j]));
		right[i * 4 + j * 2 + 1] = ((float)(d2 * EI[j]));
	      }
	  }
      }
      break;
    case SECONDARY_DATA:
      {
	double lz1[15], lz2[15], d1[15], d2[15];

	for(i = 0; i < 15; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 15; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 16; j++)
	      {
		left[256 * i  + 16 * j] = 1.0;
		right[256 * i + 16 * j] = 1.0;

		for(k = 1; k < 16; k++)
		  {
		    left[256 * i + 16 * j + k]  = ((float)(d1[k-1] * EI[15 * j + (k-1)]));
		    right[256 * i + 16 * j + k] = ((float)(d2[k-1] * EI[15 * j + (k-1)]));
		  }
	      }
	  }
      }
      break;
    case SECONDARY_DATA_6:
      {
	double lz1[5], lz2[5], d1[5], d2[5];

	for(i = 0; i < 5; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 5; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 6; j++)
	      {
		left[36 * i  + 6 * j] = 1.0;
		right[36 * i + 6 * j] = 1.0;

		for(k = 1; k < 6; k++)
		  {
		    left[36 * i + 6 * j + k]  = ((float)(d1[k-1] * EI[5 * j + (k-1)]));
		    right[36 * i + 6 * j + k] = ((float)(d2[k-1] * EI[5 * j + (k-1)]));
		  }
	      }
	  }
      }
      break;
    case SECONDARY_DATA_7:
      {
	double lz1[6], lz2[6], d1[6], d2[6];

	for(i = 0; i < 6; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 6; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 7; j++)
	      {
		left[49 * i  + 7 * j] = 1.0;
		right[49 * i + 7 * j] = 1.0;

		for(k = 1; k < 7; k++)
		  {
		    left[49 * i + 7 * j + k]  = ((float)(d1[k-1] * EI[6 * j + (k-1)]));
		    right[49 * i + 7 * j + k] = ((float)(d2[k-1] * EI[6 * j + (k-1)]));
		  }
	      }
	  }
      }
      break;
    case AA_DATA:
      {
	double lz1[19], lz2[19], d1[19], d2[19];

	for(i = 0; i < 19; i++)
	  {
	    lz1[i] = EIGN[i] * z1;
	    lz2[i] = EIGN[i] * z2;
	  }

	for(i = 0; i < numberOfCategories; i++)
	  {
	    for(j = 0; j < 19; j++)
	      {
		d1[j] = EXP (rptr[i] * lz1[j]);
		d2[j] = EXP (rptr[i] * lz2[j]);
	      }

	    for(j = 0; j < 20; j++)
	      {
		left[400 * i  + 20 * j] = 1.0;
		right[400 * i + 20 * j] = 1.0;

		for(k = 1; k < 20; k++)
		  {
		    left[400 * i + 20 * j + k]  = ((float)(d1[k-1] * EI[19 * j + (k-1)]));
		    right[400 * i + 20 * j + k] = ((float)(d2[k-1] * EI[19 * j + (k-1)]));
		  }
	      }
	  }
      }
      break;
    default:
      assert(0);
    }  
}



static void newviewGTRCAT_BINARY( int tipCase,  double *EV,  int *cptr,
				  double *x1_start,  double *x2_start,  double *x3_start,  double *tipVector,
				  int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				  int n,  double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le,
    *ri,
    *x1, *x2, *x3;
  double
    ump_x1, ump_x2, x1px2[2];
  int i, j, k, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[2 * tipX1[i]]);
	    x2 = &(tipVector[2 * tipX2[i]]);
	    x3 = &x3_start[2 * i];
	    /*printf("%f %f %f %f\n", x1[0], x1[1], x2[0], x2[1]);*/

	    le =  &left[cptr[i] * 4];
	    ri =  &right[cptr[i] * 4];

	    for(j = 0; j < 2; j++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;
		for(k = 0; k < 2; k++)
		  {
		    ump_x1 += x1[k] * le[j * 2 + k];
		    ump_x2 += x2[k] * ri[j * 2 + k];
		  }
		x1px2[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 2; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 2; j++)
	      for(k = 0; k < 2; k++)
		x3[k] += x1px2[j] * EV[j * 2 + k];	   
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[2 * tipX1[i]]);
	    x2 = &x2_start[2 * i];
	    x3 = &x3_start[2 * i];
	    /*printf("%f %f %f %f\n", x1[0], x1[1], x2[0], x2[1]);*/
	    le =  &left[cptr[i] * 4];
	    ri =  &right[cptr[i] * 4];

	    for(j = 0; j < 2; j++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;
		for(k = 0; k < 2; k++)
		  {
		    ump_x1 += x1[k] * le[j * 2 + k];
		    ump_x2 += x2[k] * ri[j * 2 + k];
		  }
		x1px2[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 2; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 2; j++)
	      for(k = 0; k < 2; k++)
		x3[k] +=  x1px2[j] *  EV[2 * j + k];	   

	    scale = 1;
	    for(j = 0; j < 2 && scale; j++)
	      scale = (x3[j] < minlikelihood && x3[j] > minusminlikelihood);

	    if(scale)
	      {
		for(j = 0; j < 2; j++)
		  x3[j] *= twotothe256;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	       
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &x1_start[2 * i];
	  x2 = &x2_start[2 * i];
	  x3 = &x3_start[2 * i];

	  le = &left[cptr[i] * 4];
	  ri = &right[cptr[i] * 4];

	  for(j = 0; j < 2; j++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;
	      for(k = 0; k < 2; k++)
		{
		  ump_x1 += x1[k] * le[j * 2 + k];
		  ump_x2 += x2[k] * ri[j * 2 + k];
		}
	      x1px2[j] = ump_x1 * ump_x2;
	    }

	  for(j = 0; j < 2; j++)
	    x3[j] = 0.0;

	  for(j = 0; j < 2; j++)
	    for(k = 0; k < 2; k++)
	      x3[k] +=  x1px2[j] *  EV[2 * j + k];	  

	  scale = 1;
	  for(j = 0; j < 2 && scale; j++)
	    scale = (x3[j] < minlikelihood && x3[j] > minusminlikelihood);

	  if(scale)
	    {
	      for(j = 0; j < 2; j++)
		x3[j] *= twotothe256;

	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;	   
	    }
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}

static void newviewGTRGAMMA_BINARY(int tipCase,
				   double *x1_start, double *x2_start, double *x3_start,
				   double *EV, double *tipVector,
				   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				   const int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling
				   )
{
  double
    *x1, *x2, *x3;
  double
    ump_x1,
    ump_x2,
    x1px2[4];
  int i, j, k, l, scale, addScale = 0;


  /* C-OPT figure out if we are at an inner node who has two tips/leaves
     as descendants TIP_TIP, a tip and another inner node as descendant
     TIP_INNER, or two inner nodes as descendants INNER_INNER */

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[2 * tipX1[i]]);
	    x2 = &(tipVector[2 * tipX2[i]]);
	    x3 = &x3_start[i * 8];

	    for(j = 0; j < 8; j++)
	      x3[j] = 0.0;

	    for (j = 0; j < 4; j++)
	      {
		for (k = 0; k < 2; k++)
		  {
		    ump_x1 = 0.0;
		    ump_x2 = 0.0;

		    for (l=0; l < 2; l++)
		      {
			ump_x1 += x1[l] * left[ j*4 + k*2 + l];
			ump_x2 += x2[l] * right[j*4 + k*2 + l];
		      }

		    x1px2[k] = ump_x1 * ump_x2;
		  }

		for(k = 0; k < 2; k++)
		  for (l = 0; l < 2; l++)
		    x3[j * 2 + l] +=  x1px2[k] * EV[2 * k + l];

	      }	   
	  }
      }
      break;
    case TIP_INNER:
      {
	 for (i = 0; i < n; i++)
	   {
	     x1 = &(tipVector[2 * tipX1[i]]);
	     x2 = &x2_start[i * 8];
	     x3 = &x3_start[i * 8];

	     for(j = 0; j < 8; j++)
	       x3[j] = 0.0;

	     for (j = 0; j < 4; j++)
	       {
		 for (k = 0; k < 2; k++)
		   {
		     ump_x1 = 0.0;
		     ump_x2 = 0.0;

		     for (l=0; l < 2; l++)
		       {
			 ump_x1 += x1[l] * left[ j*4 + k*2 + l];
			 ump_x2 += x2[j*2 + l] * right[j*4 + k*2 + l];
		       }

		     x1px2[k] = ump_x1 * ump_x2;
		   }

		 for(k = 0; k < 2; k++)
		   for (l = 0; l < 2; l++)
		     x3[j * 2 + l] +=  x1px2[k] * EV[2 * k + l];

	       }	    

	     scale = 1;
	     for(l = 0; scale && (l < 8); l++)
	       scale = (ABS(x3[l]) <  minlikelihood);

	     if(scale)
	       {
		 for (l=0; l < 8; l++)
		   x3[l] *= twotothe256;
		 
		 if(useFastScaling)
		   addScale += wgt[i];
		 else
		   ex3[i]  += 1;	       
	       }

	   }
      }
      break;
    case INNER_INNER:

      /* C-OPT here we don't do any pre-computations
	 This should be the most compute intensive loop of the three
	 cases here. If we have one or two tips as descendants
	 we can take a couple of shortcuts */


     for (i = 0; i < n; i++)
       {
	 x1 = &x1_start[i * 8];
	 x2 = &x2_start[i * 8];
	 x3 = &x3_start[i * 8];

	 for(j = 0; j < 8; j++)
	   x3[j] = 0.0;

	 for (j = 0; j < 4; j++)
	   {
	     for (k = 0; k < 2; k++)
	       {
		 ump_x1 = 0.0;
		 ump_x2 = 0.0;

		 for (l=0; l < 2; l++)
		   {
		     ump_x1 += x1[j*2 + l] * left[ j*4 + k*2 + l];
		     ump_x2 += x2[j*2 + l] * right[j*4 + k*2 + l];
		   }

		 x1px2[k] = ump_x1 * ump_x2;
	       }

	     for(k = 0; k < 2; k++)
	       for (l = 0; l < 2; l++)
		 x3[j * 2 + l] +=  x1px2[k] * EV[2 * k + l];

	   }
	 
	 scale = 1;
	 for(l = 0; scale && (l < 8); l++)
	   scale = (ABS(x3[l]) <  minlikelihood);


	 if(scale)
	   {
	     for (l=0; l<8; l++)
	       x3[l] *= twotothe256;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	  
	   }
       }
     break;

    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}


/*
 * newviewGTRCAT_FLOAT(tInfo->tipCase,  tr->partitionData[model].EV_FLOAT, tr->partitionData[model].rateCategory,
					       x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
					       ex3, tipX1, tipX2,
					       width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling
					       );
 */

static void newviewGTRCAT_FLOAT( int tipCase,  float *EV,  int *cptr,
				 float *x1_start,  float *x2_start,  float *x3_start,  float *tipVector,
				 int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				 int n,  float *left, float *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  float
    *le,
    *ri,
    *x1, *x2, *x3;
  float
    ump_x1, ump_x2, x1px2[4];
  int i, j, k, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[4 * tipX1[i]]);
	    x2 = &(tipVector[4 * tipX2[i]]);
	    x3 = &x3_start[4 * i];

	    le =  &left[cptr[i] * 16];
	    ri =  &right[cptr[i] * 16];

	    for(j = 0; j < 4; j++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;
		for(k = 0; k < 4; k++)
		  {
		    ump_x1 += x1[k] * le[j * 4 + k];
		    ump_x2 += x2[k] * ri[j * 4 + k];
		  }
		x1px2[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 4; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 4; j++)
	      for(k = 0; k < 4; k++)
		x3[k] += x1px2[j] * EV[j * 4 + k];	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[4 * tipX1[i]]);
	    x2 = &x2_start[4 * i];
	    x3 = &x3_start[4 * i];

	    le =  &left[cptr[i] * 16];
	    ri =  &right[cptr[i] * 16];

	    for(j = 0; j < 4; j++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;
		for(k = 0; k < 4; k++)
		  {
		    ump_x1 += x1[k] * le[j * 4 + k];
		    ump_x2 += x2[k] * ri[j * 4 + k];
		  }
		x1px2[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 4; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 4; j++)
	      for(k = 0; k < 4; k++)
		x3[k] +=  x1px2[j] *  EV[4 * j + k];	    

	    scale = 1;
	    for(j = 0; j < 4 && scale; j++)
	      scale = (x3[j] < minlikelihood_FLOAT && x3[j] > minusminlikelihood_FLOAT);

	    if(scale)
	      {
		for(j = 0; j < 4; j++)
		  x3[j] *= twotothe256_FLOAT;
		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];
	  x3 = &x3_start[4 * i];

	  le = &left[cptr[i] * 16];
	  ri = &right[cptr[i] * 16];

	  for(j = 0; j < 4; j++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;
	      for(k = 0; k < 4; k++)
		{
		  ump_x1 += x1[k] * le[j * 4 + k];
		  ump_x2 += x2[k] * ri[j * 4 + k];
		}
	      x1px2[j] = ump_x1 * ump_x2;
	    }

	  for(j = 0; j < 4; j++)
	    x3[j] = 0.0;

	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      x3[k] +=  x1px2[j] *  EV[4 * j + k];	  

	  scale = 1;
	  for(j = 0; j < 4 && scale; j++)
	    scale = (x3[j] < minlikelihood_FLOAT && x3[j] > minusminlikelihood_FLOAT);

	  if(scale)
	    {
	      for(j = 0; j < 4; j++)
		x3[j] *= twotothe256_FLOAT;

	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;	     
	    }
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}

#ifndef __SIM_SSE3

static void newviewGTRCAT( int tipCase,  double *EV,  int *cptr,
			   double *x1_start,  double *x2_start,  double *x3_start,  double *tipVector,
			   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			   int n,  double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le,
    *ri,
    *x1, *x2, *x3;
  double
    ump_x1, ump_x2, x1px2[4];
  int i, j, k, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[4 * tipX1[i]]);
	    x2 = &(tipVector[4 * tipX2[i]]);
	    x3 = &x3_start[4 * i];

	    le =  &left[cptr[i] * 16];
	    ri =  &right[cptr[i] * 16];

	    for(j = 0; j < 4; j++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;
		for(k = 0; k < 4; k++)
		  {
		    ump_x1 += x1[k] * le[j * 4 + k];
		    ump_x2 += x2[k] * ri[j * 4 + k];
		  }
		x1px2[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 4; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 4; j++)
	      for(k = 0; k < 4; k++)
		x3[k] += x1px2[j] * EV[j * 4 + k];	   
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[4 * tipX1[i]]);
	    x2 = &x2_start[4 * i];
	    x3 = &x3_start[4 * i];

	    le =  &left[cptr[i] * 16];
	    ri =  &right[cptr[i] * 16];

	    for(j = 0; j < 4; j++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;
		for(k = 0; k < 4; k++)
		  {
		    ump_x1 += x1[k] * le[j * 4 + k];
		    ump_x2 += x2[k] * ri[j * 4 + k];
		  }
		x1px2[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 4; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 4; j++)
	      for(k = 0; k < 4; k++)
		x3[k] +=  x1px2[j] *  EV[4 * j + k];	   

	    scale = 1;
	    for(j = 0; j < 4 && scale; j++)
	      scale = (x3[j] < minlikelihood && x3[j] > minusminlikelihood);	    	   
	    	    
	    if(scale)
	      {		    
		for(j = 0; j < 4; j++)
		  x3[j] *= twotothe256;
		
		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;		
	      }	     
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];
	  x3 = &x3_start[4 * i];

	  le = &left[cptr[i] * 16];
	  ri = &right[cptr[i] * 16];

	  for(j = 0; j < 4; j++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;
	      for(k = 0; k < 4; k++)
		{
		  ump_x1 += x1[k] * le[j * 4 + k];
		  ump_x2 += x2[k] * ri[j * 4 + k];
		}
	      x1px2[j] = ump_x1 * ump_x2;
	    }

	  for(j = 0; j < 4; j++)
	    x3[j] = 0.0;

	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      x3[k] +=  x1px2[j] *  EV[4 * j + k];
	
	  scale = 1;
	  for(j = 0; j < 4 && scale; j++)
	    scale = (x3[j] < minlikelihood && x3[j] > minusminlikelihood);

	  if(scale)
	    {		    
	      for(j = 0; j < 4; j++)
		x3[j] *= twotothe256;
	      
	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;		
	    }	  
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}

#else

static void newviewGTRCAT( int tipCase,  double *EV,  int *cptr,
			   double *x1_start,  double *x2_start,  double *x3_start,  double *tipVector,
			   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			   int n,  double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le,
    *ri,
    *x1, *x2, *x3;
  double
    ump_x1, ump_x2, x1px2[4] __attribute__ ((aligned (16)));
  int i, j, k, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {	   
	    x1 = &(tipVector[4 * tipX1[i]]);
	    x2 = &(tipVector[4 * tipX2[i]]);
	    x3 = &x3_start[4 * i];

	    le =  &left[cptr[i] * 16];
	    ri =  &right[cptr[i] * 16];

	    for(j = 0; j < 4; j++)
	      {
		__m128d u1v = _mm_setzero_pd();
		__m128d u2v = _mm_setzero_pd();
		double *ll = &le[j * 4];
		double *rr = &ri[j * 4];
		
		for(k = 0; k < 4; k+=2)
		  {
		    u1v = _mm_add_pd(u1v, _mm_mul_pd(_mm_load_pd(&x1[k]), _mm_load_pd(&ll[k])));
		    u2v = _mm_add_pd(u2v, _mm_mul_pd(_mm_load_pd(&x2[k]), _mm_load_pd(&rr[k])));		   
		  }
		u1v = _mm_hadd_pd(u1v, u1v);
		u2v = _mm_hadd_pd(u2v, u2v);
		u1v = _mm_mul_pd(u1v, u2v);
		
		_mm_storel_pd(&x1px2[j], u1v);
	      }

	    for(j = 0; j < 4; j+=2)
	      _mm_store_pd(&x3[j], _mm_setzero_pd());	    

	    for(j = 0; j < 4; j++)
	      {
		__m128d xv = _mm_set1_pd(x1px2[j]);
		for(k = 0; k < 4; k+=2)
		{
		  __m128d x3v = _mm_load_pd(&x3[k]);
		  x3v = _mm_add_pd(x3v, _mm_mul_pd(xv, _mm_load_pd(&EV[j * 4 + k])));
		  
		  _mm_store_pd(&x3[k], x3v);
		}	       
	      }
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    x1 = &(tipVector[4 * tipX1[i]]);
	    x2 = &x2_start[4 * i];
	    x3 = &x3_start[4 * i];

	    le =  &left[cptr[i] * 16];
	    ri =  &right[cptr[i] * 16];

	    for(j = 0; j < 4; j++)
	      {
		__m128d u1v = _mm_setzero_pd();
		__m128d u2v = _mm_setzero_pd();
		double *ll = &le[j * 4];
		double *rr = &ri[j * 4];

		for(k = 0; k < 4; k+=2)
		  {
		    u1v = _mm_add_pd(u1v, _mm_mul_pd(_mm_load_pd(&x1[k]), _mm_load_pd(&ll[k])));
		    u2v = _mm_add_pd(u2v, _mm_mul_pd(_mm_load_pd(&x2[k]), _mm_load_pd(&rr[k])));		   
		  }
		u1v = _mm_hadd_pd(u1v, u1v);
		u2v = _mm_hadd_pd(u2v, u2v);
		u1v = _mm_mul_pd(u1v, u2v);
		
		_mm_storel_pd(&x1px2[j], u1v);
	      }

	    for(j = 0; j < 4; j+=2)
	      _mm_store_pd(&x3[j], _mm_setzero_pd());	    

	    for(j = 0; j < 4; j++)
	      {
		__m128d xv = _mm_set1_pd(x1px2[j]);
		for(k = 0; k < 4; k+=2)
		{
		  __m128d x3v = _mm_load_pd(&x3[k]);
		  x3v = _mm_add_pd(x3v, _mm_mul_pd(xv, _mm_load_pd(&EV[j * 4 + k])));
		  
		  _mm_store_pd(&x3[k], x3v);
		}	       
	      }	   
	   
	    __m128d minlikelihood_sse = _mm_set1_pd( minlikelihood );
	      
	    scale = 1;
	    for(j = 0; scale && j < 4; j += 2)
	      {
		__m128d vv = _mm_load_pd(&x3[j]);
		__m128d v1 = _mm_and_pd(vv, absMask.m);
		v1 = _mm_cmplt_pd(v1,  minlikelihood_sse);
		if(_mm_movemask_pd( v1 ) != 3)
		  scale = 0;
	      }	    	  

	    	    
	    if(scale)
	      {		   
		__m128d sc = _mm_set1_pd(twotothe256);

		for(j = 0; j < 4; j+=2)		
		  _mm_store_pd(&x3[j], _mm_mul_pd(_mm_load_pd(&x3[j]), sc));
    	    	
		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;		
	      }	     
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];
	  x3 = &x3_start[4 * i];

	  le = &left[cptr[i] * 16];
	  ri = &right[cptr[i] * 16];

	  for(j = 0; j < 4; j++)
	    {
	      __m128d u1v = _mm_setzero_pd();
	      __m128d u2v = _mm_setzero_pd();
	      double *ll = &le[j * 4];
	      double *rr = &ri[j * 4];

	      for(k = 0; k < 4; k+=2)
		{
		  u1v = _mm_add_pd(u1v, _mm_mul_pd(_mm_load_pd(&x1[k]), _mm_load_pd(&ll[k])));
		  u2v = _mm_add_pd(u2v, _mm_mul_pd(_mm_load_pd(&x2[k]), _mm_load_pd(&rr[k])));		   
		}	      	     

	      u1v = _mm_hadd_pd(u1v, u1v);
	      u2v = _mm_hadd_pd(u2v, u2v);
	      u1v = _mm_mul_pd(u1v, u2v);
		
	      _mm_storel_pd(&x1px2[j], u1v);
	    }
	  
	  for(j = 0; j < 4; j+=2)
	    _mm_store_pd(&x3[j], _mm_setzero_pd());	    
	  
	  for(j = 0; j < 4; j++)
	    {
	      __m128d xv = _mm_set1_pd(x1px2[j]);
	      
	      for(k = 0; k < 4; k+=2)
		{
		  __m128d x3v = _mm_load_pd(&x3[k]);
		  x3v = _mm_add_pd(x3v, _mm_mul_pd(xv, _mm_load_pd(&EV[j * 4 + k])));
		  
		  _mm_store_pd(&x3[k], x3v);
		}	       
	    }	   

	  __m128d minlikelihood_sse = _mm_set1_pd( minlikelihood );
	  
	  scale = 1;
	  for(j = 0; scale && j < 4; j += 2)
	    {
	      __m128d vv = _mm_load_pd(&x3[j]);
	      __m128d v1 = _mm_and_pd(vv, absMask.m);
	      v1 = _mm_cmplt_pd(v1,  minlikelihood_sse);
	      if(_mm_movemask_pd( v1 ) != 3)
		scale = 0;
	    }	    	  
	  
	  if(scale)
	    {	
	      __m128d sc = _mm_set1_pd(twotothe256);	      

	      for(j = 0; j < 4; j+=2)		
		_mm_store_pd(&x3[j], _mm_mul_pd(_mm_load_pd(&x3[j]), sc));    	      	      
	      
	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;		
	    }	  
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}



#endif





static void newviewGTRGAMMA(int tipCase,
			    double *x1_start, double *x2_start, double *x3_start,
			    double *EV, double *tipVector,
			    int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			    const int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling
			    )
{
  int i, j, k, l, scale, addScale = 0;
  double
    *x1,
    *x2,
    *x3,
    buf,
    ump_x1,
    ump_x2;

#ifndef __SIM_SSE3
  double x1px2[4];
#else
  
  double 
    x1px2[4] __attribute__ ((aligned (16))),
    EV_t[16] __attribute__ ((aligned (16)));    

  for(k = 0; k < 4; k++)
    for (l=0; l < 4; l++)
      EV_t[4 * l + k] = EV[4 * k + l];
#endif


  switch(tipCase)
    {
    case TIP_TIP:
      {
	double *uX1, umpX1[256], *uX2, umpX2[256];

#ifndef _SIM_SSE3
	for(i = 1; i < 16; i++)
	  {
	    x1 = &(tipVector[i * 4]);

	    for(j=0; j<4; j++)
	      for(k=0; k<4; k++)
		{
		  umpX1[i*16 + j*4 + k] = 0.0;
		  umpX2[i*16 + j*4 + k] = 0.0;

		  for (l=0; l < 4; l++)
		    {
		      umpX1[i*16 + j*4 + k] += x1[l] * left[j*16 + k*4 + l];
		      umpX2[i*16 + j*4 + k] += x1[l] * right[j*16 + k*4 + l];
		    }
		}
	  }
#else
	for (i = 1; i < 16; i++)
	  {
	    __m128d x1_1 = _mm_load_pd(&(tipVector[i*4]));
	    __m128d x1_2 = _mm_load_pd(&(tipVector[i*4 + 2]));	   

	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{		 
		  __m128d left1 = _mm_load_pd(&left[j*16 + k*4]);
		  __m128d left2 = _mm_load_pd(&left[j*16 + k*4 + 2]);
		  
		  __m128d acc = _mm_setzero_pd();

		  acc = _mm_add_pd(acc, _mm_mul_pd(left1, x1_1));
		  acc = _mm_add_pd(acc, _mm_mul_pd(left2, x1_2));
		  		  
		  acc = _mm_hadd_pd(acc, acc);
		  _mm_storel_pd(&umpX1[i*16 + j*4 + k], acc);
		}
	  
	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{
		  __m128d left1 = _mm_load_pd(&right[j*16 + k*4]);
		  __m128d left2 = _mm_load_pd(&right[j*16 + k*4 + 2]);
		  
		  __m128d acc = _mm_setzero_pd();

		  acc = _mm_add_pd(acc, _mm_mul_pd(left1, x1_1));
		  acc = _mm_add_pd(acc, _mm_mul_pd(left2, x1_2));
		  		  
		  acc = _mm_hadd_pd(acc, acc);
		  _mm_storel_pd(&umpX2[i*16 + j*4 + k], acc);
		 
		}
	  }
#endif

	for (i = 0; i < n; i++)
	  {
	    x3 = &x3_start[i * 16];

	    uX1 = &umpX1[16 * tipX1[i]];
	    uX2 = &umpX2[16 * tipX2[i]];

	    for(j = 0; j < 16; j++)
	      x3[j] = 0.0;

	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{
		  buf = uX1[j*4 + k] * uX2[j*4 + k];

		  for (l=0; l<4; l++)
		    x3[j * 4 + l] +=  buf * EV[4 * k + l];
		}	   
	  }
      }
      break;
    case TIP_INNER:
      {	
	double *uX1, umpX1[256];

#ifndef __SIM_SSE3
	for (i = 1; i < 16; i++)
	  {
	    x1 = &(tipVector[i*4]);

	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{
		  umpX1[i*16 + j*4 + k] = 0.0;
		  for (l=0; l < 4; l++)
		    umpX1[i*16 + j*4 + k] += x1[l] * left[j*16 + k*4 + l];
		}
	  }
#else	
	for (i = 1; i < 16; i++)
	  {
	    __m128d x1_1 = _mm_load_pd(&(tipVector[i*4]));
	    __m128d x1_2 = _mm_load_pd(&(tipVector[i*4 + 2]));	   

	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{		 
		  __m128d left1 = _mm_load_pd(&left[j*16 + k*4]);
		  __m128d left2 = _mm_load_pd(&left[j*16 + k*4 + 2]);
		  
		  __m128d acc = _mm_setzero_pd();

		  acc = _mm_add_pd(acc, _mm_mul_pd(left1, x1_1));
		  acc = _mm_add_pd(acc, _mm_mul_pd(left2, x1_2));
		  		  
		  acc = _mm_hadd_pd(acc, acc);
		  _mm_storel_pd(&umpX1[i*16 + j*4 + k], acc);		 
		}
	  }
#endif

	 for (i = 0; i < n; i++)
	   {
	     x2 = &x2_start[i * 16];
	     x3 = &x3_start[i * 16];

	     uX1 = &umpX1[16 * tipX1[i]];

	     for(j = 0; j < 16; j++)
	       x3[j] = 0.0;

	     for (j = 0; j < 4; j++)
	       {
#ifndef __SIM_SSE3

		 for (k = 0; k < 4; k++)
		   {
		     ump_x2 = 0.0;

		     for (l=0; l<4; l++)
		       ump_x2 += x2[j*4 + l] * right[j* 16 + k*4 + l];
		     x1px2[k] = uX1[j * 4 + k] * ump_x2;
		   }

		 for(k = 0; k < 4; k++)
		   for (l=0; l<4; l++)
		     x3[j * 4 + l] +=  x1px2[k] * EV[4 * k + l];

#else
		 //
		 // multiply/add right side
		 //
		 double *x2_p = &x2[j*4];
		 double *right_k0_p = &right[j*16];
		 double *right_k1_p = &right[j*16 + 1*4];
		 double *right_k2_p = &right[j*16 + 2*4];
		 double *right_k3_p = &right[j*16 + 3*4];
		 __m128d x2_0 = _mm_load_pd( &x2_p[0] );
		 __m128d x2_2 = _mm_load_pd( &x2_p[2] );

		 __m128d right_k0_0 = _mm_load_pd( &right_k0_p[0] );
		 __m128d right_k0_2 = _mm_load_pd( &right_k0_p[2] );
		 __m128d right_k1_0 = _mm_load_pd( &right_k1_p[0] );
		 __m128d right_k1_2 = _mm_load_pd( &right_k1_p[2] );
		 __m128d right_k2_0 = _mm_load_pd( &right_k2_p[0] );
		 __m128d right_k2_2 = _mm_load_pd( &right_k2_p[2] );
		 __m128d right_k3_0 = _mm_load_pd( &right_k3_p[0] );
		 __m128d right_k3_2 = _mm_load_pd( &right_k3_p[2] );



		 right_k0_0 = _mm_mul_pd( x2_0, right_k0_0);
		 right_k0_2 = _mm_mul_pd( x2_2, right_k0_2);

		 right_k1_0 = _mm_mul_pd( x2_0, right_k1_0);
		 right_k1_2 = _mm_mul_pd( x2_2, right_k1_2);

		 right_k0_0 = _mm_hadd_pd( right_k0_0, right_k0_2);
		 right_k1_0 = _mm_hadd_pd( right_k1_0, right_k1_2);
		 right_k0_0 = _mm_hadd_pd( right_k0_0, right_k1_0);


		 right_k2_0 = _mm_mul_pd( x2_0, right_k2_0);
		 right_k2_2 = _mm_mul_pd( x2_2, right_k2_2);


		 right_k3_0 = _mm_mul_pd( x2_0, right_k3_0);
		 right_k3_2 = _mm_mul_pd( x2_2, right_k3_2);

		 right_k2_0 = _mm_hadd_pd( right_k2_0, right_k2_2);
		 right_k3_0 = _mm_hadd_pd( right_k3_0, right_k3_2);
		 right_k2_0 = _mm_hadd_pd( right_k2_0, right_k3_0);

		 {
		   //
		   // load left side from tip vector
		   //
		   
		   __m128d uX1_k0_sse = _mm_load_pd( &uX1[j * 4] );
		   __m128d uX1_k2_sse = _mm_load_pd( &uX1[j * 4 + 2] );
		 
		 
		   //
		   // multiply left * right
		   //
		   
		   __m128d x1px2_k0 = _mm_mul_pd( uX1_k0_sse, right_k0_0 );
		   __m128d x1px2_k2 = _mm_mul_pd( uX1_k2_sse, right_k2_0 );
		   
		   
		   //
		   // multiply with EV matrix (!?)
		   //
		   
		   __m128d EV_t_l0_k0 = _mm_load_pd( &EV_t[4 * 0 + 0]);
		   __m128d EV_t_l0_k2 = _mm_load_pd( &EV_t[4 * 0 + 2]);
		   __m128d EV_t_l1_k0 = _mm_load_pd( &EV_t[4 * 1 + 0]);
		   __m128d EV_t_l1_k2 = _mm_load_pd( &EV_t[4 * 1 + 2]);
		   __m128d EV_t_l2_k0 = _mm_load_pd( &EV_t[4 * 2 + 0]);
		   __m128d EV_t_l2_k2 = _mm_load_pd( &EV_t[4 * 2 + 2]);
		   __m128d EV_t_l3_k0 = _mm_load_pd( &EV_t[4 * 3 + 0]);
		   __m128d EV_t_l3_k2 = _mm_load_pd( &EV_t[4 * 3 + 2]);
		   
		   EV_t_l0_k0 = _mm_mul_pd( x1px2_k0, EV_t_l0_k0 );
		   EV_t_l0_k2 = _mm_mul_pd( x1px2_k2, EV_t_l0_k2 );
		   EV_t_l0_k0 = _mm_hadd_pd( EV_t_l0_k0, EV_t_l0_k2 );
		   
		   EV_t_l1_k0 = _mm_mul_pd( x1px2_k0, EV_t_l1_k0 );
		   EV_t_l1_k2 = _mm_mul_pd( x1px2_k2, EV_t_l1_k2 );
		   
		   EV_t_l1_k0 = _mm_hadd_pd( EV_t_l1_k0, EV_t_l1_k2 );
		   EV_t_l0_k0 = _mm_hadd_pd( EV_t_l0_k0, EV_t_l1_k0 );
		   
		   EV_t_l2_k0 = _mm_mul_pd( x1px2_k0, EV_t_l2_k0 );
		   EV_t_l2_k2 = _mm_mul_pd( x1px2_k2, EV_t_l2_k2 );
		   EV_t_l2_k0 = _mm_hadd_pd( EV_t_l2_k0, EV_t_l2_k2 );
		   		   
		   EV_t_l3_k0 = _mm_mul_pd( x1px2_k0, EV_t_l3_k0 );
		   EV_t_l3_k2 = _mm_mul_pd( x1px2_k2, EV_t_l3_k2 );
		   EV_t_l3_k0 = _mm_hadd_pd( EV_t_l3_k0, EV_t_l3_k2 );
		   
		   EV_t_l2_k0 = _mm_hadd_pd( EV_t_l2_k0, EV_t_l3_k0 );
		   
		   _mm_store_pd( &x3[j * 4 + 0], EV_t_l0_k0 );
		   _mm_store_pd( &x3[j * 4 + 2], EV_t_l2_k0 );
		 }
#endif
	       }	     

	     scale = 1;
	     for(l = 0; scale && (l < 16); l++)
	       scale = (ABS(x3[l]) <  minlikelihood);

	     if(scale)
	       {
		 for (l=0; l<16; l++)
		   x3[l] *= twotothe256;

		 if(useFastScaling)
		   addScale += wgt[i];
		 else
		   ex3[i]  += 1;		 
	       }

	   }
      }
      break;
    case INNER_INNER:

      /* C-OPT here we don't do any pre-computations
	 This should be the most compute intensive loop of the three
	 cases here. If we have one or two tips as descendants
	 we can take a couple of shortcuts */


     for (i = 0; i < n; i++)
       {
	 x1 = &x1_start[i * 16];
	 x2 = &x2_start[i * 16];
	 x3 = &x3_start[i * 16];

	 for(j = 0; j < 16; j++)
	   x3[j] = 0.0;



	 for (j = 0; j < 4; j++)
	   {

#ifndef __SIM_SSE3
	     for (k = 0; k < 4; k++)
	       {
		 ump_x1 = 0.0;
		 ump_x2 = 0.0;

		 for (l=0; l<4; l++)
		   {
		     ump_x1 += x1[j*4 + l] * left[j*16 + k*4 +l];
		     ump_x2 += x2[j*4 + l] * right[j*16 + k*4 +l];
		   }




		 x1px2[k] = ump_x1 * ump_x2;
	       }

	     for(k = 0; k < 4; k++)
	       for (l=0; l<4; l++)
	         x3[j * 4 + l] +=  x1px2[k] * EV[4 * k + l];


#else	     
	     //
	     // multiply/add left side
	     //
	     
	     double *x1_p = &x1[j*4];
	     double *left_k0_p = &left[j*16];
	     double *left_k1_p = &left[j*16 + 1*4];
	     double *left_k2_p = &left[j*16 + 2*4];
	     double *left_k3_p = &left[j*16 + 3*4];
	     
	     __m128d x1_0 = _mm_load_pd( &x1_p[0] );
	     __m128d x1_2 = _mm_load_pd( &x1_p[2] );
	     
	     __m128d left_k0_0 = _mm_load_pd( &left_k0_p[0] );
	     __m128d left_k0_2 = _mm_load_pd( &left_k0_p[2] );
	     __m128d left_k1_0 = _mm_load_pd( &left_k1_p[0] );
	     __m128d left_k1_2 = _mm_load_pd( &left_k1_p[2] );
	     __m128d left_k2_0 = _mm_load_pd( &left_k2_p[0] );
	     __m128d left_k2_2 = _mm_load_pd( &left_k2_p[2] );
	     __m128d left_k3_0 = _mm_load_pd( &left_k3_p[0] );
	     __m128d left_k3_2 = _mm_load_pd( &left_k3_p[2] );
	     
	     left_k0_0 = _mm_mul_pd(x1_0, left_k0_0);
	     left_k0_2 = _mm_mul_pd(x1_2, left_k0_2);
	     
	     left_k1_0 = _mm_mul_pd(x1_0, left_k1_0);
	     left_k1_2 = _mm_mul_pd(x1_2, left_k1_2);
	     
	     left_k0_0 = _mm_hadd_pd( left_k0_0, left_k0_2 );
	     left_k1_0 = _mm_hadd_pd( left_k1_0, left_k1_2);
	     left_k0_0 = _mm_hadd_pd( left_k0_0, left_k1_0);
	     
	     left_k2_0 = _mm_mul_pd(x1_0, left_k2_0);
	     left_k2_2 = _mm_mul_pd(x1_2, left_k2_2);
	     
	     left_k3_0 = _mm_mul_pd(x1_0, left_k3_0);
	     left_k3_2 = _mm_mul_pd(x1_2, left_k3_2);
	     
	     left_k2_0 = _mm_hadd_pd( left_k2_0, left_k2_2);
	     left_k3_0 = _mm_hadd_pd( left_k3_0, left_k3_2);
	     left_k2_0 = _mm_hadd_pd( left_k2_0, left_k3_0);
	     
	     
	     //
	     // multiply/add right side
	     //
	     double *x2_p = &x2[j*4];
	     double *right_k0_p = &right[j*16];
	     double *right_k1_p = &right[j*16 + 1*4];
	     double *right_k2_p = &right[j*16 + 2*4];
	     double *right_k3_p = &right[j*16 + 3*4];
	     __m128d x2_0 = _mm_load_pd( &x2_p[0] );
	     __m128d x2_2 = _mm_load_pd( &x2_p[2] );
	     
	     __m128d right_k0_0 = _mm_load_pd( &right_k0_p[0] );
	     __m128d right_k0_2 = _mm_load_pd( &right_k0_p[2] );
	     __m128d right_k1_0 = _mm_load_pd( &right_k1_p[0] );
	     __m128d right_k1_2 = _mm_load_pd( &right_k1_p[2] );
	     __m128d right_k2_0 = _mm_load_pd( &right_k2_p[0] );
	     __m128d right_k2_2 = _mm_load_pd( &right_k2_p[2] );
	     __m128d right_k3_0 = _mm_load_pd( &right_k3_p[0] );
	     __m128d right_k3_2 = _mm_load_pd( &right_k3_p[2] );
	     
	     right_k0_0 = _mm_mul_pd( x2_0, right_k0_0);
	     right_k0_2 = _mm_mul_pd( x2_2, right_k0_2);
	     
	     right_k1_0 = _mm_mul_pd( x2_0, right_k1_0);
	     right_k1_2 = _mm_mul_pd( x2_2, right_k1_2);
	     
	     right_k0_0 = _mm_hadd_pd( right_k0_0, right_k0_2);
	     right_k1_0 = _mm_hadd_pd( right_k1_0, right_k1_2);
	     right_k0_0 = _mm_hadd_pd( right_k0_0, right_k1_0);
	     
	     right_k2_0 = _mm_mul_pd( x2_0, right_k2_0);
	     right_k2_2 = _mm_mul_pd( x2_2, right_k2_2);
	     
	     
	     right_k3_0 = _mm_mul_pd( x2_0, right_k3_0);
	     right_k3_2 = _mm_mul_pd( x2_2, right_k3_2);
	     
	     right_k2_0 = _mm_hadd_pd( right_k2_0, right_k2_2);
	     right_k3_0 = _mm_hadd_pd( right_k3_0, right_k3_2);
	     right_k2_0 = _mm_hadd_pd( right_k2_0, right_k3_0);	   

             //
             // multiply left * right
             //

	     __m128d x1px2_k0 = _mm_mul_pd( left_k0_0, right_k0_0 );
	     __m128d x1px2_k2 = _mm_mul_pd( left_k2_0, right_k2_0 );


             //
             // multiply with EV matrix (!?)
             //

	    __m128d EV_t_l0_k0 = _mm_load_pd( &EV_t[4 * 0 + 0]);
	    __m128d EV_t_l0_k2 = _mm_load_pd( &EV_t[4 * 0 + 2]);
	    __m128d EV_t_l1_k0 = _mm_load_pd( &EV_t[4 * 1 + 0]);
	    __m128d EV_t_l1_k2 = _mm_load_pd( &EV_t[4 * 1 + 2]);
	    __m128d EV_t_l2_k0 = _mm_load_pd( &EV_t[4 * 2 + 0]);
	    __m128d EV_t_l2_k2 = _mm_load_pd( &EV_t[4 * 2 + 2]);
	    __m128d EV_t_l3_k0 = _mm_load_pd( &EV_t[4 * 3 + 0]);
	    __m128d EV_t_l3_k2 = _mm_load_pd( &EV_t[4 * 3 + 2]);

	    EV_t_l0_k0 = _mm_mul_pd( x1px2_k0, EV_t_l0_k0 );
	    EV_t_l0_k2 = _mm_mul_pd( x1px2_k2, EV_t_l0_k2 );
	    EV_t_l0_k0 = _mm_hadd_pd( EV_t_l0_k0, EV_t_l0_k2 );

	    EV_t_l1_k0 = _mm_mul_pd( x1px2_k0, EV_t_l1_k0 );
	    EV_t_l1_k2 = _mm_mul_pd( x1px2_k2, EV_t_l1_k2 );

	    EV_t_l1_k0 = _mm_hadd_pd( EV_t_l1_k0, EV_t_l1_k2 );
	    EV_t_l0_k0 = _mm_hadd_pd( EV_t_l0_k0, EV_t_l1_k0 );

	    EV_t_l2_k0 = _mm_mul_pd( x1px2_k0, EV_t_l2_k0 );
	    EV_t_l2_k2 = _mm_mul_pd( x1px2_k2, EV_t_l2_k2 );
	    EV_t_l2_k0 = _mm_hadd_pd( EV_t_l2_k0, EV_t_l2_k2 );


	    EV_t_l3_k0 = _mm_mul_pd( x1px2_k0, EV_t_l3_k0 );
            EV_t_l3_k2 = _mm_mul_pd( x1px2_k2, EV_t_l3_k2 );
            EV_t_l3_k0 = _mm_hadd_pd( EV_t_l3_k0, EV_t_l3_k2 );

            EV_t_l2_k0 = _mm_hadd_pd( EV_t_l2_k0, EV_t_l3_k0 );

            _mm_store_pd( &x3[j * 4 + 0], EV_t_l0_k0 );
            _mm_store_pd( &x3[j * 4 + 2], EV_t_l2_k0 );
#endif
           }	 

	 scale = 1;
	 for(l = 0; scale && (l < 16); l++)
	   scale = (ABS(x3[l]) <  minlikelihood);

	 if(scale)
	   {
	     for (l=0; l<16; l++)
	       x3[l] *= twotothe256;

	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;	   
	   }		      		
       }
   
     break;
    default:
      assert(0);
    }
  
  if(useFastScaling)
    *scalerIncrement = addScale;

}

//int timesCalled=0; //addFlag
/*
 * newviewGTRGAMMA_FLOAT(tInfo->tipCase,
                        x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].EV_FLOAT, tr->partitionData[model].tipVector_FLOAT,
			ex3, tipX1, tipX2,
			width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);
 */

static void newviewGTRGAMMA_FLOAT(int tipCase,
                                  float *x1_start, float *x2_start, float *x3_start,
                                  float *EV, float *tipVector,
                                  int *ex3, unsigned char *tipX1, unsigned char *tipX2,
                                  const int n, float *left, float *right, int *wgt, int *scalerIncrement, const boolean useFastScaling
                                  )
{
    //extern int timesCalled;
  float
    *x1, *x2, *x3;
  float
    buf,
    ump_x1,
    ump_x2;
  int i, j, k, l, scale, addScale = 0;
  //timesCalled++; //addFlag
#ifndef __SIM_SSE3
  float x1px2[4];
#else
  float 
    x1px2[4] __attribute__ ((aligned (16))),
    EV_t[16] __attribute__ ((aligned (16)));

  for(k = 0; k < 4; k++)
    for (l=0; l < 4; l++)
      EV_t[4 * l + k] = EV[4 * k + l];
#endif

  switch(tipCase)
    {
    case TIP_TIP:
      {
#ifndef __SIM_SSE3
        float *uX1, umpX1[256], *uX2, umpX2[256];        
#else 
	float 
	  *uX1, 
	  umpX1[256] __attribute__ ((aligned (16))), 
	  *uX2, 
	  umpX2[256] __attribute__ ((aligned (16)));	
#endif

        for(i = 1; i < 16; i++)
          {
            x1 = &(tipVector[i * 4]);

            for(j=0; j<4; j++)
              for(k=0; k<4; k++)
                {
                  umpX1[i*16 + j*4 + k] = 0.0;
                  umpX2[i*16 + j*4 + k] = 0.0;

                  for (l=0; l < 4; l++)
                    {
                      umpX1[i*16 + j*4 + k] += x1[l] * left[j*16 + k*4 + l];
                      umpX2[i*16 + j*4 + k] += x1[l] * right[j*16 + k*4 + l];
                    }
                }
          }      

        for (i = 0; i < n; i++)
          {
            x3 = &x3_start[i * 16];

            uX1 = &umpX1[16 * tipX1[i]];
            uX2 = &umpX2[16 * tipX2[i]];

            for(j = 0; j < 16; j++)
              x3[j] = 0.0;

            for (j = 0; j < 4; j++)
              {
#ifndef __SIM_SSE3
                for (k = 0; k < 4; k++)
                  {
                    buf = uX1[j*4 + k] * uX2[j*4 + k];

                    for (l=0; l<4; l++)
                      x3[j * 4 + l] +=  buf * EV[4 * k + l];

                  }
#else
                //
                // load left side from tip vector
                //

                __m128 uX1_sse = _mm_load_ps( &uX1[j * 4] );


                //
                // load right side from tip vector
                //

                __m128 uX2_sse = _mm_load_ps( &uX2[j * 4] );

                //
                // multiply left * right
                //

                __m128 x1px2 = _mm_mul_ps( uX1_sse, uX2_sse );


                //
                // multiply with EV matrix (!?)
                //

                __m128 EV_t_l0 = _mm_load_ps( &EV_t[4 * 0]);
                __m128 EV_t_l1 = _mm_load_ps( &EV_t[4 * 1]);
                __m128 EV_t_l2 = _mm_load_ps( &EV_t[4 * 2]);
                __m128 EV_t_l3 = _mm_load_ps( &EV_t[4 * 3]);

                EV_t_l0 = _mm_mul_ps( x1px2, EV_t_l0 );
                EV_t_l1 = _mm_mul_ps( x1px2, EV_t_l1 );
                EV_t_l2 = _mm_mul_ps( x1px2, EV_t_l2 );
                EV_t_l3 = _mm_mul_ps( x1px2, EV_t_l3 );

                EV_t_l0 = _mm_hadd_ps( EV_t_l0, EV_t_l1 );
                EV_t_l2 = _mm_hadd_ps( EV_t_l2, EV_t_l3 );
                EV_t_l0 = _mm_hadd_ps( EV_t_l0, EV_t_l2 );

                _mm_store_ps( &x3[j * 4], EV_t_l0 );
#endif
              }            
          }
      }
      break;
    case TIP_INNER:
      {
#ifndef __SIM_SSE3	
        float *uX1, umpX1[256];       
#else
	float 
	  *uX1, 
	  umpX1[256] __attribute__ ((aligned (16)));       
#endif

        for (i = 1; i < 16; i++)
          {
            x1 = &(tipVector[i*4]);

            for (j = 0; j < 4; j++)
              for (k = 0; k < 4; k++)
                {
                  umpX1[i*16 + j*4 + k] = 0.0;
                  for (l=0; l < 4; l++)
                    umpX1[i*16 + j*4 + k] += x1[l] * left[j*16 + k*4 + l];
                }
          }       

         for (i = 0; i < n; i++)
           {
             x2 = &x2_start[i * 16];
             x3 = &x3_start[i * 16];

             uX1 = &umpX1[16 * tipX1[i]];

             for(j = 0; j < 16; j++)
               x3[j] = 0.0;

             for (j = 0; j < 4; j++)
               {

#ifndef __SIM_SSE3
                 for (k = 0; k < 4; k++)
                   {
                     ump_x2 = 0.0;

                     for (l=0; l<4; l++)
                       ump_x2 += x2[j*4 + l] * right[j* 16 + k*4 + l];
                     x1px2[k] = uX1[j * 4 + k] * ump_x2;
                   }

                 for(k = 0; k < 4; k++)
                   for (l=0; l<4; l++)
                     x3[j * 4 + l] +=  x1px2[k] * EV[4 * k + l];
#else
                 //
                 // multiply/add right side
                 //

                 float *right_k0_p = &right[j*16];
                 float *right_k1_p = &right[j*16 + 1*4];
                 float *right_k2_p = &right[j*16 + 2*4];
                 float *right_k3_p = &right[j*16 + 3*4];

                 float *x2_p = &x2[j*4];

                 __m128 x2 = _mm_load_ps( &x2_p[0] );
                 __m128 right_k0 = _mm_load_ps( &right_k0_p[0] );
                 __m128 right_k1 = _mm_load_ps( &right_k1_p[0] );
                 __m128 right_k2 = _mm_load_ps( &right_k2_p[0] );
                 __m128 right_k3 = _mm_load_ps( &right_k3_p[0] );


                 right_k0 = _mm_mul_ps( x2, right_k0);
                 right_k1 = _mm_mul_ps( x2, right_k1);

                 right_k2 = _mm_mul_ps( x2, right_k2);
                 right_k3 = _mm_mul_ps( x2, right_k3);

                 right_k0 = _mm_hadd_ps( right_k0, right_k1);
                 right_k2 = _mm_hadd_ps( right_k2, right_k3);
                 right_k0 = _mm_hadd_ps( right_k0, right_k2);


                 //
                 // load left side from tip vector
                 //

                 __m128 uX1_sse = _mm_load_ps( &uX1[j * 4] );


                 //
                 // multiply left * right
                 //

                 __m128 x1px2 = _mm_mul_ps( uX1_sse, right_k0 );


                 //
                 // multiply with EV matrix (!?)
                 //

                 __m128 EV_t_l0 = _mm_load_ps( &EV_t[4 * 0]);
                 __m128 EV_t_l1 = _mm_load_ps( &EV_t[4 * 1]);
                 __m128 EV_t_l2 = _mm_load_ps( &EV_t[4 * 2]);
                 __m128 EV_t_l3 = _mm_load_ps( &EV_t[4 * 3]);

                 EV_t_l0 = _mm_mul_ps( x1px2, EV_t_l0 );
                 EV_t_l1 = _mm_mul_ps( x1px2, EV_t_l1 );
                 EV_t_l2 = _mm_mul_ps( x1px2, EV_t_l2 );
                 EV_t_l3 = _mm_mul_ps( x1px2, EV_t_l3 );

                 EV_t_l0 = _mm_hadd_ps( EV_t_l0, EV_t_l1 );
                 EV_t_l2 = _mm_hadd_ps( EV_t_l2, EV_t_l3 );
                 EV_t_l0 = _mm_hadd_ps( EV_t_l0, EV_t_l2 );

                 _mm_store_ps( &x3[j * 4], EV_t_l0 );
#endif
               }             


#ifndef __SIM_SSE3

             scale = 1;
             for(l = 0; scale && (l < 16); l++)
               scale = (ABS(x3[l]) <  minlikelihood_FLOAT);

             if(scale)
               {
                 for (l=0; l<16; l++)
                   x3[l] *= twotothe256_FLOAT;

		 if(useFastScaling)
		   addScale += wgt[i];
		 else
		   ex3[i]  += 1;             
               }


#else
             __m128 zero = _mm_set1_ps(0.0);

             __m128 x3_0 = _mm_load_ps(&x3[4 * 0]);
             __m128 x3_1 = _mm_load_ps(&x3[4 * 1]);
             __m128 x3_2 = _mm_load_ps(&x3[4 * 2]);
             __m128 x3_3 = _mm_load_ps(&x3[4 * 3]);

             __m128 ax3_0 = _mm_sub_ps(zero, x3_0);
             __m128 ax3_1 = _mm_sub_ps(zero, x3_1);
             __m128 ax3_2 = _mm_sub_ps(zero, x3_2);
             __m128 ax3_3 = _mm_sub_ps(zero, x3_3);

             ax3_0 = _mm_max_ps(ax3_0, x3_0);
             ax3_1 = _mm_max_ps(ax3_1, x3_1);
             ax3_2 = _mm_max_ps(ax3_2, x3_2);
             ax3_3 = _mm_max_ps(ax3_3, x3_3);


             ax3_0 = _mm_max_ps( ax3_0, ax3_1 );
             ax3_2 = _mm_max_ps( ax3_2, ax3_3 );
             ax3_0 = _mm_max_ps( ax3_0, ax3_2 );

             __m128 minlikelihood_FLOAT_sse = _mm_set1_ps( minlikelihood_FLOAT );
             ax3_0 = _mm_cmplt_ps( ax3_0, minlikelihood_FLOAT_sse );

             int scale_bits = _mm_movemask_ps(ax3_0);


             if( scale_bits == 15 )
               {
                 __m128 twotothe256_FLOAT_sse = _mm_set1_ps( twotothe256_FLOAT );
                 x3_0 = _mm_mul_ps( x3_0, twotothe256_FLOAT_sse );
                 x3_1 = _mm_mul_ps( x3_1, twotothe256_FLOAT_sse );
                 x3_2 = _mm_mul_ps( x3_2, twotothe256_FLOAT_sse );
                 x3_3 = _mm_mul_ps( x3_3, twotothe256_FLOAT_sse );

                 _mm_store_ps( &x3[4 * 0], x3_0);
                 _mm_store_ps( &x3[4 * 1], x3_1);
                 _mm_store_ps( &x3[4 * 2], x3_2);
                 _mm_store_ps( &x3[4 * 3], x3_3);

		 if(useFastScaling)
		   addScale += wgt[i];
		 else
		   ex3[i]  += 1;              
               }

    #endif

           }
      }
      break;
    case INNER_INNER:    
     for (i = 0; i < n; i++)
       {
         x1 = &x1_start[i * 16];
         x2 = &x2_start[i * 16];
         x3 = &x3_start[i * 16];

         for(j = 0; j < 16; j++)
           x3[j] = 0.0;

         for (j = 0; j < 4; j++)
           {

#ifndef __SIM_SSE3
             for (k = 0; k < 4; k++)
               {
                 ump_x1 = 0.0;
                 ump_x2 = 0.0;

                 for (l=0; l<4; l++)
                   {
                     ump_x1 += x1[j*4 + l] * left[j*16 + k*4 +l];
                     ump_x2 += x2[j*4 + l] * right[j*16 + k*4 +l];
                   }

                 x1px2[k] = ump_x1 * ump_x2;
               }




             for(k = 0; k < 4; k++)
               for (l=0; l<4; l++)
                 x3[j * 4 + l] +=  x1px2[k] * EV[4 * k + l];


#else
             //
             // multiply/add left side
             //

             float *x1_p = &x1[j*4];

             float *left_k0_p = &left[j*16];
             float *left_k1_p = &left[j*16 + 1*4];
             float *left_k2_p = &left[j*16 + 2*4];
             float *left_k3_p = &left[j*16 + 3*4];


             __m128 x1 = _mm_load_ps( &x1_p[0] );
             __m128 left_k0 = _mm_load_ps( &left_k0_p[0] );
             __m128 left_k1 = _mm_load_ps( &left_k1_p[0] );
             __m128 left_k2 = _mm_load_ps( &left_k2_p[0] );
             __m128 left_k3 = _mm_load_ps( &left_k3_p[0] );

             left_k0 = _mm_mul_ps(x1, left_k0);
             left_k1 = _mm_mul_ps(x1, left_k1);

             left_k2 = _mm_mul_ps(x1, left_k2);
             left_k3 = _mm_mul_ps(x1, left_k3);

             left_k0 = _mm_hadd_ps( left_k0, left_k1);
             left_k2 = _mm_hadd_ps( left_k2, left_k3);
             left_k0 = _mm_hadd_ps( left_k0, left_k2);

             //
             // multiply/add right side
             //

             float *right_k0_p = &right[j*16];
             float *right_k1_p = &right[j*16 + 1*4];
             float *right_k2_p = &right[j*16 + 2*4];
             float *right_k3_p = &right[j*16 + 3*4];

             float *x2_p = &x2[j*4];

             __m128 x2 = _mm_load_ps( &x2_p[0] );
             __m128 right_k0 = _mm_load_ps( &right_k0_p[0] );
             __m128 right_k1 = _mm_load_ps( &right_k1_p[0] );
             __m128 right_k2 = _mm_load_ps( &right_k2_p[0] );
             __m128 right_k3 = _mm_load_ps( &right_k3_p[0] );


             right_k0 = _mm_mul_ps( x2, right_k0);
             right_k1 = _mm_mul_ps( x2, right_k1);

             right_k2 = _mm_mul_ps( x2, right_k2);
             right_k3 = _mm_mul_ps( x2, right_k3);

             right_k0 = _mm_hadd_ps( right_k0, right_k1);
             right_k2 = _mm_hadd_ps( right_k2, right_k3);
             right_k0 = _mm_hadd_ps( right_k0, right_k2);

             //
             // multiply left * right
             //

             __m128 x1px2 = _mm_mul_ps( left_k0, right_k0 );


             //
             // multiply with EV matrix (!?)
             //

             __m128 EV_t_l0 = _mm_load_ps( &EV_t[4 * 0]);
             __m128 EV_t_l1 = _mm_load_ps( &EV_t[4 * 1]);
             __m128 EV_t_l2 = _mm_load_ps( &EV_t[4 * 2]);
             __m128 EV_t_l3 = _mm_load_ps( &EV_t[4 * 3]);

             EV_t_l0 = _mm_mul_ps( x1px2, EV_t_l0 );
             EV_t_l1 = _mm_mul_ps( x1px2, EV_t_l1 );
             EV_t_l2 = _mm_mul_ps( x1px2, EV_t_l2 );
             EV_t_l3 = _mm_mul_ps( x1px2, EV_t_l3 );

             EV_t_l0 = _mm_hadd_ps( EV_t_l0, EV_t_l1 );
             EV_t_l2 = _mm_hadd_ps( EV_t_l2, EV_t_l3 );
             EV_t_l0 = _mm_hadd_ps( EV_t_l0, EV_t_l2 );


             _mm_store_ps( &x3[j * 4], EV_t_l0 );

#endif

           }

        
         scale = 1;


#ifndef __SIM_SSE3

         for(l = 0; scale && (l < 16); l++)
           scale = (ABS(x3[l]) <  minlikelihood_FLOAT);

         if(scale)
           {
             for (l=0; l<16; l++)
               x3[l] *= twotothe256_FLOAT;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;            
           }
#else


         __m128 zero = _mm_set1_ps(0.0);

          __m128 x3_0 = _mm_load_ps(&x3[4 * 0]);
          __m128 x3_1 = _mm_load_ps(&x3[4 * 1]);
          __m128 x3_2 = _mm_load_ps(&x3[4 * 2]);
          __m128 x3_3 = _mm_load_ps(&x3[4 * 3]);

          __m128 ax3_0 = _mm_sub_ps(zero, x3_0);
          __m128 ax3_1 = _mm_sub_ps(zero, x3_1);
          __m128 ax3_2 = _mm_sub_ps(zero, x3_2);
          __m128 ax3_3 = _mm_sub_ps(zero, x3_3);

          ax3_0 = _mm_max_ps(ax3_0, x3_0);
          ax3_1 = _mm_max_ps(ax3_1, x3_1);
          ax3_2 = _mm_max_ps(ax3_2, x3_2);
          ax3_3 = _mm_max_ps(ax3_3, x3_3);


          ax3_0 = _mm_max_ps( ax3_0, ax3_1 );
          ax3_2 = _mm_max_ps( ax3_2, ax3_3 );
          ax3_0 = _mm_max_ps( ax3_0, ax3_2 );

          __m128 minlikelihood_FLOAT_sse = _mm_set1_ps( minlikelihood_FLOAT );
          ax3_0 = _mm_cmplt_ps( ax3_0, minlikelihood_FLOAT_sse );

//          unsigned int scaletest[4] __attribute__ ((aligned (16)));
//          _mm_store_ps( (float*) scaletest, ax3_0);

          int scale_bits = _mm_movemask_ps(ax3_0);

//          if(scaletest[0] && scaletest[1] && scaletest[2] && scaletest[3] )
          if( scale_bits == 15 )
            {
              __m128 twotothe256_FLOAT_sse = _mm_set1_ps( twotothe256_FLOAT );
              x3_0 = _mm_mul_ps( x3_0, twotothe256_FLOAT_sse );
              x3_1 = _mm_mul_ps( x3_1, twotothe256_FLOAT_sse );
              x3_2 = _mm_mul_ps( x3_2, twotothe256_FLOAT_sse );
              x3_3 = _mm_mul_ps( x3_3, twotothe256_FLOAT_sse );

              _mm_store_ps( &x3[4 * 0], x3_0);
              _mm_store_ps( &x3[4 * 1], x3_1);
              _mm_store_ps( &x3[4 * 2], x3_2);
              _mm_store_ps( &x3[4 * 3], x3_3);

	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;            
            }

#endif
       }
     break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}




static void newviewGTRCATPROT(int tipCase, double *extEV,
			      int *cptr,
			      double *x1, double *x2, double *x3, double *tipVector,
			      int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			      int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le, *ri, *v, *vl, *vr;
  double
    ump_x1, ump_x2, x1px2;
  int i, l, j, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 400];
	    ri = &right[cptr[i] * 400];

	    vl = &(tipVector[20 * tipX1[i]]);
	    vr = &(tipVector[20 * tipX2[i]]);
	    v  = &x3[20 * i];
#ifdef __SIM_SSE3
	    for(l = 0; l < 20; l+=2)
	      _mm_store_pd(&v[l], _mm_setzero_pd());	      		
#else
	    for(l = 0; l < 20; l++)
	      v[l] = 0.0;
#endif

	    for(l = 0; l < 20; l++)
	      {
#ifdef __SIM_SSE3
		__m128d x1v = _mm_setzero_pd();
		__m128d x2v = _mm_setzero_pd();	 
		double 
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];

		for(j = 0; j < 20; j+=2)
		  {
		    x1v = _mm_add_pd(x1v, _mm_mul_pd(_mm_load_pd(&vl[j]), _mm_load_pd(&lv[j])));		    
		    x2v = _mm_add_pd(x2v, _mm_mul_pd(_mm_load_pd(&vr[j]), _mm_load_pd(&rv[j])));
		  }

		x1v = _mm_hadd_pd(x1v, x1v);
		x2v = _mm_hadd_pd(x2v, x2v);

		x1v = _mm_mul_pd(x1v, x2v);
		
		for(j = 0; j < 20; j+=2)
		  {
		    __m128d vv = _mm_load_pd(&v[j]);
		    vv = _mm_add_pd(vv, _mm_mul_pd(x1v, _mm_load_pd(&ev[j])));
		    _mm_store_pd(&v[j], vv);
		  }		    
#else
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 20; j++)
		  {
		    ump_x1 += vl[j] * le[l * 20 + j];
		    ump_x2 += vr[j] * ri[l * 20 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 20; j++)
		  v[j] += x1px2 * extEV[l * 20 + j];
#endif
	      }	   
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 400];
	    ri = &right[cptr[i] * 400];

	    vl = &(tipVector[20 * tipX1[i]]);
	    vr = &x2[20 * i];
	    v  = &x3[20 * i];

#ifdef __SIM_SSE3
	    for(l = 0; l < 20; l+=2)
	      _mm_store_pd(&v[l], _mm_setzero_pd());	      		
#else
	    for(l = 0; l < 20; l++)
	      v[l] = 0.0;
#endif
	   

	    for(l = 0; l < 20; l++)
	      {
#ifdef __SIM_SSE3

		__m128d x1v = _mm_setzero_pd();
		__m128d x2v = _mm_setzero_pd();	
		double 
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];

		for(j = 0; j < 20; j+=2)
		  {
		    x1v = _mm_add_pd(x1v, _mm_mul_pd(_mm_load_pd(&vl[j]), _mm_load_pd(&lv[j])));		    
		    x2v = _mm_add_pd(x2v, _mm_mul_pd(_mm_load_pd(&vr[j]), _mm_load_pd(&rv[j])));
		  }

		x1v = _mm_hadd_pd(x1v, x1v);
		x2v = _mm_hadd_pd(x2v, x2v);

		x1v = _mm_mul_pd(x1v, x2v);
		
		for(j = 0; j < 20; j+=2)
		  {
		    __m128d vv = _mm_load_pd(&v[j]);
		    vv = _mm_add_pd(vv, _mm_mul_pd(x1v, _mm_load_pd(&ev[j])));
		    _mm_store_pd(&v[j], vv);
		  }		    
#else
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 20; j++)
		  {
		    ump_x1 += vl[j] * le[l * 20 + j];
		    ump_x2 += vr[j] * ri[l * 20 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 20; j++)
		  v[j] += x1px2 * extEV[l * 20 + j];
#endif
	      }
#ifdef __SIM_SSE3
	    { 	    
	      __m128d minlikelihood_sse = _mm_set1_pd( minlikelihood );
	      
	      scale = 1;
	      for(l = 0; scale && (l < 20); l += 2)
		{
		  __m128d vv = _mm_load_pd(&v[l]);
		  __m128d v1 = _mm_and_pd(vv, absMask.m);
		  v1 = _mm_cmplt_pd(v1,  minlikelihood_sse);
		  if(_mm_movemask_pd( v1 ) != 3)
		    scale = 0;
		}	    	  
	    }
#else
	    scale = 1;
	    for(l = 0; scale && (l < 20); l++)
	      scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	   
#endif 

	    if(scale)
	      {
#ifdef __SIM_SSE3
		__m128d twoto = _mm_set_pd(twotothe256, twotothe256);

		for(l = 0; l < 20; l+=2)
		  {
		    __m128d ex3v = _mm_load_pd(&v[l]);
		    _mm_store_pd(&v[l], _mm_mul_pd(ex3v,twoto));		    
		  }
#else
		for(l = 0; l < 20; l++)
		  v[l] *= twotothe256;
#endif

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 400];
	  ri = &right[cptr[i] * 400];

	  vl = &x1[20 * i];
	  vr = &x2[20 * i];
	  v = &x3[20 * i];

#ifdef __SIM_SSE3
	    for(l = 0; l < 20; l+=2)
	      _mm_store_pd(&v[l], _mm_setzero_pd());	      		
#else
	    for(l = 0; l < 20; l++)
	      v[l] = 0.0;
#endif
	 
	  for(l = 0; l < 20; l++)
	    {
#ifdef __SIM_SSE3
		__m128d x1v = _mm_setzero_pd();
		__m128d x2v = _mm_setzero_pd();
		double 
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];


		for(j = 0; j < 20; j+=2)
		  {
		    x1v = _mm_add_pd(x1v, _mm_mul_pd(_mm_load_pd(&vl[j]), _mm_load_pd(&lv[j])));		    
		    x2v = _mm_add_pd(x2v, _mm_mul_pd(_mm_load_pd(&vr[j]), _mm_load_pd(&rv[j])));
		  }

		x1v = _mm_hadd_pd(x1v, x1v);
		x2v = _mm_hadd_pd(x2v, x2v);

		x1v = _mm_mul_pd(x1v, x2v);
		
		for(j = 0; j < 20; j+=2)
		  {
		    __m128d vv = _mm_load_pd(&v[j]);
		    vv = _mm_add_pd(vv, _mm_mul_pd(x1v, _mm_load_pd(&ev[j])));
		    _mm_store_pd(&v[j], vv);
		  }		    
#else
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;

	      for(j = 0; j < 20; j++)
		{
		  ump_x1 += vl[j] * le[l * 20 + j];
		  ump_x2 += vr[j] * ri[l * 20 + j];
		}

	      x1px2 =  ump_x1 * ump_x2;

	      for(j = 0; j < 20; j++)
		v[j] += x1px2 * extEV[l * 20 + j];
#endif
	    }
#ifdef __SIM_SSE3
	    { 	    
	      __m128d minlikelihood_sse = _mm_set1_pd( minlikelihood );
	      
	      scale = 1;
	      for(l = 0; scale && (l < 20); l += 2)
		{
		  __m128d vv = _mm_load_pd(&v[l]);
		  __m128d v1 = _mm_and_pd(vv, absMask.m);
		  v1 = _mm_cmplt_pd(v1,  minlikelihood_sse);
		  if(_mm_movemask_pd( v1 ) != 3)
		    scale = 0;
		}	    	  
	    }
#else
	   scale = 1;
	   for(l = 0; scale && (l < 20); l++)
	     scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));
#endif	   

	   if(scale)
	     {
#ifdef __SIM_SSE3
	       __m128d twoto = _mm_set_pd(twotothe256, twotothe256);
	       
	       for(l = 0; l < 20; l+=2)
		 {
		   __m128d ex3v = _mm_load_pd(&v[l]);		  
		   _mm_store_pd(&v[l], _mm_mul_pd(ex3v,twoto));	
		 }		   		  
#else
	       for(l = 0; l < 20; l++)
		 v[l] *= twotothe256;
#endif

	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
	}
      break;
    default:
      assert(0);
    }
  
  if(useFastScaling)
    *scalerIncrement = addScale;

}

static void newviewGTRCATPROT_FLOAT(int tipCase, float *extEV,
				    int *cptr,
				    float *x1, float *x2, float *x3, float *tipVector,
				    int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				    int n, float *left, float *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  float
    *le, *ri, *v, *vl, *vr;
  float
    ump_x1, ump_x2, x1px2;
  int i, l, j, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 400];
	    ri = &right[cptr[i] * 400];

	    vl = &(tipVector[20 * tipX1[i]]);
	    vr = &(tipVector[20 * tipX2[i]]);
	    v  = &x3[20 * i];

#ifdef __SIM_SSE3
	    for(l = 0; l < 20; l+=4)
	      _mm_store_ps(&v[l], _mm_setzero_ps());	      		
#else
	    for(l = 0; l < 20; l++)
	      v[l] = 0.0;
#endif

	    for(l = 0; l < 20; l++)
	      {
#ifdef __SIM_SSE3
		__m128 x1v = _mm_setzero_ps();
		__m128 x2v = _mm_setzero_ps();	 
		float 
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];
		
		
		
		for(j = 0; j < 20; j+=4)
		  {
		    x1v = _mm_add_ps(x1v, _mm_mul_ps(_mm_load_ps(&vl[j]), _mm_load_ps(&lv[j])));		    
		    x2v = _mm_add_ps(x2v, _mm_mul_ps(_mm_load_ps(&vr[j]), _mm_load_ps(&rv[j])));
		  }
		
		x1v = _mm_hadd_ps(x1v, x1v);
		x1v = _mm_hadd_ps(x1v, x1v);
		x2v = _mm_hadd_ps(x2v, x2v);
		x2v = _mm_hadd_ps(x2v, x2v);
		
		x1v = _mm_mul_ps(x1v, x2v);
		
		for(j = 0; j < 20; j+=4)
		  {
		    __m128 vv = _mm_load_ps(&v[j]);
		    vv = _mm_add_ps(vv, _mm_mul_ps(x1v, _mm_load_ps(&ev[j])));
		    _mm_store_ps(&v[j], vv);
		  }	
#else
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 20; j++)
		  {
		    ump_x1 += vl[j] * le[l * 20 + j];
		    ump_x2 += vr[j] * ri[l * 20 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 20; j++)
		  v[j] += x1px2 * extEV[l * 20 + j];
#endif
	      }
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 400];
	    ri = &right[cptr[i] * 400];

	    vl = &(tipVector[20 * tipX1[i]]);
	    vr = &x2[20 * i];
	    v  = &x3[20 * i];

#ifdef __SIM_SSE3
	    for(l = 0; l < 20; l+=4)
	      _mm_store_ps(&v[l], _mm_setzero_ps());	      		
#else
	    for(l = 0; l < 20; l++)
	      v[l] = 0.0;
#endif
	   

	    for(l = 0; l < 20; l++)
	      {
#ifdef __SIM_SSE3

		__m128 x1v = _mm_setzero_ps();
		__m128 x2v = _mm_setzero_ps();	
		float 
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];

		for(j = 0; j < 20; j+=4)
		  {
		    x1v = _mm_add_ps(x1v, _mm_mul_ps(_mm_load_ps(&vl[j]), _mm_load_ps(&lv[j])));		    
		    x2v = _mm_add_ps(x2v, _mm_mul_ps(_mm_load_ps(&vr[j]), _mm_load_ps(&rv[j])));
		  }

		x1v = _mm_hadd_ps(x1v, x1v);
		x1v = _mm_hadd_ps(x1v, x1v);
		x2v = _mm_hadd_ps(x2v, x2v);
		x2v = _mm_hadd_ps(x2v, x2v);

		x1v = _mm_mul_ps(x1v, x2v);
		
		for(j = 0; j < 20; j+=4)
		  {
		    __m128 vv = _mm_load_ps(&v[j]);
		    vv = _mm_add_ps(vv, _mm_mul_ps(x1v, _mm_load_ps(&ev[j])));
		    _mm_store_ps(&v[j], vv);
		  }		    
#else
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 20; j++)
		  {
		    ump_x1 += vl[j] * le[l * 20 + j];
		    ump_x2 += vr[j] * ri[l * 20 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 20; j++)
		  v[j] += x1px2 * extEV[l * 20 + j];
#endif
	      }
#ifdef __SIM_SSE3	  
	   __m128 minlikelihood_sse = _mm_set1_ps( minlikelihood_FLOAT );
	   
	   scale = 1;
	   for(l = 0; scale && (l < 20); l += 4)
	     {
	       __m128 vv = _mm_load_ps(&v[l]);
	       __m128 v1 = _mm_and_ps(vv, absMask_FLOAT.m);
	       v1 = _mm_cmplt_ps(v1,  minlikelihood_sse);
	       if(_mm_movemask_ps( v1 ) != 15)
		 scale = 0;
	     }	    	  	

	   if (scale)
	     {	       
	       __m128 tt = _mm_set1_ps(twotothe256_FLOAT);

	       for(l = 0; l < 20; l+=4)
		 {
		   __m128 vv = _mm_load_ps(&v[l]);
		   _mm_store_ps(&v[l], _mm_mul_ps(vv, tt));
		 }	       
	       
	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
    
#else
	    scale = 1;
	    for(l = 0; scale && (l < 20); l++)
	      scale = ((v[l] < minlikelihood_FLOAT) && (v[l] > minusminlikelihood_FLOAT));	   

	    if(scale)
	      {
		for(l = 0; l < 20; l++)
		  v[l] *= twotothe256_FLOAT;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
#endif
	  }
      }
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 400];
	  ri = &right[cptr[i] * 400];

	  vl = &x1[20 * i];
	  vr = &x2[20 * i];
	  v = &x3[20 * i];

#ifdef __SIM_SSE3
	    for(l = 0; l < 20; l+=4)
	      _mm_store_ps(&v[l], _mm_setzero_ps());	      		
#else
	  for(l = 0; l < 20; l++)
	    v[l] = 0.0;
#endif
	 
	  for(l = 0; l < 20; l++)
	    {
#ifdef __SIM_SSE3
		__m128 x1v = _mm_setzero_ps();
		__m128 x2v = _mm_setzero_ps();

		float
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];


		for(j = 0; j < 20; j+=4)
		  {
		    x1v = _mm_add_ps(x1v, _mm_mul_ps(_mm_load_ps(&vl[j]), _mm_load_ps(&lv[j])));		    
		    x2v = _mm_add_ps(x2v, _mm_mul_ps(_mm_load_ps(&vr[j]), _mm_load_ps(&rv[j])));
		  }

		x1v = _mm_hadd_ps(x1v, x1v);
		x1v = _mm_hadd_ps(x1v, x1v);
		x2v = _mm_hadd_ps(x2v, x2v);
		x2v = _mm_hadd_ps(x2v, x2v);

		x1v = _mm_mul_ps(x1v, x2v);
		
		for(j = 0; j < 20; j+=4)
		  {
		    __m128 vv = _mm_load_ps(&v[j]);
		    vv = _mm_add_ps(vv, _mm_mul_ps(x1v, _mm_load_ps(&ev[j])));
		    _mm_store_ps(&v[j], vv);
		  }		    
#else
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;

	      for(j = 0; j < 20; j++)
		{
		  ump_x1 += vl[j] * le[l * 20 + j];
		  ump_x2 += vr[j] * ri[l * 20 + j];
		}

	      x1px2 =  ump_x1 * ump_x2;

	      for(j = 0; j < 20; j++)
		v[j] += x1px2 * extEV[l * 20 + j];
#endif
	    }
#ifdef __SIM_SSE3	  
	   __m128 minlikelihood_sse = _mm_set1_ps( minlikelihood_FLOAT );
	   
	   scale = 1;
	   for(l = 0; scale && (l < 20); l += 4)
	     {
	       __m128 vv = _mm_load_ps(&v[l]);
	       __m128 v1 = _mm_and_ps(vv, absMask_FLOAT.m);
	       v1 = _mm_cmplt_ps(v1,  minlikelihood_sse);
	       if(_mm_movemask_ps( v1 ) != 15)
		 scale = 0;
	     }	    	  	

	   if (scale)
	     {	       
	       __m128 tt = _mm_set1_ps(twotothe256_FLOAT);

	       for(l = 0; l < 20; l+=4)
		 {
		   __m128 vv = _mm_load_ps(&v[l]);
		   _mm_store_ps(&v[l], _mm_mul_ps(vv, tt));
		 }	       
	       
	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
    
#else
	   scale = 1;
	   for(l = 0; scale && (l < 20); l++)
	     scale = ((v[l] < minlikelihood_FLOAT) && (v[l] > minusminlikelihood_FLOAT));	   

	   if(scale)
	     {

	       for(l = 0; l < 20; l++)
		 v[l] *= twotothe256_FLOAT;

	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
#endif
	}
      break;
    default:
      assert(0);
    }
  
  if(useFastScaling)
    *scalerIncrement = addScale;

}





static void newviewGTRCATSECONDARY(int tipCase, double *extEV,
				   int *cptr,
				   double *x1, double *x2, double *x3, double *tipVector,
				   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				   int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le, *ri, *v, *vl, *vr;
  double
    ump_x1, ump_x2, x1px2;
  int i, l, j, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 256];
	    ri = &right[cptr[i] * 256];

	    vl = &(tipVector[16 * tipX1[i]]);
	    vr = &(tipVector[16 * tipX2[i]]);
	    v  = &x3[16 * i];

	    for(l = 0; l < 16; l++)
	      v[l] = 0.0;

	    for(l = 0; l < 16; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 16; j++)
		  {
		    ump_x1 += vl[j] * le[l * 16 + j];
		    ump_x2 += vr[j] * ri[l * 16 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 16; j++)
		  v[j] += x1px2 * extEV[l * 16 + j];
	      }	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 256];
	    ri = &right[cptr[i] * 256];

	    vl = &(tipVector[16 * tipX1[i]]);
	    vr = &x2[16 * i];
	    v  = &x3[16 * i];

	    for(l = 0; l < 16; l++)
	      v[l] = 0.0;

	    for(l = 0; l < 16; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 16; j++)
		  {
		    ump_x1 += vl[j] * le[l * 16 + j];
		    ump_x2 += vr[j] * ri[l * 16 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 16; j++)
		  v[j] += x1px2 * extEV[l * 16 + j];
	      }

	    scale = 1;
	    for(l = 0; scale && (l < 16); l++)
	      scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	    

	    if(scale)
	      {
		for(l = 0; l < 16; l++)
		  v[l] *= twotothe256;
		
		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 256];
	  ri = &right[cptr[i] * 256];

	  vl = &x1[16 * i];
	  vr = &x2[16 * i];
	  v = &x3[16 * i];

	  for(l = 0; l < 16; l++)
	    v[l] = 0.0;

	  for(l = 0; l < 16; l++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;

	      for(j = 0; j < 16; j++)
		{
		  ump_x1 += vl[j] * le[l * 16 + j];
		  ump_x2 += vr[j] * ri[l * 16 + j];
		}

	      x1px2 =  ump_x1 * ump_x2;

	      for(j = 0; j < 16; j++)
		v[j] += x1px2 * extEV[l * 16 + j];
	    }

	   scale = 1;
	   for(l = 0; scale && (l < 16); l++)
	     scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));
	  
	   if(scale)
	     {
	       for(l = 0; l < 16; l++)
		 v[l] *= twotothe256;

	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	     
	     }
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}

static void newviewGTRCATSECONDARY_6(int tipCase, double *extEV,
				   int *cptr,
				   double *x1, double *x2, double *x3, double *tipVector,
				   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				   int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le, *ri, *v, *vl, *vr;
  double
    ump_x1, ump_x2, x1px2;
  int i, l, j, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 36];
	    ri = &right[cptr[i] * 36];

	    vl = &(tipVector[6 * tipX1[i]]);
	    vr = &(tipVector[6 * tipX2[i]]);
	    v  = &x3[6 * i];

	    for(l = 0; l < 6; l++)
	      v[l] = 0.0;

	    for(l = 0; l < 6; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 6; j++)
		  {
		    ump_x1 += vl[j] * le[l * 6 + j];
		    ump_x2 += vr[j] * ri[l * 6 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 6; j++)
		  v[j] += x1px2 * extEV[l * 6 + j];
	      }	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 36];
	    ri = &right[cptr[i] * 36];

	    vl = &(tipVector[6 * tipX1[i]]);
	    vr = &x2[6 * i];
	    v  = &x3[6 * i];

	    for(l = 0; l < 6; l++)
	      v[l] = 0.0;

	    for(l = 0; l < 6; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 6; j++)
		  {
		    ump_x1 += vl[j] * le[l * 6 + j];
		    ump_x2 += vr[j] * ri[l * 6 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 6; j++)
		  v[j] += x1px2 * extEV[l * 6 + j];
	      }

	    scale = 1;
	    for(l = 0; scale && (l < 6); l++)
	      scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	   

	    if(scale)
	      {
		for(l = 0; l < 6; l++)
		  v[l] *= twotothe256;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	       
	      }
	  }
      }
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 36];
	  ri = &right[cptr[i] * 36];

	  vl = &x1[6 * i];
	  vr = &x2[6 * i];
	  v = &x3[6 * i];

	  for(l = 0; l < 6; l++)
	    v[l] = 0.0;

	  for(l = 0; l < 6; l++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;

	      for(j = 0; j < 6; j++)
		{
		  ump_x1 += vl[j] * le[l * 6 + j];
		  ump_x2 += vr[j] * ri[l * 6 + j];
		}

	      x1px2 =  ump_x1 * ump_x2;

	      for(j = 0; j < 6; j++)
		v[j] += x1px2 * extEV[l * 6 + j];
	    }

	   scale = 1;
	   for(l = 0; scale && (l < 6); l++)
	     scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	  

	   if(scale)
	     {
	       for(l = 0; l < 6; l++)
		 v[l] *= twotothe256;
	       
	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;
	     }
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}

static void newviewGTRCATSECONDARY_7(int tipCase, double *extEV,
				     int *cptr,
				     double *x1, double *x2, double *x3, double *tipVector,
				     int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				     int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le, *ri, *v, *vl, *vr;
  double
    ump_x1, ump_x2, x1px2;
  int i, l, j, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 49];
	    ri = &right[cptr[i] * 49];

	    vl = &(tipVector[7 * tipX1[i]]);
	    vr = &(tipVector[7 * tipX2[i]]);
	    v  = &x3[7 * i];

	    for(l = 0; l < 7; l++)
	      v[l] = 0.0;

	    for(l = 0; l < 7; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 7; j++)
		  {
		    ump_x1 += vl[j] * le[l * 7 + j];
		    ump_x2 += vr[j] * ri[l * 7 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 7; j++)
		  v[j] += x1px2 * extEV[l * 7 + j];
	      }	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    le = &left[cptr[i] * 49];
	    ri = &right[cptr[i] * 49];

	    vl = &(tipVector[7 * tipX1[i]]);
	    vr = &x2[7 * i];
	    v  = &x3[7 * i];

	    for(l = 0; l < 7; l++)
	      v[l] = 0.0;

	    for(l = 0; l < 7; l++)
	      {
		ump_x1 = 0.0;
		ump_x2 = 0.0;

		for(j = 0; j < 7; j++)
		  {
		    ump_x1 += vl[j] * le[l * 7 + j];
		    ump_x2 += vr[j] * ri[l * 7 + j];
		  }

		x1px2 = ump_x1 * ump_x2;

		for(j = 0; j < 7; j++)
		  v[j] += x1px2 * extEV[l * 7 + j];
	      }

	    scale = 1;
	    for(l = 0; scale && (l < 7); l++)
	      scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	    

	    if(scale)
	      {
		for(l = 0; l < 7; l++)
		  v[l] *= twotothe256;

		if(useFastScaling)
		 addScale += wgt[i];
		else
		  ex3[i]  += 1;			     
	      }
	  }
      }
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 49];
	  ri = &right[cptr[i] * 49];

	  vl = &x1[7 * i];
	  vr = &x2[7 * i];
	  v = &x3[7 * i];

	  for(l = 0; l < 7; l++)
	    v[l] = 0.0;

	  for(l = 0; l < 7; l++)
	    {
	      ump_x1 = 0.0;
	      ump_x2 = 0.0;

	      for(j = 0; j < 7; j++)
		{
		  ump_x1 += vl[j] * le[l * 7 + j];
		  ump_x2 += vr[j] * ri[l * 7 + j];
		}

	      x1px2 =  ump_x1 * ump_x2;

	      for(j = 0; j < 7; j++)
		v[j] += x1px2 * extEV[l * 7 + j];
	    }

	   scale = 1;
	   for(l = 0; scale && (l < 7); l++)
	     scale = ((v[l] < minlikelihood) && (v[l] > minusminlikelihood));	  

	   if(scale)
	     {
	       for(l = 0; l < 7; l++)
		 v[l] *= twotothe256;

	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
	}
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}



static void newviewGTRGAMMAPROT(int tipCase,
				double *x1, double *x2, double *x3, double *extEV, double *tipVector,
				int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double  *uX1, *uX2, *v;
  double x1px2;
  int  i, j, l, k, scale, addScale = 0;
  double *vl, *vr, al, ar;



  switch(tipCase)
    {
    case TIP_TIP:
      {
	double umpX1[1840], umpX2[1840];

	for(i = 0; i < 23; i++)
	  {
	    v = &(tipVector[20 * i]);

	    for(k = 0; k < 80; k++)
	      {
#ifdef __SIM_SSE3
		double *ll =  &left[k * 20];
		double *rr =  &right[k * 20];
		
		__m128d umpX1v = _mm_setzero_pd();
		__m128d umpX2v = _mm_setzero_pd();

		for(l = 0; l < 20; l+=2)
		  {
		    __m128d vv = _mm_load_pd(&v[l]);
		    umpX1v = _mm_add_pd(umpX1v, _mm_mul_pd(vv, _mm_load_pd(&ll[l])));
		    umpX2v = _mm_add_pd(umpX2v, _mm_mul_pd(vv, _mm_load_pd(&rr[l])));					
		  }
		
		umpX1v = _mm_hadd_pd(umpX1v, umpX1v);
		umpX2v = _mm_hadd_pd(umpX2v, umpX2v);
		
		_mm_storel_pd(&umpX1[80 * i + k], umpX1v);
		_mm_storel_pd(&umpX2[80 * i + k], umpX2v);
#else
		umpX1[80 * i + k] = 0.0;
		umpX2[80 * i + k] = 0.0;

		for(l = 0; l < 20; l++)
		  {
		    umpX1[80 * i + k] +=  v[l] *  left[k * 20 + l];
		    umpX2[80 * i + k] +=  v[l] * right[k * 20 + l];
		  }
#endif
	      }
	  }

	for(i = 0; i < n; i++)
	  {
	    uX1 = &umpX1[80 * tipX1[i]];
	    uX2 = &umpX2[80 * tipX2[i]];

	    for(j = 0; j < 4; j++)
	      {
		v = &x3[i * 80 + j * 20];

#ifdef __SIM_SSE3
		__m128d zero =  _mm_setzero_pd();
		for(k = 0; k < 20; k+=2)		  		    
		  _mm_store_pd(&v[k], zero);

		for(k = 0; k < 20; k++)
		  { 
		    double *eev = &extEV[k * 20];
		    x1px2 = uX1[j * 20 + k] * uX2[j * 20 + k];
		    __m128d x1px2v = _mm_set1_pd(x1px2);

		    for(l = 0; l < 20; l+=2)
		      {
		      	__m128d vv = _mm_load_pd(&v[l]);
			__m128d ee = _mm_load_pd(&eev[l]);

			vv = _mm_add_pd(vv, _mm_mul_pd(x1px2v,ee));
			
			_mm_store_pd(&v[l], vv);
		      }
		  }

#else

		for(k = 0; k < 20; k++)
		  v[k] = 0.0;

		for(k = 0; k < 20; k++)
		  {		   
		    x1px2 = uX1[j * 20 + k] * uX2[j * 20 + k];
		   
		    for(l = 0; l < 20; l++)		      					
		      v[l] += x1px2 * extEV[20 * k + l];		     
		  }
#endif
	      }	   
	  }
      }
      break;
    case TIP_INNER:
      {
	double umpX1[1840], ump_x2[20];


	for(i = 0; i < 23; i++)
	  {
	    v = &(tipVector[20 * i]);

	    for(k = 0; k < 80; k++)
	      {
#ifdef __SIM_SSE3
		double *ll =  &left[k * 20];
				
		__m128d umpX1v = _mm_setzero_pd();
		
		for(l = 0; l < 20; l+=2)
		  {
		    __m128d vv = _mm_load_pd(&v[l]);
		    umpX1v = _mm_add_pd(umpX1v, _mm_mul_pd(vv, _mm_load_pd(&ll[l])));		    					
		  }
		
		umpX1v = _mm_hadd_pd(umpX1v, umpX1v);				
		_mm_storel_pd(&umpX1[80 * i + k], umpX1v);		
#else	    
		umpX1[80 * i + k] = 0.0;

		for(l = 0; l < 20; l++)
		  umpX1[80 * i + k] +=  v[l] * left[k * 20 + l];
#endif

	      }
	  }

	for (i = 0; i < n; i++)
	  {
	    uX1 = &umpX1[80 * tipX1[i]];

	    for(k = 0; k < 4; k++)
	      {
		v = &(x2[80 * i + k * 20]);
#ifdef __SIM_SSE3	       
		for(l = 0; l < 20; l++)
		  {		   
		    double *r =  &right[k * 400 + l * 20];
		    __m128d ump_x2v = _mm_setzero_pd();	    
		    
		    for(j = 0; j < 20; j+= 2)
		      {
			__m128d vv = _mm_load_pd(&v[j]);
			__m128d rr = _mm_load_pd(&r[j]);
			ump_x2v = _mm_add_pd(ump_x2v, _mm_mul_pd(vv, rr));
		      }
		     
		    ump_x2v = _mm_hadd_pd(ump_x2v, ump_x2v);
		    
		    _mm_storel_pd(&ump_x2[l], ump_x2v);		   		     
		  }

		v = &(x3[80 * i + 20 * k]);

		__m128d zero =  _mm_setzero_pd();
		for(l = 0; l < 20; l+=2)		  		    
		  _mm_store_pd(&v[l], zero);
		  
		for(l = 0; l < 20; l++)
		  {
		    double *eev = &extEV[l * 20];
		    x1px2 = uX1[k * 20 + l]  * ump_x2[l];
		    __m128d x1px2v = _mm_set1_pd(x1px2);
		  
		    for(j = 0; j < 20; j+=2)
		      {
			__m128d vv = _mm_load_pd(&v[j]);
			__m128d ee = _mm_load_pd(&eev[j]);
			
			vv = _mm_add_pd(vv, _mm_mul_pd(x1px2v,ee));
			
			_mm_store_pd(&v[j], vv);
		      }		     		    
		  }			
#else
		for(l = 0; l < 20; l++)
		  {
		    ump_x2[l] = 0.0;

		    for(j = 0; j < 20; j++)
		      ump_x2[l] += v[j] * right[k * 400 + l * 20 + j];
		  }

		v = &(x3[80 * i + 20 * k]);

		for(l = 0; l < 20; l++)
		  v[l] = 0;

		for(l = 0; l < 20; l++)
		  {
		    x1px2 = uX1[k * 20 + l]  * ump_x2[l];
		    for(j = 0; j < 20; j++)
		      v[j] += x1px2 * extEV[l * 20  + j];
		  }
#endif
	      }
	   
#ifdef __SIM_SSE3
	    { 
	      v = &(x3[80 * i]);
	      __m128d minlikelihood_sse = _mm_set1_pd( minlikelihood );
	      
	      scale = 1;
	      for(l = 0; scale && (l < 80); l += 2)
		{
		  __m128d vv = _mm_load_pd(&v[l]);
		  __m128d v1 = _mm_and_pd(vv, absMask.m);
		  v1 = _mm_cmplt_pd(v1,  minlikelihood_sse);
		  if(_mm_movemask_pd( v1 ) != 3)
		    scale = 0;
		}	    	  
	    }
#else
	    v = &x3[80 * i];
	    scale = 1;
	    for(l = 0; scale && (l < 80); l++)
	      scale = (ABS(v[l]) <  minlikelihood);
#endif

	    if (scale)
	      {
#ifdef __SIM_SSE3
	       __m128d twoto = _mm_set_pd(twotothe256, twotothe256);
	       
	       for(l = 0; l < 80; l+=2)
		 {
		   __m128d ex3v = _mm_load_pd(&v[l]);		  
		   _mm_store_pd(&v[l], _mm_mul_pd(ex3v,twoto));	
		 }		   		  
#else
		for(l = 0; l < 80; l++)
		  v[l] *= twotothe256;
#endif

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	       
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
       {
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[80 * i + 20 * k]);
	     vr = &(x2[80 * i + 20 * k]);
	     v =  &(x3[80 * i + 20 * k]);

#ifdef __SIM_SSE3
	     __m128d zero =  _mm_setzero_pd();
	     for(l = 0; l < 20; l+=2)		  		    
	       _mm_store_pd(&v[l], zero);
#else
	     for(l = 0; l < 20; l++)
	       v[l] = 0;
#endif

	     for(l = 0; l < 20; l++)
	       {		 
#ifdef __SIM_SSE3
		 {
		   __m128d al = _mm_setzero_pd();
		   __m128d ar = _mm_setzero_pd();

		   double *ll   = &left[k * 400 + l * 20];
		   double *rr   = &right[k * 400 + l * 20];
		   double *EVEV = &extEV[20 * l];
		   
		   for(j = 0; j < 20; j+=2)
		     {
		       __m128d lv  = _mm_load_pd(&ll[j]);
		       __m128d rv  = _mm_load_pd(&rr[j]);
		       __m128d vll = _mm_load_pd(&vl[j]);
		       __m128d vrr = _mm_load_pd(&vr[j]);
		       
		       al = _mm_add_pd(al, _mm_mul_pd(vll, lv));
		       ar = _mm_add_pd(ar, _mm_mul_pd(vrr, rv));
		     }  		 
		       
		   al = _mm_hadd_pd(al, al);
		   ar = _mm_hadd_pd(ar, ar);
		   
		   al = _mm_mul_pd(al, ar);

		   for(j = 0; j < 20; j+=2)
		     {
		       __m128d vv  = _mm_load_pd(&v[j]);
		       __m128d EVV = _mm_load_pd(&EVEV[j]);

		       vv = _mm_add_pd(vv, _mm_mul_pd(al, EVV));

		       _mm_store_pd(&v[j], vv);
		     }		  		   		  
		 }		 
#else
		 al = 0.0;
		 ar = 0.0;

		 for(j = 0; j < 20; j++)
		   {
		     al += vl[j] * left[k * 400 + l * 20 + j];
		     ar += vr[j] * right[k * 400 + l * 20 + j];
		   }

		 x1px2 = al * ar;

		 for(j = 0; j < 20; j++)
		   v[j] += x1px2 * extEV[20 * l + j];
#endif
	       }
	   }
	 

#ifdef __SIM_SSE3
	 { 
	   v = &(x3[80 * i]);
	   __m128d minlikelihood_sse = _mm_set1_pd( minlikelihood );
	   
	   scale = 1;
	   for(l = 0; scale && (l < 80); l += 2)
	     {
	       __m128d vv = _mm_load_pd(&v[l]);
	       __m128d v1 = _mm_and_pd(vv, absMask.m);
	       v1 = _mm_cmplt_pd(v1,  minlikelihood_sse);
	       if(_mm_movemask_pd( v1 ) != 3)
		 scale = 0;
	     }	    	  
	 }
#else
	 v = &(x3[80 * i]);
	 scale = 1;
	 for(l = 0; scale && (l < 80); l++)
	   scale = ((ABS(v[l]) <  minlikelihood));
#endif

	 if (scale)
	   {
#ifdef __SIM_SSE3
	       __m128d twoto = _mm_set_pd(twotothe256, twotothe256);
	       
	       for(l = 0; l < 80; l+=2)
		 {
		   __m128d ex3v = _mm_load_pd(&v[l]);		  
		   _mm_store_pd(&v[l], _mm_mul_pd(ex3v,twoto));	
		 }		   		  
#else	     
	     for(l = 0; l < 80; l++)
	       v[l] *= twotothe256;
#endif

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	  
	   }
       }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}


static void newviewGTRGAMMAPROT_FLOAT(int tipCase,
				      float *x1, float *x2, float *x3, float *extEV, float *tipVector,
				      int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				      int n, float *left, float *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  float
    *uX1,
    *uX2,
    *v,
    x1px2,
    *vl,
    *vr,
    al,
    ar;

  int  i, j, l, k, scale, addScale = 0;



  switch(tipCase)
    {
    case TIP_TIP:
      {
	float umpX1[1840], umpX2[1840];

	for(i = 0; i < 23; i++)
	  {
	    v = &(tipVector[20 * i]);

	    for(k = 0; k < 80; k++)
	      {
#ifdef __SIM_SSE3
		float *ll =  &left[k * 20];
		float *rr =  &right[k * 20];
		
		__m128 umpX1v = _mm_setzero_ps();
		__m128 umpX2v = _mm_setzero_ps();

		for(l = 0; l < 20; l+=4)
		  {
		    __m128 vv = _mm_load_ps(&v[l]);
		    umpX1v = _mm_add_ps(umpX1v, _mm_mul_ps(vv, _mm_load_ps(&ll[l])));
		    umpX2v = _mm_add_ps(umpX2v, _mm_mul_ps(vv, _mm_load_ps(&rr[l])));					
		  }
		
		umpX1v = _mm_hadd_ps(umpX1v, umpX1v);
		umpX1v = _mm_hadd_ps(umpX1v, umpX1v);
		umpX2v = _mm_hadd_ps(umpX2v, umpX2v);
		umpX2v = _mm_hadd_ps(umpX2v, umpX2v);

		_mm_store_ss(&umpX1[80 * i + k], umpX1v);
		_mm_store_ss(&umpX2[80 * i + k], umpX2v);
#else


		umpX1[80 * i + k] = 0.0;
		umpX2[80 * i + k] = 0.0;

		for(l = 0; l < 20; l++)
		  {
		    umpX1[80 * i + k] +=  v[l] *  left[k * 20 + l];
		    umpX2[80 * i + k] +=  v[l] * right[k * 20 + l];
		  }
#endif
	      }
	  }

	for(i = 0; i < n; i++)
	  {
	    uX1 = &umpX1[80 * tipX1[i]];
	    uX2 = &umpX2[80 * tipX2[i]];
#ifdef __SIM_SSE3
	    for(j = 0; j < 4; j++)
	      {
		v = &x3[i * 80 + j * 20];	       
		
		for(k = 0; k < 20; k+=4)
		  _mm_store_ps(&v[k], _mm_setzero_ps());
		 

		for(k = 0; k < 20; k++)
		  {
		    float *eev = &extEV[20 * k];
		    x1px2 = uX1[j * 20 + k] * uX2[j * 20 + k];
		    __m128 x1px2v = _mm_set1_ps(x1px2);
		    
		    for(l = 0; l < 20; l+=4)
		      {
			__m128 vv = _mm_load_ps(&v[l]);
			__m128 ee = _mm_load_ps(&eev[l]);

			vv = _mm_add_ps(vv, _mm_mul_ps(x1px2v,ee));
			_mm_store_ps(&v[l], vv);

			/*v[l] += x1px2 * extEV[20 * k + l];*/
		      }
		  }
	      }	   

#else
	    for(j = 0; j < 4; j++)
	      {
		v = &x3[i * 80 + j * 20];		
		
		for(k = 0; k < 20; k+=4)		 
		  v[k] = 0.0;

		for(k = 0; k < 20; k++)
		  {
		    x1px2 = uX1[j * 20 + k] * uX2[j * 20 + k];
		    for(l = 0; l < 20; l++)
		      v[l] += x1px2 * extEV[20 * k + l];
		  }
	      }	    
#endif
	  }
      }
      break;
    case TIP_INNER:
      {
	float umpX1[1840], ump_x2[20];


	for(i = 0; i < 23; i++)
	  {
	    v = &(tipVector[20 * i]);

	    for(k = 0; k < 80; k++)
	      {
#ifdef __SIM_SSE3
		float *ll =  &left[k * 20];
				
		__m128 umpX1v = _mm_setzero_ps();
		
		for(l = 0; l < 20; l+=4)
		  {
		    __m128 vv = _mm_load_ps(&v[l]);
		    umpX1v = _mm_add_ps(umpX1v, _mm_mul_ps(vv, _mm_load_ps(&ll[l])));		    					
		  }
		
		umpX1v = _mm_hadd_ps(umpX1v, umpX1v);				
		umpX1v = _mm_hadd_ps(umpX1v, umpX1v);

		_mm_store_ss(&umpX1[80 * i + k], umpX1v);		
#else

		umpX1[80 * i + k] = 0.0;

		for(l = 0; l < 20; l++)
		  umpX1[80 * i + k] +=  v[l] * left[k * 20 + l];
#endif

	      }
	  }

	for (i = 0; i < n; i++)
	  {
	    uX1 = &umpX1[80 * tipX1[i]];

#ifdef __SIM_SSE3
	   for(k = 0; k < 4; k++)
	      {		
		v = &(x2[80 * i + k * 20]);
		for(l = 0; l < 20; l++)
		  {
		    float *r =  &right[k * 400 + l * 20];
		    __m128 ump_x2v = _mm_setzero_ps();	 
		    

		    for(j = 0; j < 20; j+=4)
		      {
			__m128 vv = _mm_load_ps(&v[j]);
			__m128 rr = _mm_load_ps(&r[j]);
			ump_x2v = _mm_add_ps(ump_x2v, _mm_mul_ps(vv, rr));			
		      }

		    ump_x2v = _mm_hadd_ps(ump_x2v, ump_x2v);
		    ump_x2v = _mm_hadd_ps(ump_x2v, ump_x2v);

		    _mm_store_ss(&ump_x2[l], ump_x2v);
		  }

		v = &(x3[80 * i + 20 * k]);

		for(l = 0; l < 20; l+=4)
		  _mm_store_ps(&v[l], _mm_setzero_ps());

		for(l = 0; l < 20; l++)
		  {
		    float *eev = &extEV[l * 20];
		    x1px2 = uX1[k * 20 + l]  * ump_x2[l];
		    __m128 x1px2v = _mm_set1_ps(x1px2);

		    for(j = 0; j < 20; j+=4)
		      {
			__m128 vv = _mm_load_ps(&v[j]);
			__m128 ee = _mm_load_ps(&eev[j]);
			
			vv = _mm_add_ps(vv, _mm_mul_ps(x1px2v,ee));
			
			_mm_store_ps(&v[j], vv);
		      }			    
		  }
	      } 

#else
	    for(k = 0; k < 4; k++)
	      {
		v = &(x2[80 * i + k * 20]);
		for(l = 0; l < 20; l++)
		  {
		    ump_x2[l] = 0.0;

		    for(j = 0; j < 20; j++)
		      ump_x2[l] += v[j] * right[k * 400 + l * 20 + j];
		  }

		v = &(x3[80 * i + 20 * k]);

		for(l = 0; l < 20; l++)
		  v[l] = 0;

		for(l = 0; l < 20; l++)
		  {
		    x1px2 = uX1[k * 20 + l]  * ump_x2[l];
		    for(j = 0; j < 20; j++)
		      v[j] += x1px2 * extEV[l * 20  + j];
		  }
	      }
#endif
	    

#ifdef __SIM_SSE3
	   v = &(x3[80 * i]);
	   __m128 minlikelihood_sse = _mm_set1_ps( minlikelihood_FLOAT );
	   
	   scale = 1;
	   for(l = 0; scale && (l < 80); l += 4)
	     {
	       __m128 vv = _mm_load_ps(&v[l]);
	       __m128 v1 = _mm_and_ps(vv, absMask_FLOAT.m);
	       v1 = _mm_cmplt_ps(v1,  minlikelihood_sse);
	       if(_mm_movemask_ps( v1 ) != 15)
		 scale = 0;
	     }	    	  	

	   if (scale)
	     {	       
	       __m128 tt = _mm_set1_ps(twotothe256_FLOAT);

	       for(l = 0; l < 80; l+=4)
		 {
		   __m128 vv = _mm_load_ps(&v[l]);
		   _mm_store_ps(&v[l], _mm_mul_ps(vv, tt));
		 }
	       /* v[l] *= twotothe256_FLOAT;*/
	       
	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
    
#else
	    v = &x3[80 * i];
	    scale = 1;
	    for(l = 0; scale && (l < 80); l++)
	      scale = (ABS(v[l]) <  minlikelihood_FLOAT);

	    if (scale)
	      {	       
		for(l = 0; l < 80; l++)
		  v[l] *= twotothe256_FLOAT;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
#endif
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
       {
#ifdef __SIM_SSE3
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[80 * i + 20 * k]);
	     vr = &(x2[80 * i + 20 * k]);
	     v =  &(x3[80 * i + 20 * k]);

	     for(l = 0; l < 20; l+=4)
	       _mm_store_ps(&v[l], _mm_setzero_ps());

	     for(l = 0; l < 20; l++)
	       {
		 __m128 al = _mm_setzero_ps();
		 __m128 ar = _mm_setzero_ps();
		 
		 float *ll   = &left[k * 400 + l * 20];
		 float *rr   = &right[k * 400 + l * 20];
		 float *EVEV = &extEV[20 * l];
		 
		 for(j = 0; j < 20; j+=4)
		   {
		     __m128 lv  = _mm_load_ps(&ll[j]);
		     __m128 rv  = _mm_load_ps(&rr[j]);
		     __m128 vll = _mm_load_ps(&vl[j]);
		     __m128 vrr = _mm_load_ps(&vr[j]);
		     
		     al = _mm_add_ps(al, _mm_mul_ps(vll, lv));
		     ar = _mm_add_ps(ar, _mm_mul_ps(vrr, rv));
		   }  
		 
		 al = _mm_hadd_ps(al, al);
		 al = _mm_hadd_ps(al, al);
		 ar = _mm_hadd_ps(ar, ar);
		 ar = _mm_hadd_ps(ar, ar);
		 
		 al = _mm_mul_ps(al, ar);
		 
		 for(j = 0; j < 20; j+=4)
		   {
		     __m128 vv  = _mm_load_ps(&v[j]);
		     __m128 EVV = _mm_load_ps(&EVEV[j]);
		     
		     vv = _mm_add_ps(vv, _mm_mul_ps(al, EVV));
		     
		     _mm_store_ps(&v[j], vv);
		   }			 
	       }
	   }

#else
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[80 * i + 20 * k]);
	     vr = &(x2[80 * i + 20 * k]);
	     v =  &(x3[80 * i + 20 * k]);

	     for(l = 0; l < 20; l++)
	       v[l] = 0;

	     for(l = 0; l < 20; l++)
	       {
		 al = 0.0;
		 ar = 0.0;
		 for(j = 0; j < 20; j++)
		   {
		     al += vl[j] * left[k * 400 + l * 20 + j];
		     ar += vr[j] * right[k * 400 + l * 20 + j];
		   }

		 x1px2 = al * ar;
		 for(j = 0; j < 20; j++)
		   v[j] += x1px2 * extEV[20 * l + j];
	       }
	   }
#endif
#ifdef __SIM_SSE3
	   v = &(x3[80 * i]);
	   __m128 minlikelihood_sse = _mm_set1_ps( minlikelihood_FLOAT );
	   
	   scale = 1;
	   for(l = 0; scale && (l < 80); l += 4)
	     {
	       __m128 vv = _mm_load_ps(&v[l]);
	       __m128 v1 = _mm_and_ps(vv, absMask_FLOAT.m);
	       v1 = _mm_cmplt_ps(v1,  minlikelihood_sse);
	       if(_mm_movemask_ps( v1 ) != 15)
		 scale = 0;
	     }	    	  	

	   if (scale)
	     {	       
	       __m128 tt = _mm_set1_ps(twotothe256_FLOAT);

	       for(l = 0; l < 80; l+=4)
		 {
		   __m128 vv = _mm_load_ps(&v[l]);
		   _mm_store_ps(&v[l], _mm_mul_ps(vv, tt));
		 }
	       /* v[l] *= twotothe256_FLOAT;*/
	       
	       if(useFastScaling)
		 addScale += wgt[i];
	       else
		 ex3[i]  += 1;	      
	     }
    
#else	 
	 v = &(x3[80 * i]);
	 scale = 1;
	 for(l = 0; scale && (l < 80); l++)
	   scale = ((ABS(v[l]) <  minlikelihood_FLOAT));

	 if (scale)
	   {	    
	     for(l = 0; l < 80; l++)
	       v[l] *= twotothe256_FLOAT;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	   
	   }
#endif
       }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}


static void newviewGTRGAMMASECONDARY(int tipCase,
				     double *x1, double *x2, double *x3, double *extEV, double *tipVector,
				     int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				     int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double  *v;
  double x1px2;
  int  i, j, l, k, scale, addScale = 0;
  double *vl, *vr, al, ar;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for(i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[16 * tipX1[i]]);
		vr = &(tipVector[16 * tipX2[i]]);
		v =  &(x3[64 * i + 16 * k]);

		for(l = 0; l < 16; l++)
		  v[l] = 0;

		for(l = 0; l < 16; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < 16; j++)
		      {
			al += vl[j] * left[k * 256 + l * 16 + j];
			ar += vr[j] * right[k * 256 + l * 16 + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < 16; j++)
		      v[j] += x1px2 * extEV[16 * l + j];
		  }
	      }	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[16 * tipX1[i]]);
		vr = &(x2[64 * i + 16 * k]);
		v =  &(x3[64 * i + 16 * k]);

		for(l = 0; l < 16; l++)
		  v[l] = 0;

		for(l = 0; l < 16; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < 16; j++)
		      {
			al += vl[j] * left[k * 256 + l * 16 + j];
			ar += vr[j] * right[k * 256 + l * 16 + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < 16; j++)
		      v[j] += x1px2 * extEV[16 * l + j];
		  }
	      }
	   
	    v = &x3[64 * i];
	    scale = 1;
	    for(l = 0; scale && (l < 64); l++)
	      scale = (ABS(v[l]) <  minlikelihood);

	    if (scale)
	      {
		for(l = 0; l < 64; l++)
		  v[l] *= twotothe256;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
       {
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[64 * i + 16 * k]);
	     vr = &(x2[64 * i + 16 * k]);
	     v =  &(x3[64 * i + 16 * k]);

	     for(l = 0; l < 16; l++)
	       v[l] = 0;

	     for(l = 0; l < 16; l++)
	       {
		 al = 0.0;
		 ar = 0.0;
		 for(j = 0; j < 16; j++)
		   {
		     al += vl[j] * left[k * 256 + l * 16 + j];
		     ar += vr[j] * right[k * 256 + l * 16 + j];
		   }

		 x1px2 = al * ar;
		 for(j = 0; j < 16; j++)
		   v[j] += x1px2 * extEV[16 * l + j];
	       }
	   }
	 
	 v = &(x3[64 * i]);
	 scale = 1;
	 for(l = 0; scale && (l < 64); l++)
	   scale = ((ABS(v[l]) <  minlikelihood));

	 if (scale)
	   {
	     for(l = 0; l < 64; l++)
	       v[l] *= twotothe256;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	    
	   }
       }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}


static void newviewGTRGAMMASECONDARY_6(int tipCase,
				       double *x1, double *x2, double *x3, double *extEV, double *tipVector,
				       int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				       int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double  *v;
  double x1px2;
  int  i, j, l, k, scale, addScale = 0;
  double *vl, *vr, al, ar;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for(i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[6 * tipX1[i]]);
		vr = &(tipVector[6 * tipX2[i]]);
		v =  &(x3[24 * i + 6 * k]);

		for(l = 0; l < 6; l++)
		  v[l] = 0;

		for(l = 0; l < 6; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < 6; j++)
		      {
			al += vl[j] * left[k * 36 + l * 6 + j];
			ar += vr[j] * right[k * 36 + l * 6 + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < 6; j++)
		      v[j] += x1px2 * extEV[6 * l + j];
		  }
	      }	   
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[6 * tipX1[i]]);
		vr = &(x2[24 * i + 6 * k]);
		v =  &(x3[24 * i + 6 * k]);

		for(l = 0; l < 6; l++)
		  v[l] = 0;

		for(l = 0; l < 6; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < 6; j++)
		      {
			al += vl[j] * left[k * 36 + l * 6 + j];
			ar += vr[j] * right[k * 36 + l * 6 + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < 6; j++)
		      v[j] += x1px2 * extEV[6 * l + j];
		  }
	      }
	   
	    v = &x3[24 * i];
	    scale = 1;
	    for(l = 0; scale && (l < 24); l++)
	      scale = (ABS(v[l]) <  minlikelihood);

	    if(scale)
	      {
		for(l = 0; l < 24; l++)
		  v[l] *= twotothe256;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;		
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
       {
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[24 * i + 6 * k]);
	     vr = &(x2[24 * i + 6 * k]);
	     v =  &(x3[24 * i + 6 * k]);

	     for(l = 0; l < 6; l++)
	       v[l] = 0;

	     for(l = 0; l < 6; l++)
	       {
		 al = 0.0;
		 ar = 0.0;
		 for(j = 0; j < 6; j++)
		   {
		     al += vl[j] * left[k * 36 + l * 6 + j];
		     ar += vr[j] * right[k * 36 + l * 6 + j];
		   }

		 x1px2 = al * ar;
		 for(j = 0; j < 6; j++)
		   v[j] += x1px2 * extEV[6 * l + j];
	       }
	   }
	 
	 v = &(x3[24 * i]);
	 scale = 1;
	 for(l = 0; scale && (l < 24); l++)
	   scale = ((ABS(v[l]) <  minlikelihood));

	 if (scale)
	   {
	     for(l = 0; l < 24; l++)
	       v[l] *= twotothe256;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	   
	   }
       }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;

}


static void newviewGTRGAMMASECONDARY_7(int tipCase,
				       double *x1, double *x2, double *x3, double *extEV, double *tipVector,
				       int *ex3, unsigned char *tipX1, unsigned char *tipX2,
				       int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double  *v;
  double x1px2;
  int  i, j, l, k, scale, addScale = 0;
  double *vl, *vr, al, ar;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for(i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[7 * tipX1[i]]);
		vr = &(tipVector[7 * tipX2[i]]);
		v =  &(x3[28 * i + 7 * k]);

		for(l = 0; l < 7; l++)
		  v[l] = 0;

		for(l = 0; l < 7; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < 7; j++)
		      {
			al += vl[j] * left[k * 49 + l * 7 + j];
			ar += vr[j] * right[k * 49 + l * 7 + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < 7; j++)
		      v[j] += x1px2 * extEV[7 * l + j];
		  }
	      }	   
	  }
      }
      break;
    case TIP_INNER:
      {
	for (i = 0; i < n; i++)
	  {
	    for(k = 0; k < 4; k++)
	      {
		vl = &(tipVector[7 * tipX1[i]]);
		vr = &(x2[28 * i + 7 * k]);
		v =  &(x3[28 * i + 7 * k]);

		for(l = 0; l < 7; l++)
		  v[l] = 0;

		for(l = 0; l < 7; l++)
		  {
		    al = 0.0;
		    ar = 0.0;
		    for(j = 0; j < 7; j++)
		      {
			al += vl[j] * left[k * 49 + l * 7 + j];
			ar += vr[j] * right[k * 49 + l * 7 + j];
		      }

		    x1px2 = al * ar;
		    for(j = 0; j < 7; j++)
		      v[j] += x1px2 * extEV[7 * l + j];
		  }
	      }
	   
	    v = &x3[28 * i];
	    scale = 1;
	    for(l = 0; scale && (l < 28); l++)
	      scale = (ABS(v[l]) <  minlikelihood);

	    if (scale)
	      {
		for(l = 0; l < 28; l++)
		  v[l] *= twotothe256;

		if(useFastScaling)
		  addScale += wgt[i];
		else
		  ex3[i]  += 1;	      
	      }
	  }
      }
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
       {
	 for(k = 0; k < 4; k++)
	   {
	     vl = &(x1[28 * i + 7 * k]);
	     vr = &(x2[28 * i + 7 * k]);
	     v =  &(x3[28 * i + 7 * k]);

	     for(l = 0; l < 7; l++)
	       v[l] = 0;

	     for(l = 0; l < 7; l++)
	       {
		 al = 0.0;
		 ar = 0.0;
		 for(j = 0; j < 7; j++)
		   {
		     al += vl[j] * left[k * 49 + l * 7 + j];
		     ar += vr[j] * right[k * 49 + l * 7 + j];
		   }

		 x1px2 = al * ar;
		 for(j = 0; j < 7; j++)
		   v[j] += x1px2 * extEV[7 * l + j];
	       }
	   }
	 
	 v = &(x3[28 * i]);
	 scale = 1;
	 for(l = 0; scale && (l < 28); l++)
	   scale = ((ABS(v[l]) <  minlikelihood));

	 if (scale)
	   {
	     for(l = 0; l < 28; l++)
	       v[l] *= twotothe256;

	     if(useFastScaling)
	       addScale += wgt[i];
	     else
	       ex3[i]  += 1;	   
	   }
       }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;
}







void computeTraversalInfo(nodeptr p, traversalInfo *ti, int *counter, int maxTips, int numBranches)
{
  if(isTip(p->number, maxTips))
    return;

  {
    int i;
    nodeptr q = p->next->back;
    nodeptr r = p->next->next->back;

    if(isTip(r->number, maxTips) && isTip(q->number, maxTips))
      {
	while (! p->x)
	 {
	   if (! p->x)
	     getxnode(p);
	 }

	ti[*counter].tipCase = TIP_TIP;
	ti[*counter].pNumber = p->number;
	ti[*counter].qNumber = q->number;
	ti[*counter].rNumber = r->number;
	for(i = 0; i < numBranches; i++)
	  {
	    double z;
	    z = q->z[i];
	    z = (z > zmin) ? log(z) : log(zmin);
	    ti[*counter].qz[i] = z;

	    z = r->z[i];
	    z = (z > zmin) ? log(z) : log(zmin);
	    ti[*counter].rz[i] = z;
	  }
	*counter = *counter + 1;
      }
    else
      {
	if(isTip(r->number, maxTips) || isTip(q->number, maxTips))
	  {
	    nodeptr tmp;

	    if(isTip(r->number, maxTips))
	      {
		tmp = r;
		r = q;
		q = tmp;
	      }

	    while ((! p->x) || (! r->x))
	      {
		if (! r->x)
		  computeTraversalInfo(r, ti, counter, maxTips, numBranches);
		if (! p->x)
		  getxnode(p);
	      }

	    ti[*counter].tipCase = TIP_INNER;
	    ti[*counter].pNumber = p->number;
	    ti[*counter].qNumber = q->number;
	    ti[*counter].rNumber = r->number;
	    for(i = 0; i < numBranches; i++)
	      {
		double z;
		z = q->z[i];
		z = (z > zmin) ? log(z) : log(zmin);
		ti[*counter].qz[i] = z;

		z = r->z[i];
		z = (z > zmin) ? log(z) : log(zmin);
		ti[*counter].rz[i] = z;
	      }

	    *counter = *counter + 1;
	  }
	else
	  {

	    while ((! p->x) || (! q->x) || (! r->x))
	      {
		if (! q->x)
		  computeTraversalInfo(q, ti, counter, maxTips, numBranches);
		if (! r->x)
		  computeTraversalInfo(r, ti, counter, maxTips, numBranches);
		if (! p->x)
		  getxnode(p);
	      }

	    ti[*counter].tipCase = INNER_INNER;
	    ti[*counter].pNumber = p->number;
	    ti[*counter].qNumber = q->number;
	    ti[*counter].rNumber = r->number;
	    for(i = 0; i < numBranches; i++)
	      {
		double z;
		z = q->z[i];
		z = (z > zmin) ? log(z) : log(zmin);
		ti[*counter].qz[i] = z;

		z = r->z[i];
		z = (z > zmin) ? log(z) : log(zmin);
		ti[*counter].rz[i] = z;
	      }

	    *counter = *counter + 1;
	  }
      }
  }

  

}



void computeTraversalInfoMulti(nodeptr p, traversalInfo *ti, int *counter, int maxTips, int model)
{
  if(isTip(p->number, maxTips))
    {
      assert(p->isPresent[model / MASK_LENGTH] & mask32[model % MASK_LENGTH]);
      assert(p->backs[model]);
      return;
    }

 
  assert(p->backs[model]);

  {     
    nodeptr q = p->next->backs[model];
    nodeptr r = p->next->next->backs[model];
    
    assert(p == p->next->next->next);
      
    assert(q && r);

    if(isTip(r->number, maxTips) && isTip(q->number, maxTips))
      {	  
	while (! p->xs[model])
	 {	  
	   if (! p->xs[model])
	     getxsnode(p, model); 	   
	 }

	assert(p->xs[model]);

	ti[*counter].tipCase = TIP_TIP; 
	ti[*counter].pNumber = p->number;
	ti[*counter].qNumber = q->number;
	ti[*counter].rNumber = r->number;
	
	{
	  double z;
	  z = q->z[model];
	  z = (z > zmin) ? log(z) : log(zmin);
	  ti[*counter].qz[model] = z;
	  
	  z = r->z[model];
	  z = (z > zmin) ? log(z) : log(zmin);
	  ti[*counter].rz[model] = z;	    
	}     
	*counter = *counter + 1;
      }  
    else
      {
	if(isTip(r->number, maxTips) || isTip(q->number, maxTips))
	  {		
	    nodeptr tmp;

	    if(isTip(r->number, maxTips))
	      {
		tmp = r;
		r = q;
		q = tmp;
	      }

	    while ((! p->xs[model]) || (! r->xs[model])) 
	      {	 			
		if (! r->xs[model]) 
		  computeTraversalInfoMulti(r, ti, counter, maxTips, model);
		if (! p->xs[model]) 
		  getxsnode(p, model);	
	      }
	    	   
	    assert(p->xs[model] && r->xs[model]);

	    ti[*counter].tipCase = TIP_INNER; 
	    ti[*counter].pNumber = p->number;
	    ti[*counter].qNumber = q->number;
	    ti[*counter].rNumber = r->number;
	   
	    {
	      double z;
	      z = q->z[model];
	      z = (z > zmin) ? log(z) : log(zmin);
	      ti[*counter].qz[model] = z;
	      
	      z = r->z[model];
	      z = (z > zmin) ? log(z) : log(zmin);
		ti[*counter].rz[model] = z;		
	    }   
	    
	    *counter = *counter + 1;
	  }
	else
	  {	 

	    while ((! p->xs[model]) || (! q->xs[model]) || (! r->xs[model])) 
	      {		
		if (! q->xs[model]) 
		  computeTraversalInfoMulti(q, ti, counter, maxTips, model);
		if (! r->xs[model]) 
		  computeTraversalInfoMulti(r, ti, counter, maxTips, model);
		if (! p->xs[model]) 
		  getxsnode(p, model);	
	      }

	    assert(p->xs[model] && r->xs[model] && q->xs[model]);

	    ti[*counter].tipCase = INNER_INNER; 
	    ti[*counter].pNumber = p->number;
	    ti[*counter].qNumber = q->number;
	    ti[*counter].rNumber = r->number;
	   
	    {
	      double z;
	      z = q->z[model];
	      z = (z > zmin) ? log(z) : log(zmin);
	      ti[*counter].qz[model] = z;
	      
	      z = r->z[model];
	      z = (z > zmin) ? log(z) : log(zmin);
	      ti[*counter].rz[model] = z;		
	    }   
	    
	    *counter = *counter + 1;
	  }
      }    
  }

}




#ifndef MEMORG_TIMES

#if defined(__i386__)

static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int x;
     __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
     return x;
}
#elif defined(__x86_64__)

static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#elif defined(__powerpc__)

static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ volatile(
                "0:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     0b         \n"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = upper;
  result = result<<32;
  result = result|lower;

  return(result);
}

#else

#error "No tick counter is available!"

#endif

#endif //memorg_times




extern float timeEventsKernel;
extern float timeEventsAll;
#ifdef MEMORG

void newviewIterativeGPU (tree *tr) //from evaluateIterative
{

    static int countr = 0;
    
    uintptr_t tmp1;
    uintptr_t tmp2;
    uintptr_t tmp3;

//    unsigned long long partTmp1, partTmp2, partTime1, partTime2, fullTime1, fullTime2;

#ifdef cudaEvent
    cudaEvent_t startK, stopK, startT2D, stopT2D, startT2H, stopT2H;
    float tmpTime;
    
    cudaEventCreate(&startK);
    cudaEventCreate(&stopK);
    cudaEventCreate(&startT2D);
    cudaEventCreate(&stopT2D);
    cudaEventCreate(&startT2H);
    cudaEventCreate(&stopT2H);
#endif
    
#ifdef testKernelAccssMem
#ifdef MEMORG
uintptr_t tmp1;
uintptr_t tmp2;
uintptr_t tmp3;
  if (gCounter == 12)
  {
   printf("1. tipVector_FLOAT[30] before memCpy to device: %f...\n", tr->partitionData[0].tipVector_FLOAT[30]);
   printf("2. yVector[%d][%d] before memCpy to device: %d...\n", ydim1, ydim2, tr->partitionData[0].yVector[ydim1][ydim2]);
   printf("2a. yVector[100][2] ==  before memCpy back to host: ? %d\n", tr->partitionData[0].yVector[100][2]);

   printf("3. xVector[4][0] before memCpy to device: %f...\n", tr->partitionData[0].xVector_FLOAT[4][0]);
   printf("4. wgt[2048] before memCpy to device: %d...\n", tr->partitionData[0].wgt[2048]);  //int *d_wgt;
   printf("5. xVector[40][2047] before memCpy to device: %f...\n", tr->partitionData[0].xVector_FLOAT[40][2047]);

   printf("before tr->td[0].count == %d\n", tr->td[0].count);  //tree *d_tree;
   printf("before tr->td[0].ti->tipCase == %d\n", tr->td[0].ti[1].tipCase);  //traversalInfo *d_ti;  
   printf("before tr->partitionData[0].left_FLOAT[399] == %f\n", tr->partitionData[0].left_FLOAT[399]);  //float *d_left;
  

   
//float *d_EV;
//float *d_tipVector;

//float *d_right;
//double *d_patrat;
//double *d_ei;
//double *d_eign;
//int *d_rateCategory;



 
   
   tmp1 = (uintptr_t) globalpStart;
   tmp2 = (uintptr_t) globalpEnd;
   tmp3 = tmp2 - tmp1;
   

   cudaMemcpy(d_globalpStart, globalpStart, (size_t)tmp3, cudaMemcpyHostToDevice);
	
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
  
  printf("\n......KERNEL start....\n");
  kernelFunc(0);
  


  cudaMemcpy(globalpStart, d_globalpStart, (size_t)tmp3, cudaMemcpyDeviceToHost);
  printf("\n......KERNEL stop....\n");
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
  printf("1. tipVector_FLOAT[30] after memCpy from device: %f...\n", tr->partitionData[0].tipVector_FLOAT[30]);
  printf("2. yVector[%d][%d] == G (71)  after memCpy back to host: ? %d\n", ydim1, ydim2, tr->partitionData[0].yVector[ydim1][ydim2]);
  printf("2a. yVector[100][2] ==  after memCpy back to host: ? %d\n", tr->partitionData[0].yVector[100][2]);

  printf("3. xVector[4][0] after memCpy to device: %f...\n", tr->partitionData[0].xVector_FLOAT[4][0]);
  printf("4. wgt[2048] after memCpy to device: %d...\n", tr->partitionData[0].wgt[2048]);
  printf("5. xVector[40][2047] after memCpy to device: %f...\n", tr->partitionData[0].xVector_FLOAT[40][2047]);

   printf("after tr->td[0].count == %d\n", tr->td[0].count);  //tree *d_tree;
   printf("after tr->td[0].ti->tipCase == %d\n", tr->td[0].ti[1].tipCase);  //traversalInfo *d_ti;  
   printf("after tr->partitionData[0].left_FLOAT[399] == %f\n\n", tr->partitionData[0].left_FLOAT[399]);  //float *d_left;
  
  }
  gCounter++;
#endif
#endif


  /* 
   * start GPU execution
   */
    int i;

    //transfer  only usefull parts of data
 
    cudaError_t error;
    
    //fullTime1 = rdtsc();
    
    //partTime1 = rdtsc();

#ifdef cudaEvent    
    cudaEventRecord(startT2D, 0);
#endif
    
    if (firstIter)
    {
        //testDataTransfer = (float *) malloc(dataTransferSizeToHost);
        //printf("\n-----> sizeof dataTransferSizeInit %zu\n", dataTransferSizeInit);
        //printf("-----> sizeof dataTransferSizePart1 %zu\n", dataTransferSizePart1);
        //printf("-----> sizeof dataTransferSizePart2 %zu\n", dataTransferSizePart2);
        //printf("-----> sizeof dataTransferSizeToHost %zu\n", dataTransferSizeToHost);
        
        
        tmp1 = (uintptr_t) d_globalpStart;
        tmp2 = (uintptr_t) d_globalpEnd;
        tmp3 = tmp2 - tmp1;
        //assert(tmp3 == (globalpEnd-globalpStart));
        printf("\n-----> sizeof dataTransferSizeInit %zu\n", tmp3);

        cudaMemcpy(d_globalpStart, globalpStart, (size_t)tmp3, cudaMemcpyHostToDevice);
	
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy firstIter: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
        
        
        /*
        for (i=0; i<nofSpecies; i++){
        
        cudaMemcpy(xVector_dp[i], tr->partitionData[0].xVector_FLOAT[i], memoryRequirements, cudaMemcpyHostToDevice);
	
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy xVector_dp: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
        }
        */
      
        firstIter = FALSE;
    }
    else
    {
#ifndef reduseMemTransf_onlyTransGammaRates
        
        tmp1 = (uintptr_t) d_globalpStart;
        tmp2 = (uintptr_t) d_yVector;
        tmp3 = tmp2 - tmp1;
        //printf("-----> sizeof dataTransferSizePart1 %zu\n", tmp3);
        cudaMemcpy(d_globalpStart, globalpStart, (size_t)tmp3, cudaMemcpyHostToDevice); 
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy dataTransferSizePart1: %s\n", cudaGetErrorString(error));
            printf("%d\n", countr);
	    // we can't recover from the error -- exit the program
	    exit(1);
	} 

        tmp1 = (uintptr_t) d_eign;
        tmp2 = (uintptr_t) d_globalScaler; //d_globalScaler
        tmp3 = tmp2 - tmp1;      
        //printf("-----> sizeof dataTransferSizePart2 %zu\n", tmp3);
        cudaMemcpy(d_eign, tr->partitionData[0].EIGN, (size_t)tmp3, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy dataTransferSizePart2: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
    
#else
        tmp1 = (uintptr_t) d_gammaRates;
        tmp2 = (uintptr_t) d_globalScaler; 
        tmp3 = tmp2 - tmp1;      
        //printf("-----> sizeof dataTransferSizePart2 %zu\n", tmp3);
        cudaMemcpy(d_gammaRates, tr->partitionData[0].gammaRates, (size_t)tmp3, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy gammaRates: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	} 
        
        tmp1 = (uintptr_t) d_globalpStart;
        tmp2 = (uintptr_t) d_wgt;
        tmp3 = tmp2 - tmp1;
        //printf("-----> sizeof dataTransfer1 newz to device %zu\n", tmp3);
        cudaMemcpy(d_globalpStart, globalpStart, (size_t)tmp3, cudaMemcpyHostToDevice); 
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy everyNewviewToGpu1: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	} 
        
        
        tmp1 = (uintptr_t) d_ti;
        tmp2 = (uintptr_t) d_yVector; //d_globalScaler
        tmp3 = tmp2 - tmp1;      
        //printf("-----> sizeof dataTransfer2 newz to device %zu\n", tmp3);
        cudaMemcpy(d_ti, tr->td[0].ti, (size_t)tmp3, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy everyNewviewToGpu2: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	} 
    
#endif

    }
    
#ifdef cudaEvent
    cudaEventRecord(stopT2D, 0);
#endif
    
/*
    partTime2 = rdtsc();
    
    partTmp1 = partTime2 - partTime1;

    fullTimeTr += partTmp1/(double)292700000;
*/
    /* MEMORG
     * 
     * printf("\n!!!!STEP 1\n");
     * d_wgt            tr->cdta->aliaswgt      wgt_size
     * d_patrat         tr->cdta->patrat        patrat_size 
     * d_rateCategory   tr->cdta->rateCategory  rateCategory_size
     * d_wr             tr->cdta->wr_FLOAT          wr_size
     * d_wr2            tr->cdta->wr2_FLOAT         wr2_size
     * 
     * 
     * printf("\n!!!!STEP 2\n");
     * d_ti             tr->td[0].ti            traversalInfo_size 
     *
     * //
     * printf("\n!!!!STEP 3\n");
     * d_yVector
     * // 
     * 
     * printf("STEP 3a-\n");
     * d_eign           tr->partitionData[0].EIGN       ei_size
     * d_ei             tr->partitionData[0].EI         eign_size
     * 
     * printf("STEP 3a\n");
     * d_EV             tr->partitionData[0].EV_FLOAT           EV_size
     * d_tipVector      tr->partitionData[0].tipVector_FLOAT    tipVector_size
     * d_gammaRates     tr->partitionData[i].gammaRates         gammaRates_size
     * d_globalScaler   tr->partitionData[i].globalScaler       globalScaler_size
     //wrong
     * d_left           tr->partitionData[i].left_FLOAT         left_size
     * d_right          tr->partitionData[i].right_FLOAT        right_size
     *  
     * printf("\n!!!!STEP 4\n");
     * d_xVector        tr->partitionData[0].xVector_FLOAT      xVector_size
     * 
     * printf("\n!!!!STEP 4a\n");
     * likelihoodArray_FLOAT
     */
        


    int pIsTip; 
    int qIsTip; 

    pIsTip = isTip(tr->td[0].ti[0].pNumber, tr->mxtips);
    qIsTip = isTip(tr->td[0].ti[0].qNumber, tr->mxtips);

    /*<<<<<<<<<<<<<<<<<<<KERNEL>>>>>>>>>>>>>>>*/
    
    assert(tr->mxtips == nofSpecies);
    assert(tr->partitionData[0].width == alignLength);
    assert(!(pIsTip&&qIsTip));
#ifdef cudaEvent
    cudaEventRecord(startK, 0);
#endif
    kernelnewViewEvaluate(countr, pIsTip, qIsTip, tr->executeModel[0]);
    
    timesCalledEvaluate++;
#ifdef cudaEvent
    cudaEventRecord(stopK, 0);
#endif
    /*<<<<<<<<<<<<<<<<<<<KERNEL>>>>>>>>>>>>>>>*/
    
    countr++;
    /*
     * transfer data back to host [xVector] (todo: check the rest for changed values)
     * d_xVector[0] == d_xVector + tr->innerNodes 107
     * transfer only nodes p, q to compute likellihood score. INNER xVector NODES to opt???
     * 
     */
  
    


#ifdef transferTopLikelihoodVectors

    
    if(!isTip(pNumber, tr->mxtips))
    {
        //d_pointer = (float *) (d_xVector + tr->innerNodes);
        //d_pointer = d_pointer + (pNumber - tr->mxtips -1)*memoryRequirements;
        cudaMemcpy(tr->partitionData[0].xVector_FLOAT[pNumber - tr->mxtips -1], xVector_dp[pNumber - tr->mxtips -1], dataTransferSizeToHost, cudaMemcpyDeviceToHost);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // something's gone wrong
            // print out the CUDA error as a string
            printf("CUDA Error AFTER cudaMemcpy pNumber: %s\n", cudaGetErrorString(error));

            // we can't recover from the error -- exit the program
            exit(1);
        }
    }
    
    if(!isTip(qNumber, tr->mxtips))
    {
        //d_pointer = (float *) d_xVector + tr->innerNodes;
        //d_pointer = d_pointer + (qNumber - tr->mxtips -1)*memoryRequirements;
        cudaMemcpy(tr->partitionData[0].xVector_FLOAT[qNumber - tr->mxtips -1], xVector_dp[qNumber - tr->mxtips -1], dataTransferSizeToHost, cudaMemcpyDeviceToHost);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // something's gone wrong
            // print out the CUDA error as a string
            printf("CUDA Error AFTER cudaMemcpy qNumber: %s\n", cudaGetErrorString(error));
            printf("%d\n", countr);
            // we can't recover from the error -- exit the program
            exit(1);
        }
    }    
#endif
        //kernelScale(d_scalerThread, d_globalScaler);
   //cudaThreadSynchronize();


    
    //partTime1 = rdtsc();
    
        
        
#ifdef trxVback
    /*transfer back xVector*/
    tmp1 = (uintptr_t) d_right;
    tmp2 = (uintptr_t) d_globalpEnd;
    tmp3 = tmp2 - tmp1;
        
        cudaMemcpy( tr->partitionData[0].right_FLOAT, d_right, (size_t)tmp3, cudaMemcpyDeviceToHost);
        //cudaMemcpy(d_globalpStart, globalpStart, dataTransferSizeInit, cudaMemcpyHostToDevice);
	
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy trxVback: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
    
#endif //trxVback
       
    //to gbg!!
    int pNumber, qNumber;
    pNumber = tr->td[0].ti[0].pNumber;
    qNumber = tr->td[0].ti[0].qNumber;        

#ifdef cudaEvent
    cudaEventRecord(startT2H, 0);
#endif
    
    cudaMemcpy(h_partitionLikelihood, d_partitionLikelihood, alignLength*sizeof(double), cudaMemcpyDeviceToHost);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("iter:%d\n", countr);
            printf("pTip:%d qTip:%d\n", pIsTip, qIsTip);
            printf("qNumber - tr->mxtips - 1: %d\n", qNumber - tr->mxtips - 1);
            printf("pNumber - tr->mxtips - 1: %d\n", pNumber - tr->mxtips - 1);

            // something's gone wrong
            // print out the CUDA error as a string
            printf("CUDA Error AFTER cudaMemcpy partitionLikelihood: %s\n", cudaGetErrorString(error));

            // we can't recover from the error -- exit the program
            exit(1);
        }
        
#ifdef cudaEvent        
    cudaEventRecord(stopT2H, 0);
    cudaEventSynchronize(stopT2H);

    cudaEventElapsedTime(&tmpTime, startT2D, stopT2D);
    T2Devaluate +=tmpTime;
    cudaEventElapsedTime(&tmpTime, startK, stopK);
    KenrelTevaluate +=tmpTime;
    cudaEventElapsedTime(&tmpTime, startT2H, stopT2H);
    T2Hevaluate +=tmpTime;        
#endif
    
/*
    partTime2 = rdtsc();
    fullTime2 = rdtsc();
    
    partTmp2 = partTime2 - partTime1;
    
    fullTimeTr += (double)partTmp2/(double)292700000; 
    
    fullTimeKe += (double)((fullTime2 - fullTime1)- (partTmp1 + partTmp2))/(double)2927000000;
*/
    

  /* 
   * end GPU execution
   */

  


}

#endif //MEMORG



void newviewIterative (tree *tr)
{
 #ifdef cudaEvent 
    cudaEvent_t startK, stopK, startT2D, stopT2D, startT2H, stopT2H;
    float tmpTime;
   
    cudaEventCreate(&startK);
    cudaEventCreate(&stopK);
    cudaEventCreate(&startT2D);
    cudaEventCreate(&stopT2D);
    cudaEventCreate(&startT2H);
    cudaEventCreate(&stopT2H);
#endif
    
#ifndef everyNewviewToGpu
    cudaError_t error;
    
    uintptr_t tmp1;
    uintptr_t tmp2;
    uintptr_t tmp3;   
    
#ifdef cudaEvent    
    cudaEventRecord(startT2D, 0);
#endif
    
#ifdef reduseMemTransf_onlyTransGammaRates
    tmp1 = (uintptr_t) d_globalpStart;
    tmp2 = (uintptr_t) d_yVector;
    tmp3 = tmp2 - tmp1;
        //printf("-----> sizeof dataTransferSizePart1 %zu\n", tmp3);
        cudaMemcpy(d_globalpStart, globalpStart, (size_t)tmp3, cudaMemcpyHostToDevice); 
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy everyNewviewToGpu1: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	} 

        tmp1 = (uintptr_t) d_eign;
        tmp2 = (uintptr_t) d_globalScaler; //d_globalScaler
        tmp3 = tmp2 - tmp1;      
        //printf("-----> sizeof dataTransferSizePart2 %zu\n", tmp3);
        cudaMemcpy(d_eign, tr->partitionData[0].EIGN, (size_t)tmp3, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy everyNewviewToGpu2: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	}
#else/*
        tmp1 = (uintptr_t) d_globalpStart;
        tmp2 = (uintptr_t) d_wgt;
        tmp3 = tmp2 - tmp1;
        //printf("-----> sizeof dataTransfer1 newz to device %zu\n", tmp3);
        cudaMemcpy(d_globalpStart, globalpStart, (size_t)tmp3, cudaMemcpyHostToDevice); 
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy everyNewviewToGpu1: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	} 
        */
        
        tmp1 = (uintptr_t) d_ti;
        tmp2 = (uintptr_t) d_yVector; //d_globalScaler
        tmp3 = tmp2 - tmp1;      
        //printf("-----> sizeof dataTransfer2 newz to device %zu\n", tmp3);
        cudaMemcpy(d_ti, tr->td[0].ti, (size_t)tmp3, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaMemcpy everyNewviewToGpu2: %s\n", cudaGetErrorString(error));
            
	    // we can't recover from the error -- exit the program
	    exit(1);
	}    
        
        
#endif
        
#ifdef cudaEvent        
    cudaEventRecord(stopT2D, 0);
    
    cudaEventRecord(startK, 0);
    
#endif
    kernelNewview(tr);
    
 
    timesCalledNewView++;
#ifdef cudaEvent 
    cudaEventRecord(stopK, 0);
#endif
    

        
#ifdef cudaEvent        
    cudaEventRecord(startT2H, 0);
#endif
    
    cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
	    // something's gone wrong
	    // print out the CUDA error as a string
	    printf("CUDA Error AFTER cudaDeviceSynchronize barrier: %s\n", cudaGetErrorString(error));
		
	    // we can't recover from the error -- exit the program
	    exit(1);
	}  
        

#ifdef cudaEvent
    cudaEventRecord(stopT2H, 0);
    cudaEventSynchronize(stopT2H);

    cudaEventElapsedTime(&tmpTime, startT2D, stopT2D);
    T2DnewView +=tmpTime;
    cudaEventElapsedTime(&tmpTime, startK, stopK);
    KenrelTnewView +=tmpTime;
    cudaEventElapsedTime(&tmpTime, startT2H, stopT2H);
    T2HnewView +=tmpTime;
#endif
        
#else //everyNewviewToGpu
    
#ifndef oncpu
  traversalInfo *ti   = tr->td[0].ti;
  int i, model;
  
  for(i = 1; i < tr->td[0].count; i++)
    {
      traversalInfo *tInfo = &ti[i];

      for(model = 0; model < tr->NumberOfModels; model++)
	{
          
	  if(tr->executeModel[model])
	    {	      
	      double
		*x1_start = (double*)NULL,
		*x2_start = (double*)NULL,
		*x3_start = (double*)NULL,
		*left     = (double*)NULL,
		*right    = (double*)NULL;
	      float
		*x1_start_FLOAT = (float*)NULL,
		*x2_start_FLOAT = (float*)NULL,
		*x3_start_FLOAT = (float*)NULL,
		*left_FLOAT     = (float*)NULL,
		*right_FLOAT    = (float*)NULL;
	      int
		states = tr->partitionData[model].states,
		scalerIncrement = 0,
		*wgt = (int*)NULL,	       
		*ex3 = (int*)NULL;
	      unsigned char
		*tipX1 = (unsigned char *)NULL,
		*tipX2 = (unsigned char *)NULL;
	      double qz, rz;
	      int width =  tr->partitionData[model].width;

	      if(tr->useFastScaling)		
		wgt   =  tr->partitionData[model].wgt;		 		  				
              //printf("tipcase %d \n", tInfo->tipCase);
              

              
              
	      switch(tInfo->tipCase)
		{
		case TIP_TIP:
		  tipX1    = tr->partitionData[model].yVector[tInfo->qNumber];
		  tipX2    = tr->partitionData[model].yVector[tInfo->rNumber];

		  if(!tr->useFloat)
		    x3_start       = tr->partitionData[model].xVector[tInfo->pNumber - tr->mxtips - 1];
		  else
                    //printf("\nfailed before:\n" );
		    x3_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->pNumber - tr->mxtips - 1];
                    //printf("\nfailed after:\n" );
		  if(!tr->useFastScaling)
		    {        
		      int k;
		      ex3      = tr->partitionData[model].expVector[tInfo->pNumber - tr->mxtips - 1];

		      for(k = 0; k < width; k++)
			ex3[k] = 0;
		    }

		  break;
		case TIP_INNER:
		  tipX1    =  tr->partitionData[model].yVector[tInfo->qNumber];

		  if(!tr->useFloat)
		    {
		      x2_start       = tr->partitionData[model].xVector[tInfo->rNumber - tr->mxtips - 1];
		      x3_start       = tr->partitionData[model].xVector[tInfo->pNumber - tr->mxtips - 1];
		    }
		  else
		    {
		      x2_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->rNumber - tr->mxtips - 1];		 		    		 
		      x3_start_FLOAT =  tr->partitionData[model].xVector_FLOAT[tInfo->pNumber - tr->mxtips - 1];
		    }

		  if(!tr->useFastScaling)
		    {
		      int 
			k,
			*ex2;
		      
		      ex2      = tr->partitionData[model].expVector[tInfo->rNumber - tr->mxtips - 1];
		      ex3      = tr->partitionData[model].expVector[tInfo->pNumber - tr->mxtips - 1];
		      
		      for(k = 0; k < width; k++)
			ex3[k] = ex2[k];
		    }

		  break;
		case INNER_INNER:

		  if(!tr->useFloat)
		    {
		      x1_start       = tr->partitionData[model].xVector[tInfo->qNumber - tr->mxtips - 1];
		      x2_start       = tr->partitionData[model].xVector[tInfo->rNumber - tr->mxtips - 1];
		      x3_start       = tr->partitionData[model].xVector[tInfo->pNumber - tr->mxtips - 1];
		    }		      
		  else
		    {
		      x1_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->qNumber - tr->mxtips - 1];		  
		      x2_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->rNumber - tr->mxtips - 1];		 
		      x3_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->pNumber - tr->mxtips - 1];
		    }
                  //printf("tr->useFastScaling %d", (int)tr->useFastScaling);
		  //tr->useFastScaling => 1
                  if(!tr->useFastScaling)
		    {
		      int 
			k,
			*ex1,
			*ex2;

		      ex1      = tr->partitionData[model].expVector[tInfo->qNumber - tr->mxtips - 1];
		      ex2      = tr->partitionData[model].expVector[tInfo->rNumber - tr->mxtips - 1];
		      ex3      = tr->partitionData[model].expVector[tInfo->pNumber - tr->mxtips - 1];
		      
		      for(k = 0; k < width; k++)
			ex3[k] = ex1[k] + ex2[k];
		    }		  
		  break;
		default:
		  assert(0);
		}

	      if(tr->useFloat)
		{
		  left_FLOAT = tr->partitionData[model].left_FLOAT;
		  right_FLOAT = tr->partitionData[model].right_FLOAT;
		}
	      else
		{
		  left  = tr->partitionData[model].left;
		  right = tr->partitionData[model].right;
		}

	      if(tr->multiBranch)
		{
                  
		  qz = tInfo->qz[model];
		  rz = tInfo->rz[model];
		}
	      else
		{
                  
		  qz = tInfo->qz[0];
		  rz = tInfo->rz[0];
		}
              


	      switch(tr->partitionData[model].dataType)
		{
		case BINARY_DATA:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {			
			makeP(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN, tr->NumberOfCategories,
			      left, right, BINARY_DATA);

			newviewGTRCAT_BINARY(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					     x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					     ex3, tipX1, tipX2,
					     width, left, right, wgt, &scalerIncrement, tr->useFastScaling
					     );		
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
			      4, left, right, BINARY_DATA);

			newviewGTRGAMMA_BINARY(tInfo->tipCase,
					       x1_start, x2_start, x3_start, tr->partitionData[model].EV, tr->partitionData[model].tipVector,
					       ex3, tipX1, tipX2,
					       width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case DNA_DATA:		  
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      if(tr->useFloat)
			{			  
			   makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);
#ifdef checkLR
                           if (i==1 && countr==1)
                           {
                               printf("\nhost left[4]= %f\n", left_FLOAT[4]);
                               printf("device left[4]= %f\n", checkL[4]);
                           }
#endif

			   newviewGTRCAT_FLOAT(tInfo->tipCase,  tr->partitionData[model].EV_FLOAT, tr->partitionData[model].rateCategory,
					       x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
					       ex3, tipX1, tipX2,
					       width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling
					       );			  
			}
		      else
			{			  
			  makeP(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				tr->partitionData[model].EIGN, tr->NumberOfCategories,
				left, right, DNA_DATA);

			  newviewGTRCAT(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					ex3, tipX1, tipX2,
					width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      if(tr->useFloat)
			{
			  makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, DNA_DATA);
#ifdef checkLR
                           if (i==1 && countr==1)
                           {
                               printf("\nhost left[4]= %f\n", left_FLOAT[4]);
                               printf("device left[4]= %f\n", checkL[4]);
                           }
#endif
			  newviewGTRGAMMA_FLOAT(tInfo->tipCase,
						x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].EV_FLOAT, tr->partitionData[model].tipVector_FLOAT,
						ex3, tipX1, tipX2,
						width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);			

			}
		      else
			{			 
			  makeP(qz, rz, tr->partitionData[model].gammaRates,
				tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				4, left, right, DNA_DATA);

			  newviewGTRGAMMA(tInfo->tipCase,
					  x1_start, x2_start, x3_start, tr->partitionData[model].EV, tr->partitionData[model].tipVector,
					  ex3, tipX1, tipX2,
					  width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      break;
		    default:
		      assert(0);
		      }
		  break;
		case AA_DATA:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      if(tr->useFloat)
			{			 
			  makeP_FLOAT(qz, rz, tr->cdta->patrat,
				      tr->partitionData[model].EI,
				      tr->partitionData[model].EIGN,
				      tr->NumberOfCategories, left_FLOAT, right_FLOAT, AA_DATA);			 			 
			  
			   newviewGTRCATPROT_FLOAT(tInfo->tipCase,  tr->partitionData[model].EV_FLOAT, tr->partitionData[model].rateCategory,
						   x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
						   ex3, tipX1, tipX2, width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);	
			}
		      else
			{
			  makeP(qz, rz, tr->cdta->patrat,
				tr->partitionData[model].EI,
				tr->partitionData[model].EIGN,
				tr->NumberOfCategories, left, right, AA_DATA);
			  
			  newviewGTRCATPROT(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					    x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					    ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);			
			}
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      if(tr->useFloat)
			{
			  makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI,
				      tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, AA_DATA);

			  newviewGTRGAMMAPROT_FLOAT(tInfo->tipCase,
						    x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT,
						    tr->partitionData[model].EV_FLOAT,
						    tr->partitionData[model].tipVector_FLOAT,
						    ex3, tipX1, tipX2,
						    width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      else
			{
			  makeP(qz, rz, tr->partitionData[model].gammaRates,
				tr->partitionData[model].EI,
				tr->partitionData[model].EIGN,
				4, left, right, AA_DATA);

			  newviewGTRGAMMAPROT(tInfo->tipCase,
					      x1_start, x2_start, x3_start,
					      tr->partitionData[model].EV,
					      tr->partitionData[model].tipVector,
					      ex3, tipX1, tipX2,
					      width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case SECONDARY_DATA_6:		 
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP(qz, rz, tr->cdta->patrat,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      tr->NumberOfCategories, left, right, SECONDARY_DATA_6);

			newviewGTRCATSECONDARY_6(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
						 x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
						 ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      4, left, right, SECONDARY_DATA_6);

			newviewGTRGAMMASECONDARY_6(tInfo->tipCase,
						   x1_start, x2_start, x3_start,
						   tr->partitionData[model].EV,
						   tr->partitionData[model].tipVector,
						   ex3, tipX1, tipX2,
						   width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case SECONDARY_DATA_7:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP(qz, rz, tr->cdta->patrat,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      tr->NumberOfCategories, left, right, SECONDARY_DATA_7);

			newviewGTRCATSECONDARY_7(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
						 x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
						 ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      4, left, right, SECONDARY_DATA_7);

			newviewGTRGAMMASECONDARY_7(tInfo->tipCase,
						   x1_start, x2_start, x3_start,
						   tr->partitionData[model].EV,
						   tr->partitionData[model].tipVector,
						   ex3, tipX1, tipX2,
						   width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case SECONDARY_DATA:
		 switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP(qz, rz, tr->cdta->patrat,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      tr->NumberOfCategories, left, right, SECONDARY_DATA);

			newviewGTRCATSECONDARY(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					       x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					       ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      4, left, right, SECONDARY_DATA);

			newviewGTRGAMMASECONDARY(tInfo->tipCase,
						 x1_start, x2_start, x3_start,
						 tr->partitionData[model].EV,
						 tr->partitionData[model].tipVector,
						 ex3, tipX1, tipX2,
						 width, left, right, wgt, &scalerIncrement, tr->useFastScaling);		
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case GENERIC_32:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP_Flex(qz, rz, tr->cdta->patrat,
				   tr->partitionData[model].EI,
				   tr->partitionData[model].EIGN,
				   tr->NumberOfCategories, left, right, states);
			
			newviewFlexCat(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
				       x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
				       ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling, states);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP_Flex(qz, rz, tr->partitionData[model].gammaRates,
				   tr->partitionData[model].EI,
				   tr->partitionData[model].EIGN,
				   4, left, right, states);

			newviewFlexGamma(tInfo->tipCase,
					 x1_start, x2_start, x3_start,
					 tr->partitionData[model].EV,
					 tr->partitionData[model].tipVector,
					 ex3, tipX1, tipX2,
					 width, left, right, wgt, &scalerIncrement, tr->useFastScaling, states);		
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case GENERIC_64:
		  break;
		default:
		  assert(0);
		}
	      if(tr->useFastScaling)
		{
		  tr->partitionData[model].globalScaler[tInfo->pNumber] = 
		    tr->partitionData[model].globalScaler[tInfo->qNumber] + 
		    tr->partitionData[model].globalScaler[tInfo->rNumber] +
		    (unsigned int)scalerIncrement;
		  assert(tr->partitionData[model].globalScaler[tInfo->pNumber] < INT_MAX);
		}
	    }
	}



    }
#endif
#endif //everyNewviewToGpu
}

void newviewIterativeMulti (tree *tr)
{ 
  int i, model;

  assert(tr->multiBranch);

  for(model = 0; model < tr->NumberOfModels; model++)    
    {         
      if(tr->executeModel[model])
	{	      
	  double
	    *x1_start = (double*)NULL,
	    *x2_start = (double*)NULL,
	    *x3_start = (double*)NULL,
	    *left     = (double*)NULL,
	    *right    = (double*)NULL;
	  float
	    *x1_start_FLOAT = (float*)NULL,
	    *x2_start_FLOAT = (float*)NULL,
	    *x3_start_FLOAT = (float*)NULL,
	    *left_FLOAT     = (float*)NULL,
	    *right_FLOAT    = (float*)NULL;
	  int		
	    scalerIncrement = 0,
	    *wgt = (int*)NULL,	       
	    *ex3 = (int*)NULL;
	  unsigned char
	    *tipX1 = (unsigned char *)NULL,
	    *tipX2 = (unsigned char *)NULL;
	  double qz, rz;
	  int width =  tr->partitionData[model].width;
	  traversalInfo *ti   = tr->td[model].ti;

	  if(tr->useFastScaling)		
	    wgt   =  tr->partitionData[model].wgt;		 		  				
	  
	  for(i = 1; i < tr->td[model].count; i++)
	    {    	      
	      traversalInfo *tInfo = &ti[i];

	      switch(tInfo->tipCase)
		{
		case TIP_TIP:
		  tipX1    = tr->partitionData[model].yVector[tInfo->qNumber];
		  tipX2    = tr->partitionData[model].yVector[tInfo->rNumber];

		  if(!tr->useFloat)
		    x3_start       = tr->partitionData[model].xVector[tInfo->pNumber - tr->mxtips - 1];
		  else
		    x3_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->pNumber - tr->mxtips - 1];

		  if(!tr->useFastScaling)
		    {
		      int k;
		      ex3      = tr->partitionData[model].expVector[tInfo->pNumber - tr->mxtips - 1];

		      for(k = 0; k < width; k++)
			ex3[k] = 0;
		    }

		  break;
		case TIP_INNER:
		  tipX1    =  tr->partitionData[model].yVector[tInfo->qNumber];

		  if(!tr->useFloat)
		    {
		      x2_start       = tr->partitionData[model].xVector[tInfo->rNumber - tr->mxtips - 1];
		      x3_start       = tr->partitionData[model].xVector[tInfo->pNumber - tr->mxtips - 1];
		    }
		  else
		    {
		      x2_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->rNumber - tr->mxtips - 1];		 		    		 
		      x3_start_FLOAT =  tr->partitionData[model].xVector_FLOAT[tInfo->pNumber - tr->mxtips - 1];
		    }

		  if(!tr->useFastScaling)
		    {
		      int 
			k,
			*ex2;
		      
		      ex2      = tr->partitionData[model].expVector[tInfo->rNumber - tr->mxtips - 1];
		      ex3      = tr->partitionData[model].expVector[tInfo->pNumber - tr->mxtips - 1];
		      
		      for(k = 0; k < width; k++)
			ex3[k] = ex2[k];
		    }

		  break;
		case INNER_INNER:

		  if(!tr->useFloat)
		    {
		      x1_start       = tr->partitionData[model].xVector[tInfo->qNumber - tr->mxtips - 1];
		      x2_start       = tr->partitionData[model].xVector[tInfo->rNumber - tr->mxtips - 1];
		      x3_start       = tr->partitionData[model].xVector[tInfo->pNumber - tr->mxtips - 1];
		    }		      
		  else
		    {
		      x1_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->qNumber - tr->mxtips - 1];		  
		      x2_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->rNumber - tr->mxtips - 1];		 
		      x3_start_FLOAT = tr->partitionData[model].xVector_FLOAT[tInfo->pNumber - tr->mxtips - 1];
		    }

		  if(!tr->useFastScaling)
		    {
		      int 
			k,
			*ex1,
			*ex2;

		      ex1      = tr->partitionData[model].expVector[tInfo->qNumber - tr->mxtips - 1];
		      ex2      = tr->partitionData[model].expVector[tInfo->rNumber - tr->mxtips - 1];
		      ex3      = tr->partitionData[model].expVector[tInfo->pNumber - tr->mxtips - 1];
		      
		      for(k = 0; k < width; k++)
			ex3[k] = ex1[k] + ex2[k];
		    }		  
		  break;
		default:
		  assert(0);
		}

	      if(tr->useFloat)
		{
		  left_FLOAT = tr->partitionData[model].left_FLOAT;
		  right_FLOAT = tr->partitionData[model].right_FLOAT;
		}
	      else
		{
		  left  = tr->partitionData[model].left;
		  right = tr->partitionData[model].right;
		}

	      if(tr->multiBranch)
		{
		  qz = tInfo->qz[model];
		  rz = tInfo->rz[model];
		}
	      else
		{
		  assert(0);
		  qz = tInfo->qz[0];
		  rz = tInfo->rz[0];
		}


	      switch(tr->partitionData[model].dataType)
		{
		case BINARY_DATA:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {			
			makeP(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN, tr->NumberOfCategories,
			      left, right, BINARY_DATA);

			newviewGTRCAT_BINARY(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					     x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					     ex3, tipX1, tipX2,
					     width, left, right, wgt, &scalerIncrement, tr->useFastScaling
					     );		
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
			      4, left, right, BINARY_DATA);

			newviewGTRGAMMA_BINARY(tInfo->tipCase,
					       x1_start, x2_start, x3_start, tr->partitionData[model].EV, tr->partitionData[model].tipVector,
					       ex3, tipX1, tipX2,
					       width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case DNA_DATA:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      if(tr->useFloat)
			{			  
			   makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);


			   newviewGTRCAT_FLOAT(tInfo->tipCase,  tr->partitionData[model].EV_FLOAT, tr->partitionData[model].rateCategory,
					       x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
					       ex3, tipX1, tipX2,
					       width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling
					       );			  
			}
		      else
			{			  
			  makeP(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				tr->partitionData[model].EIGN, tr->NumberOfCategories,
				left, right, DNA_DATA);

			  newviewGTRCAT(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					ex3, tipX1, tipX2,
					width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      if(tr->useFloat)
			{
			  makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, DNA_DATA);

			  newviewGTRGAMMA_FLOAT(tInfo->tipCase,
						x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].EV_FLOAT, tr->partitionData[model].tipVector_FLOAT,
						ex3, tipX1, tipX2,
						width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);			

			}
		      else
			{			 
			  makeP(qz, rz, tr->partitionData[model].gammaRates,
				tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				4, left, right, DNA_DATA);

			  newviewGTRGAMMA(tInfo->tipCase,
					  x1_start, x2_start, x3_start, tr->partitionData[model].EV, tr->partitionData[model].tipVector,
					  ex3, tipX1, tipX2,
					  width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case AA_DATA:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      if(tr->useFloat)
			{			 
			  makeP_FLOAT(qz, rz, tr->cdta->patrat,
				      tr->partitionData[model].EI,
				      tr->partitionData[model].EIGN,
				      tr->NumberOfCategories, left_FLOAT, right_FLOAT, AA_DATA);			 			 
			  
			   newviewGTRCATPROT_FLOAT(tInfo->tipCase,  tr->partitionData[model].EV_FLOAT, tr->partitionData[model].rateCategory,
						   x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
						   ex3, tipX1, tipX2, width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);	
			}
		      else
			{
			  makeP(qz, rz, tr->cdta->patrat,
				tr->partitionData[model].EI,
				tr->partitionData[model].EIGN,
				tr->NumberOfCategories, left, right, AA_DATA);
			  
			  newviewGTRCATPROT(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					    x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					    ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);			
			}
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      if(tr->useFloat)
			{
			  makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI,
				      tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, AA_DATA);

			  newviewGTRGAMMAPROT_FLOAT(tInfo->tipCase,
						    x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT,
						    tr->partitionData[model].EV_FLOAT,
						    tr->partitionData[model].tipVector_FLOAT,
						    ex3, tipX1, tipX2,
						    width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      else
			{
			  makeP(qz, rz, tr->partitionData[model].gammaRates,
				tr->partitionData[model].EI,
				tr->partitionData[model].EIGN,
				4, left, right, AA_DATA);

			  newviewGTRGAMMAPROT(tInfo->tipCase,
					      x1_start, x2_start, x3_start,
					      tr->partitionData[model].EV,
					      tr->partitionData[model].tipVector,
					      ex3, tipX1, tipX2,
					      width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
			}
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case SECONDARY_DATA_6:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP(qz, rz, tr->cdta->patrat,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      tr->NumberOfCategories, left, right, SECONDARY_DATA_6);

			newviewGTRCATSECONDARY_6(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
						 x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
						 ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      4, left, right, SECONDARY_DATA_6);

			newviewGTRGAMMASECONDARY_6(tInfo->tipCase,
						   x1_start, x2_start, x3_start,
						   tr->partitionData[model].EV,
						   tr->partitionData[model].tipVector,
						   ex3, tipX1, tipX2,
						   width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case SECONDARY_DATA_7:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP(qz, rz, tr->cdta->patrat,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      tr->NumberOfCategories, left, right, SECONDARY_DATA_7);

			newviewGTRCATSECONDARY_7(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
						 x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
						 ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      4, left, right, SECONDARY_DATA_7);

			newviewGTRGAMMASECONDARY_7(tInfo->tipCase,
						   x1_start, x2_start, x3_start,
						   tr->partitionData[model].EV,
						   tr->partitionData[model].tipVector,
						   ex3, tipX1, tipX2,
						   width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		case SECONDARY_DATA:
		  switch(tr->rateHetModel)
		    {
		    case CAT:
		      {
			makeP(qz, rz, tr->cdta->patrat,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      tr->NumberOfCategories, left, right, SECONDARY_DATA);

			newviewGTRCATSECONDARY(tInfo->tipCase,  tr->partitionData[model].EV, tr->partitionData[model].rateCategory,
					       x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
					       ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
		      }
		      break;
		    case GAMMA:
		    case GAMMA_I:
		      {
			makeP(qz, rz, tr->partitionData[model].gammaRates,
			      tr->partitionData[model].EI,
			      tr->partitionData[model].EIGN,
			      4, left, right, SECONDARY_DATA);

			newviewGTRGAMMASECONDARY(tInfo->tipCase,
						 x1_start, x2_start, x3_start,
						 tr->partitionData[model].EV,
						 tr->partitionData[model].tipVector,
						 ex3, tipX1, tipX2,
						 width, left, right, wgt, &scalerIncrement, tr->useFastScaling);		
		      }
		      break;
		    default:
		      assert(0);
		    }
		  break;
		default:
		  assert(0);
		}
	      if(tr->useFastScaling)
		{
		  tr->partitionData[model].globalScaler[tInfo->pNumber] = 
		    tr->partitionData[model].globalScaler[tInfo->qNumber] + 
		    tr->partitionData[model].globalScaler[tInfo->rNumber] +
		    (unsigned int)scalerIncrement;
		  assert(tr->partitionData[model].globalScaler[tInfo->pNumber] < INT_MAX);
		}
	    }
	}
    }
}


void newviewGeneric (tree *tr, nodeptr p)
{  
  if(isTip(p->number, tr->mxtips))
    return;

  if(tr->multiGene)
    {	           
      int i;
      for(i = 0; i < tr->NumberOfModels; i++)
	{
	  if(tr->executeModel[i])
	    {
	      tr->td[i].count = 1; 
	      computeTraversalInfoMulti(p, &(tr->td[i].ti[0]), &(tr->td[i].count), tr->mxtips, i); 
	    }
	}
      /* if(tr->td[i].count > 1)*/
      newviewIterativeMulti(tr);
    }
  else
    {
      tr->td[0].count = 1;
      computeTraversalInfo(p, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
      
      if(tr->td[0].count > 1)
	{
#ifdef _USE_PTHREADS
	  masterBarrier(THREAD_NEWVIEW, tr);
#else
	  newviewIterative(tr);
#endif
	}
    }
}


void newviewGenericMulti (tree *tr, nodeptr p, int model)
{  
  assert(tr->multiGene);

  assert(!isTip(p->number, tr->mxtips)); 
  
  {	           
    int i;
    
    assert(p->backs[model]);

    for(i = 0; i < tr->NumberOfModels; i++)
      tr->executeModel[i] = FALSE;
    tr->executeModel[model] = TRUE;

   	   
    tr->td[model].count = 1; 
    computeTraversalInfoMulti(p, &(tr->td[model].ti[0]), &(tr->td[model].count), tr->mxtips, model);    
    
    /* if(tr->td[i].count > 1)*/
    newviewIterativeMulti(tr);

   for(i = 0; i < tr->NumberOfModels; i++)
      tr->executeModel[i] = TRUE; 
  }
}

void newviewGenericMasked(tree *tr, nodeptr p)
{
  if(isTip(p->number, tr->mxtips))
    return;

  {
    int i;

    for(i = 0; i < tr->NumberOfModels; i++)
      {
	if(tr->partitionConverged[i])
	  tr->executeModel[i] = FALSE;
	else
	  tr->executeModel[i] = TRUE;
      }
    
    if(tr->multiGene)
      {
	for(i = 0; i < tr->NumberOfModels; i++)
	  {
	    if(tr->executeModel[i])
	      {
		tr->td[i].count = 1; 
		computeTraversalInfoMulti(p, &(tr->td[i].ti[0]), &(tr->td[i].count), tr->mxtips, i); 
	      }
	    else
	      tr->td[i].count = 0; 
	  }
	/* if(tr->td[i].count > 1) ? */
	newviewIterativeMulti(tr);
      }
    else
      {
	tr->td[0].count = 1;
	computeTraversalInfo(p, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);

	if(tr->td[0].count > 1)
	  {
#ifdef _USE_PTHREADS
	    masterBarrier(THREAD_NEWVIEW_MASKED, tr);
#else
	    newviewIterative(tr);
#endif
	  }
      }

    for(i = 0; i < tr->NumberOfModels; i++)
      tr->executeModel[i] = TRUE;
  }
}


#ifdef  _USE_PTHREADS

static void newviewMultiGrain(tree *tr,  double *x1, double *x2, double *x3, int *_ex1, int *_ex2, int *_ex3, unsigned char *_tipX1, unsigned char *_tipX2, 
			      int tipCase, double *_pz, double *_qz)
{
  int    
    scalerIncrement = 0,   
    model,         
    columnCounter = 0,
    offsetCounter = 0;

  assert(!tr->useFastScaling && !tr->useFloat);

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      double
	*x1_start = (double*)NULL,
	*x2_start = (double*)NULL,
	*x3_start = (double*)NULL,
	*left     = tr->partitionData[model].left,
	*right    = tr->partitionData[model].right,
	pz, qz;

      int
	*wgt          = &tr->contiguousWgt[columnCounter],
	*rateCategory = &tr->contiguousRateCategory[columnCounter],
	*ex1 = (int*)NULL,
	*ex2 = (int*)NULL,
	*ex3 = (int*)NULL,
	width = tr->partitionData[model].upper - tr->partitionData[model].lower;

      unsigned char
	*tipX1 = (unsigned char *)NULL,
	*tipX2 = (unsigned char *)NULL;     

      switch(tipCase)
	{
	case TIP_TIP:    
	  tipX1 =    &_tipX1[columnCounter];
	  tipX2 =    &_tipX2[columnCounter];
	  ex3   =    &_ex3[columnCounter];
	  x3_start = &x3[offsetCounter];
	 
	  if(!tr->useFastScaling)
	    {
	      int k;	      
	      
	      for(k = 0; k < width; k++)
		ex3[k] = 0;
	    }
	  break;
	case TIP_INNER:
	  tipX1 =    &_tipX1[columnCounter];

	  ex2   =    &_ex2[columnCounter];
	  x2_start = &x2[offsetCounter];
	  
	  ex3   =    &_ex3[columnCounter];
	  x3_start = &x3[offsetCounter];
	 
	  if(!tr->useFastScaling)
	    {
	      int k;	      
	      
	      for(k = 0; k < width; k++)
		ex3[k] = ex2[k];
	    }
	  break;
	case INNER_INNER:
	  ex1   =    &_ex1[columnCounter];
	  x1_start = &x1[offsetCounter];

	  ex2   =    &_ex2[columnCounter];
	  x2_start = &x2[offsetCounter];
	  
	  ex3   =    &_ex3[columnCounter];
	  x3_start = &x3[offsetCounter];
	 
	  if(!tr->useFastScaling)
	    {
	      int k;	      
	      
	      for(k = 0; k < width; k++)
		ex3[k] = ex1[k] + ex2[k];
	    }
	  break;
	default:
	  assert(0);
	}    

      if(tr->multiBranch)
	{
	  pz = _pz[model];
	  pz = (pz > zmin) ? log(pz) : log(zmin);
	  
	  qz = _qz[model];
	  qz = (qz > zmin) ? log(qz) : log(zmin);
	}
      else
	{	  
	  pz = _pz[0];
	  pz = (pz > zmin) ? log(pz) : log(zmin);

	  qz = _qz[0];
	  qz = (qz > zmin) ? log(qz) : log(zmin);
	}

      

      switch(tr->partitionData[model].dataType)
	{
	case BINARY_DATA:
	  switch(tr->rateHetModel)
	    {
	    case CAT:	      
	      makeP(pz, qz, tr->cdta->patrat,   tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN, tr->NumberOfCategories,
		    left, right, BINARY_DATA);
	      
	      newviewGTRCAT_BINARY(tipCase,  tr->partitionData[model].EV, rateCategory,
				   x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
				   ex3, tipX1, tipX2,
				   width, left, right, wgt, &scalerIncrement, tr->useFastScaling
				   );	      	      
	      break;
	    case GAMMA:
	    case GAMMA_I:	      		
	      makeP(pz, qz, tr->partitionData[model].gammaRates,
		    tr->partitionData[model].EI, tr->partitionData[model].EIGN,
		    4, left, right, BINARY_DATA);
	      
	      newviewGTRGAMMA_BINARY(tipCase,
				     x1_start, x2_start, x3_start, tr->partitionData[model].EV, tr->partitionData[model].tipVector,
				     ex3, tipX1, tipX2,
				     width, left, right, wgt, &scalerIncrement, tr->useFastScaling);	      	      
	      break;
	    default:
	      assert(0);
	    }
	  break;
	case DNA_DATA:
	  switch(tr->rateHetModel)
	    {
	    case CAT:	      			
	      makeP(pz, qz, tr->cdta->patrat,   tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN, tr->NumberOfCategories,
		    left, right, DNA_DATA);
	      
	      newviewGTRCAT(tipCase,  tr->partitionData[model].EV, rateCategory,
			    x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
			    ex3, tipX1, tipX2,
			    width, left, right, wgt, &scalerIncrement, tr->useFastScaling
			    );
	      	      
	      break;
	    case GAMMA:
	    case GAMMA_I:	     		
	      makeP(pz, qz, tr->partitionData[model].gammaRates,
		    tr->partitionData[model].EI, tr->partitionData[model].EIGN,
		    4, left, right, DNA_DATA);
	      
	      newviewGTRGAMMA(tipCase,
			      x1_start, x2_start, x3_start, tr->partitionData[model].EV, tr->partitionData[model].tipVector,
			      ex3, tipX1, tipX2,
			      width, left, right, wgt, &scalerIncrement, tr->useFastScaling);			      	     
	      break;
	    default:
	      assert(0);
	    }
	  break;
	case AA_DATA:
	  switch(tr->rateHetModel)
	    {
	    case CAT:	      
	      makeP(pz, qz, tr->cdta->patrat,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    tr->NumberOfCategories, left, right, AA_DATA);
	      
	      newviewGTRCATPROT(tipCase,  tr->partitionData[model].EV, rateCategory,
				x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
				ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);	      	      
	      break;
	    case GAMMA:
	    case GAMMA_I:	      
	      makeP(pz, qz, tr->partitionData[model].gammaRates,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    4, left, right, AA_DATA);
	      
	      newviewGTRGAMMAPROT(tipCase,
				  x1_start, x2_start, x3_start,
				  tr->partitionData[model].EV,
				  tr->partitionData[model].tipVector,
				  ex3, tipX1, tipX2,
				  width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
	      	      
	      break;
	    default:
	      assert(0);
	    }
	  break;
	case SECONDARY_DATA:
	  switch(tr->rateHetModel)
	    {
	    case CAT:	      
	      makeP(pz, qz, tr->cdta->patrat,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    tr->NumberOfCategories, left, right, SECONDARY_DATA);
	      
	      newviewGTRCATSECONDARY(tipCase,  tr->partitionData[model].EV, rateCategory,
				     x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
				     ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
	      	      
	      break;
	    case GAMMA:
	    case GAMMA_I:	      
	      makeP(pz, qz, tr->partitionData[model].gammaRates,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    4, left, right, SECONDARY_DATA);

	      newviewGTRGAMMASECONDARY(tipCase,
				       x1_start, x2_start, x3_start,
				       tr->partitionData[model].EV,
				       tr->partitionData[model].tipVector,
				       ex3, tipX1, tipX2,
				       width, left, right, wgt, &scalerIncrement, tr->useFastScaling);
	    
	      break;
	    default:
	      assert(0);
	    }
	  break;
	case SECONDARY_DATA_6:
	  switch(tr->rateHetModel)
	    {
	    case CAT:		      
	      makeP(pz, qz, tr->cdta->patrat,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    tr->NumberOfCategories, left, right, SECONDARY_DATA_6);
	      
	      newviewGTRCATSECONDARY_6(tipCase,  tr->partitionData[model].EV, rateCategory,
				       x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
				       ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);	     
	      break;
	    case GAMMA:
	    case GAMMA_I:		      
	      makeP(pz, qz, tr->partitionData[model].gammaRates,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    4, left, right, SECONDARY_DATA_6);

	      newviewGTRGAMMASECONDARY_6(tipCase,
					 x1_start, x2_start, x3_start,
					 tr->partitionData[model].EV,
					 tr->partitionData[model].tipVector,
					 ex3, tipX1, tipX2,
					 width, left, right, wgt, &scalerIncrement, tr->useFastScaling);	    
	      break;
	    default:
	      assert(0);
	    }
	  break;
	case SECONDARY_DATA_7:
	  switch(tr->rateHetModel)
	    {
	    case CAT:		      
	      makeP(pz, qz, tr->cdta->patrat,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    tr->NumberOfCategories, left, right, SECONDARY_DATA_7);
	      
	      newviewGTRCATSECONDARY_7(tipCase,  tr->partitionData[model].EV, rateCategory,
				       x1_start, x2_start, x3_start, tr->partitionData[model].tipVector,
				       ex3, tipX1, tipX2, width, left, right, wgt, &scalerIncrement, tr->useFastScaling);	      
	      break;
	    case GAMMA:
	    case GAMMA_I:	      
	      makeP(pz, qz, tr->partitionData[model].gammaRates,
		    tr->partitionData[model].EI,
		    tr->partitionData[model].EIGN,
		    4, left, right, SECONDARY_DATA_7);

	      newviewGTRGAMMASECONDARY_7(tipCase,
					 x1_start, x2_start, x3_start,
					 tr->partitionData[model].EV,
					 tr->partitionData[model].tipVector,
					 ex3, tipX1, tipX2,
					 width, left, right, wgt, &scalerIncrement, tr->useFastScaling);	      
	      break;
	    default:
	      assert(0);
	    }
	  break;
	default:
	  assert(0);
	}
      columnCounter += width;
      offsetCounter += width * tr->partitionData[model].states * tr->discreteRateCategories;
    }
}



void newviewClassify(tree *tr, branchInfo *b, double *z)
{
  int 
    leftNumber = b->epa->leftNodeNumber,
    rightNumber = b->epa->rightNodeNumber,
    tipCase = -1,
    *ex1 = (int*)NULL,
    *ex2 = (int*)NULL,
    *ex3 = tr->temporaryScaling;

  double
    *x1_start = (double*)NULL,
    *x2_start = (double*)NULL,
    *x3_start = tr->temporaryVector;

  unsigned char
    *tipX1 = (unsigned char*)NULL,
    *tipX2 = (unsigned char*)NULL;
  
  if (isTip(leftNumber, tr->mxtips) && isTip(rightNumber, tr->mxtips))
    {	  
      tipCase = TIP_TIP;
      
      tipX1 = tr->contiguousTips[leftNumber];
      tipX2 = tr->contiguousTips[rightNumber];
    }
  else
    {
      if (isTip(leftNumber, tr->mxtips))
	{	      
	  tipCase = TIP_INNER;
	  
	  tipX1 = tr->contiguousTips[leftNumber];

	  x2_start = b->epa->right;
	  ex2      = b->epa->rightScaling;	     	  
	}
      else
	{
	  if(isTip(rightNumber, tr->mxtips))
	    {		  
	      tipCase = TIP_INNER;
	      
	      tipX1 = tr->contiguousTips[rightNumber];
	      
	      x2_start = b->epa->left;
	      ex2      = b->epa->leftScaling;		 
	    }
	  else
	    {
	      tipCase = INNER_INNER;
	      
	      x1_start = b->epa->left;
	      ex1      = b->epa->leftScaling;
	      
	      x2_start = b->epa->right;
	      ex2      = b->epa->rightScaling;	      	    
	    }
	}
    }
    
  newviewMultiGrain(tr,  x1_start, x2_start, x3_start, ex1, ex2, ex3, tipX1, tipX2, 
		    tipCase, z, z);

}



void newviewClassifySpecial(tree *tr, double *x1_start, double *x2_start, double *x3_start, int *ex1, int *ex2, int *ex3,
			    unsigned char *tipX1,  unsigned char *tipX2, int tipCase, double *pz, double *qz)
{
  newviewMultiGrain(tr,  x1_start, x2_start, x3_start, ex1, ex2, ex3, tipX1, tipX2, 
		    tipCase, pz, qz);
  
}




#endif


