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
 *  Alexandros Stamatakis:"RAxML-VI-HPC: maximum likelihood-based phylogenetic analyses with thousands 
 *  of taxa and mixed models". 
 *  Bioinformatics 2006; doi: 10.1093/bioinformatics/btl446
 */

#ifndef WIN32
#include <unistd.h>
#endif

#include <math.h>
#include <time.h> 
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "axml.h"

static const double MNBRAK_GOLD =    1.618034;
static const double MNBRAK_TINY =      1.e-20;
static const double MNBRAK_GLIMIT =     100.0;
static const double BRENT_ZEPS  =      1.e-5;
static const double BRENT_CGOLD =   0.3819660;

extern int optimizeRatesInvocations;
extern int optimizeRateCategoryInvocations;
extern int optimizeAlphaInvocations;
extern int optimizeInvarInvocations;
extern double masterTime;
extern char ratesFileName[1024];
extern char workdir[1024];
extern char run_id[128];
extern char lengthFileName[1024];
extern char lengthFileNameModel[1024];



#ifdef _USE_PTHREADS
extern volatile int             NumberOfThreads;
extern volatile double          *reductionBuffer;
#endif



/*********************FUNCTIONS FOOR EXACT MODEL OPTIMIZATION UNDER GTRGAMMA ***************************************/


static void setRateModel(tree *tr, int model, double rate, int position)
{
  int
    states   = tr->partitionData[model].states,
    numRates = (states * states - states) / 2;

  if(tr->partitionData[model].dataType == DNA_DATA)
    assert(position >= 0 && position < (numRates - 1));
  else
    assert(position >= 0 && position < numRates);

  assert(tr->partitionData[model].dataType != BINARY_DATA);   

  if(tr->partitionData[model].nonGTR)
    {    
      int 
	i, 
	k = tr->partitionData[model].symmetryVector[position];

      assert(tr->partitionData[model].dataType == SECONDARY_DATA ||
	     tr->partitionData[model].dataType == SECONDARY_DATA_6 ||
	     tr->partitionData[model].dataType == SECONDARY_DATA_7);

      if(k == -1)
	tr->partitionData[model].substRates[position] = 0.0;
      else
	{
	  if(k == tr->partitionData[model].symmetryVector[numRates - 1])
	    {
	      for(i = 0; i < numRates - 1; i++)
		if(tr->partitionData[model].symmetryVector[i] == k)
		  tr->partitionData[model].substRates[position] = 1.0;
	    }
	  else
	    {
	      for(i = 0; i < numRates - 1; i++)
		{
		  if(tr->partitionData[model].symmetryVector[i] == k)
		    tr->partitionData[model].substRates[i] = rate; 
		}	      	     
	    }
	}
    }
  else
    tr->partitionData[model].substRates[position] = rate;
}





static linkageList* initLinkageList(int *linkList, tree *tr)
{
  int 
    k,
    partitions,
    numberOfModels = 0,
    i,
    pos;
  linkageList* ll = (linkageList*)malloc(sizeof(linkageList));
      
  for(i = 0; i < tr->NumberOfModels; i++)
    {
      if(linkList[i] > numberOfModels)
	numberOfModels = linkList[i];
    }

  numberOfModels++;
  
  ll->entries = numberOfModels;
  ll->ld      = (linkageData*)malloc(sizeof(linkageData) * numberOfModels);


  for(i = 0; i < numberOfModels; i++)
    {
      ll->ld[i].valid = TRUE;
      partitions = 0;

      for(k = 0; k < tr->NumberOfModels; k++)	
	if(linkList[k] == i)
	  partitions++;	    

      ll->ld[i].partitions = partitions;
      ll->ld[i].partitionList = (int*)malloc(sizeof(int) * partitions);
      
      for(k = 0, pos = 0; k < tr->NumberOfModels; k++)	
	if(linkList[k] == i)
	  ll->ld[i].partitionList[pos++] = k;
    }

  return ll;
}


static linkageList* initLinkageListGTR(tree *tr)
{
  int
    i,
    *links = (int*)malloc(sizeof(int) * tr->NumberOfModels),
    firstAA = tr->NumberOfModels + 2,
    countGTR = 0,
    countOtherModel = 0;
  linkageList* ll;

  for(i = 0; i < tr->NumberOfModels; i++)
    {     
      if(tr->partitionData[i].dataType == AA_DATA)
	{
	  if(tr->partitionData[i].protModels == GTR)
	    {
	      if(i < firstAA)
		firstAA = i;
	      countGTR++;
	    }
	  else
	    countOtherModel++;
	}
    }
  
  assert((countGTR > 0 && countOtherModel == 0) || (countGTR == 0 && countOtherModel > 0) ||  (countGTR == 0 && countOtherModel == 0));

  if(countGTR == 0)
    {
      for(i = 0; i < tr->NumberOfModels; i++)
	links[i] = i;
    }
  else
    {
      for(i = 0; i < tr->NumberOfModels; i++)
	{
	  switch(tr->partitionData[i].dataType)
	    {	   
	    case DNA_DATA:
	    case BINARY_DATA:
	    case GENERIC_32:
	    case GENERIC_64:
	    case SECONDARY_DATA:
	    case SECONDARY_DATA_6:
	    case SECONDARY_DATA_7: 
	      links[i] = i;
	      break;
	    case AA_DATA:	  
	      links[i] = firstAA;
	      break;
	    default:
	      assert(0);
	    }
	}
    }
  

  ll = initLinkageList(links, tr);

  free(links);
  
  return ll;
}



static void freeLinkageList( linkageList* ll)
{
  int i;    

  for(i = 0; i < ll->entries; i++)    
    free(ll->ld[i].partitionList);         

  free(ll->ld);
  free(ll);   
}

#define ALPHA_F 0
#define INVAR_F 1
#define RATE_F  2




static void evaluateChange(tree *tr, int rateNumber, double *value, double *result, boolean* converged, analdef *adef, int whichFunction, int numberOfModels, linkageList *ll)
{ 
  int i, k, pos;

  switch(whichFunction)
    {
    case ALPHA_F:
      for(i = 0; i < ll->entries; i++)
	{
	  if(converged[i])
	    {
	      for(k = 0; k < ll->ld[i].partitions; k++)
		tr->executeModel[ll->ld[i].partitionList[k]] = FALSE;
	    }
	  else
	    {
	      for(k = 0; k < ll->ld[i].partitions; k++)
		{
		  int index = ll->ld[i].partitionList[k];
		  tr->executeModel[index] = TRUE;
		  tr->partitionData[index].alpha = value[i];
		  makeGammaCats(tr->partitionData[index].alpha, tr->partitionData[index].gammaRates, 4);
		}
	    }
	}
#ifdef _USE_PTHREADS   
      {
	volatile double result;
	
	masterBarrier(THREAD_OPT_ALPHA, tr);
	if(tr->NumberOfModels == 1)
	  {
	    for(i = 0, result = 0.0; i < NumberOfThreads; i++)    	  
	      result += reductionBuffer[i];  	        
	    tr->perPartitionLH[0] = result;
	  }
	else
	  {
	    int j;
	    volatile double partitionResult;
	
	    result = 0.0;

	    for(j = 0; j < tr->NumberOfModels; j++)
	      {
		for(i = 0, partitionResult = 0.0; i < NumberOfThreads; i++)          	      
		  partitionResult += reductionBuffer[i * tr->NumberOfModels + j];
		result +=  partitionResult;
		tr->perPartitionLH[j] = partitionResult;
	      }
	  }
      }
#else
      evaluateGenericInitrav(tr, tr->start);
#endif
            
      for(i = 0; i < ll->entries; i++)	
	{	  
	  result[i] = 0.0;
	  
	  for(k = 0; k < ll->ld[i].partitions; k++)
	    {
	      int index = ll->ld[i].partitionList[k];

	      result[i] -= tr->perPartitionLH[index];	            
	      tr->executeModel[index] = TRUE;
	    }
	}
      break;
    case INVAR_F:

       for(i = 0; i < ll->entries; i++)
	{
	  if(converged[i])
	    {
	      for(k = 0; k < ll->ld[i].partitions; k++)
		tr->executeModel[ll->ld[i].partitionList[k]] = FALSE;
	    }
	  else
	    {
	      for(k = 0; k < ll->ld[i].partitions; k++)
		{
		  int index = ll->ld[i].partitionList[k];
		  tr->executeModel[index] = TRUE;
		  tr->partitionData[index].propInvariant = value[i];		 
		}
	    }
	}      

#ifdef _USE_PTHREADS
      {
	volatile double result;
		
	masterBarrier(THREAD_OPT_INVAR, tr);
	if(tr->NumberOfModels == 1)
	  {
	    for(i = 0, result = 0.0; i < NumberOfThreads; i++)    	  
	      result += reductionBuffer[i];  	        
	    tr->perPartitionLH[0] = result;
	  }
	else
	  {
	    int j;
	    volatile double partitionResult;
	
	    result = 0.0;

	    for(j = 0; j < tr->NumberOfModels; j++)
	      {
		for(i = 0, partitionResult = 0.0; i < NumberOfThreads; i++)          	      
		  partitionResult += reductionBuffer[i * tr->NumberOfModels + j];
		result +=  partitionResult;
		tr->perPartitionLH[j] = partitionResult;
	      }
	  }	
      }
#else
      evaluateGeneric(tr, tr->start);
#endif



    

      for(i = 0; i < ll->entries; i++)	
	{	  
	  result[i] = 0.0;
	  
	  for(k = 0; k < ll->ld[i].partitions; k++)
	    {
	      int index = ll->ld[i].partitionList[k];

	      result[i] -= tr->perPartitionLH[index];	            
	      tr->executeModel[index] = TRUE;
	    }
	}
         
      break;
    case RATE_F:
      for(i = 0, pos = 0; i < ll->entries; i++)
	{
	  if(ll->ld[i].valid)
	    {
	      if(converged[pos])
		{
		  for(k = 0; k < ll->ld[i].partitions; k++)
		    tr->executeModel[ll->ld[i].partitionList[k]] = FALSE;
		}
	      else
		{
		  for(k = 0; k < ll->ld[i].partitions; k++)
		    {
		      int index = ll->ld[i].partitionList[k];		  	      
		      setRateModel(tr, index, value[pos], rateNumber);  
		      initReversibleGTR(tr, adef, index);		 
		    }
		}
	      pos++;
	    }
	  else
	    {
	      for(k = 0; k < ll->ld[i].partitions; k++)
		tr->executeModel[ll->ld[i].partitionList[k]] = FALSE;	     
	    }
	 
	}

      assert(pos == numberOfModels);

#ifdef _USE_PTHREADS
      {
	volatile double result;
	
	masterBarrier(THREAD_OPT_RATE, tr);
	if(tr->NumberOfModels == 1)
	  {
	    for(i = 0, result = 0.0; i < NumberOfThreads; i++)    	  
	      result += reductionBuffer[i];  	        
	    tr->perPartitionLH[0] = result;
	  }
	else
	  {
	    int j;
	    volatile double partitionResult;
	
	    result = 0.0;

	    for(j = 0; j < tr->NumberOfModels; j++)
	      {
		for(i = 0, partitionResult = 0.0; i < NumberOfThreads; i++)          	      
		  partitionResult += reductionBuffer[i * tr->NumberOfModels + j];
		result +=  partitionResult;
		tr->perPartitionLH[j] = partitionResult;
	      }
	  }
      }
#else
      evaluateGenericInitrav(tr, tr->start);      
#endif
     
      
      for(i = 0, pos = 0; i < ll->entries; i++)	
	{
	  if(ll->ld[i].valid)
	    {
	      result[pos] = 0.0;
	      for(k = 0; k < ll->ld[i].partitions; k++)
		{
		  int index = ll->ld[i].partitionList[k];

		  result[pos] -= tr->perPartitionLH[index];	            		  
		}
	      pos++;
	    }
	   for(k = 0; k < ll->ld[i].partitions; k++)
	     {
	       int index = ll->ld[i].partitionList[k];
	       tr->executeModel[index] = TRUE;
	     }	  
	}

      assert(pos == numberOfModels);
      break;
    default:
      assert(0);	
    }

}



static void brentGeneric(double *ax, double *bx, double *cx, double *fb, double tol, double *xmin, double *result, int numberOfModels, 
			 int whichFunction, int rateNumber, analdef *adef, tree *tr, linkageList *ll)
{
  int iter, i;
  double 
    *a     = (double *)malloc(sizeof(double) * numberOfModels),
    *b     = (double *)malloc(sizeof(double) * numberOfModels),
    *d     = (double *)malloc(sizeof(double) * numberOfModels),
    *etemp = (double *)malloc(sizeof(double) * numberOfModels),
    *fu    = (double *)malloc(sizeof(double) * numberOfModels),
    *fv    = (double *)malloc(sizeof(double) * numberOfModels),
    *fw    = (double *)malloc(sizeof(double) * numberOfModels),
    *fx    = (double *)malloc(sizeof(double) * numberOfModels),
    *p     = (double *)malloc(sizeof(double) * numberOfModels),
    *q     = (double *)malloc(sizeof(double) * numberOfModels),
    *r     = (double *)malloc(sizeof(double) * numberOfModels),
    *tol1  = (double *)malloc(sizeof(double) * numberOfModels),
    *tol2  = (double *)malloc(sizeof(double) * numberOfModels),
    *u     = (double *)malloc(sizeof(double) * numberOfModels),
    *v     = (double *)malloc(sizeof(double) * numberOfModels),
    *w     = (double *)malloc(sizeof(double) * numberOfModels),
    *x     = (double *)malloc(sizeof(double) * numberOfModels),
    *xm    = (double *)malloc(sizeof(double) * numberOfModels),
    *e     = (double *)malloc(sizeof(double) * numberOfModels);
  boolean *converged = (boolean *)malloc(sizeof(boolean) * numberOfModels);
  boolean allConverged;
  
  for(i = 0; i < numberOfModels; i++)    
    converged[i] = FALSE;

  for(i = 0; i < numberOfModels; i++)
    {
      e[i] = 0.0;
      d[i] = 0.0;
    }

  for(i = 0; i < numberOfModels; i++)
    {
      a[i]=((ax[i] < cx[i]) ? ax[i] : cx[i]);
      b[i]=((ax[i] > cx[i]) ? ax[i] : cx[i]);
      x[i] = w[i] = v[i] = bx[i];
      fw[i] = fv[i] = fx[i] = fb[i];
    }

  for(iter = 1; iter <= ITMAX; iter++)
    {
      allConverged = TRUE;

      for(i = 0; i < numberOfModels && allConverged; i++)
	allConverged = allConverged && converged[i];

      if(allConverged)
	{
	  free(converged);
	  free(a);
	  free(b);
	  free(d);
	  free(etemp);
	  free(fu);
	  free(fv);
	  free(fw);
	  free(fx);
	  free(p);
	  free(q);
	  free(r);
	  free(tol1);
	  free(tol2);
	  free(u);
	  free(v);
	  free(w);
	  free(x);
	  free(xm);
	  free(e);
	  return;
	}     



      for(i = 0; i < numberOfModels; i++)
	{
	  if(!converged[i])
	    {
	      xm[i] = 0.5 * (a[i] + b[i]);
	      tol2[i] = 2.0 * (tol1[i] = tol * fabs(x[i]) + BRENT_ZEPS);
	  
	      if(fabs(x[i] - xm[i]) <= (tol2[i] - 0.5 * (b[i] - a[i])))
		{
		  result[i] =  -fx[i];
		  xmin[i]   = x[i];
		  converged[i] = TRUE;		  
		}
	      else
		{
		  if(fabs(e[i]) > tol1[i])
		    {
		      r[i] = (x[i] - w[i]) * (fx[i] - fv[i]);
		      q[i] = (x[i] - v[i]) * (fx[i] - fw[i]);
		      p[i] = (x[i] - v[i]) * q[i] - (x[i] - w[i]) * r[i];
		      q[i] = 2.0 * (q[i] - r[i]);
		      if(q[i] > 0.0)
			p[i] = -p[i];
		      q[i] = fabs(q[i]);
		      etemp[i] = e[i];
		      e[i] = d[i];
		      if((fabs(p[i]) >= fabs(0.5 * q[i] * etemp[i])) || (p[i] <= q[i] * (a[i]-x[i])) || (p[i] >= q[i] * (b[i] - x[i])))
			d[i] = BRENT_CGOLD * (e[i] = (x[i] >= xm[i] ? a[i] - x[i] : b[i] - x[i]));
		      else
			{
			  d[i] = p[i] / q[i];
			  u[i] = x[i] + d[i];
			  if( u[i] - a[i] < tol2[i] || b[i] - u[i] < tol2[i])
			    d[i] = SIGN(tol1[i], xm[i] - x[i]);
			}
		    }
		  else
		    {
		      d[i] = BRENT_CGOLD * (e[i] = (x[i] >= xm[i] ? a[i] - x[i]: b[i] - x[i]));
		    }
		  u[i] = (fabs(d[i]) >= tol1[i] ? x[i] + d[i]: x[i] +SIGN(tol1[i], d[i]));
		}
	    }
	}

      
      evaluateChange(tr, rateNumber, u, fu, converged, adef, whichFunction, numberOfModels, ll);

      for(i = 0; i < numberOfModels; i++)
	{
	  if(!converged[i])
	    {
	      if(fu[i] <= fx[i])
		{
		  if(u[i] >= x[i])
		    a[i] = x[i];
		  else
		    b[i] = x[i];
		  SHFT(v[i],w[i],x[i],u[i]);
		  SHFT(fv[i],fw[i],fx[i],fu[i]);
		}
	      else
		{
		  if(u[i] < x[i])
		    a[i] = u[i];
		  else
		    b[i] = u[i];
		  if(fu[i] <= fw[i] || w[i] == x[i])
		    {
		      v[i] = w[i];
		      w[i] = u[i];
		      fv[i] = fw[i];
		      fw[i] = fu[i];
		    }
		  else
		    {
		      if(fu[i] <= fv[i] || v[i] == x[i] || v[i] == w[i])
			{
			  v[i] = u[i];
			  fv[i] = fu[i];
			}
		    }	    
		}
	    }
	}
    }

  free(converged);
  free(a);
  free(b);
  free(d);
  free(etemp);
  free(fu);
  free(fv);
  free(fw);
  free(fx);
  free(p);
  free(q);
  free(r);
  free(tol1);
  free(tol2);
  free(u);
  free(v);
  free(w);
  free(x);
  free(xm);
  free(e);

  printf("\n. Too many iterations in BRENT !");
  assert(0);
}



static int brakGeneric(double *param, double *ax, double *bx, double *cx, double *fa, double *fb, 
		       double *fc, double lim_inf, double lim_sup, 
		       int numberOfModels, int rateNumber, analdef *adef, int whichFunction, tree *tr, linkageList *ll)
{
  double 
    *ulim = (double *)malloc(sizeof(double) * numberOfModels),
    *u    = (double *)malloc(sizeof(double) * numberOfModels),
    *r    = (double *)malloc(sizeof(double) * numberOfModels),
    *q    = (double *)malloc(sizeof(double) * numberOfModels),
    *fu   = (double *)malloc(sizeof(double) * numberOfModels),
    *dum  = (double *)malloc(sizeof(double) * numberOfModels), 
    *temp = (double *)malloc(sizeof(double) * numberOfModels);
  
  int 
    i,
    *state    = (int *)malloc(sizeof(int) * numberOfModels),
    *endState = (int *)malloc(sizeof(int) * numberOfModels);

  boolean *converged = (boolean *)malloc(sizeof(boolean) * numberOfModels);
  boolean allConverged;

  for(i = 0; i < numberOfModels; i++)
    converged[i] = FALSE;


  for(i = 0; i < numberOfModels; i++)
    {
      state[i] = 0;
      endState[i] = 0;

      u[i] = 0.0;

      param[i] = ax[i];

      if(param[i] > lim_sup) param[i] = lim_sup;
      if(param[i] < lim_inf) param[i] = lim_inf;
    }
   
  
  evaluateChange(tr, rateNumber, param, fa, converged, adef, whichFunction, numberOfModels, ll);


  for(i = 0; i < numberOfModels; i++)
    {
      param[i] = bx[i];
      if(param[i] > lim_sup) param[i] = lim_sup;
      if(param[i] < lim_inf) param[i] = lim_inf;
    }

  

  evaluateChange(tr, rateNumber, param, fb, converged, adef, whichFunction, numberOfModels, ll);

  for(i = 0; i < numberOfModels; i++)  
    {
      if (fb[i] > fa[i]) 
	{
	  SHFT(dum[i],ax[i],bx[i],dum[i]);
	  SHFT(dum[i],fa[i],fb[i],dum[i]);
	}
      
      cx[i]=(bx[i])+MNBRAK_GOLD*(bx[i]-ax[i]);
      param[i] = cx[i];
      if(param[i] > lim_sup) param[i] = cx[i] = lim_sup;
      if(param[i] < lim_inf) param[i] = cx[i] = lim_inf;
    }
  
 
  evaluateChange(tr, rateNumber, param, fc, converged, adef, whichFunction, numberOfModels,  ll);

   while(1) 
     {       
       allConverged = TRUE;

       for(i = 0; i < numberOfModels && allConverged; i++)
	 allConverged = allConverged && converged[i];

       if(allConverged)
	 {
	   free(converged);
	   free(ulim);
	   free(u);
	   free(r);
	   free(q);
	   free(fu);
	   free(dum); 
	   free(temp);
	   free(state);   
	   free(endState);
	   return 0;
	   
	 }


 
       for(i = 0; i < numberOfModels; i++)
	 {
	   if(!converged[i])
	     {
	       switch(state[i])
		 {
		 case 0:
		   endState[i] = 0;
		   if(!(fb[i] > fc[i]))
		     {	     
		       converged[i] = TRUE;		       
		     }
		   else
		     {
		   
		       if(ax[i] > lim_sup) ax[i] = lim_sup;
		       if(ax[i] < lim_inf) ax[i] = lim_inf;
		       if(bx[i] > lim_sup) bx[i] = lim_sup;
		       if(bx[i] < lim_inf) bx[i] = lim_inf;
		       if(cx[i] > lim_sup) cx[i] = lim_sup;
		       if(cx[i] < lim_inf) cx[i] = lim_inf;
		       
		       r[i]=(bx[i]-ax[i])*(fb[i]-fc[i]);
		       q[i]=(bx[i]-cx[i])*(fb[i]-fa[i]);
		       u[i]=(bx[i])-((bx[i]-cx[i])*q[i]-(bx[i]-ax[i])*r[i])/
			 (2.0*SIGN(MAX(fabs(q[i]-r[i]),MNBRAK_TINY),q[i]-r[i]));
		       ulim[i]=(bx[i])+MNBRAK_GLIMIT*(cx[i]-bx[i]);
		       
		       if(u[i] > lim_sup) u[i] = lim_sup;
		       if(u[i] < lim_inf) u[i] = lim_inf;
		       if(ulim[i] > lim_sup) ulim[i] = lim_sup;
		       if(ulim[i] < lim_inf) ulim[i] = lim_inf;
		       
		       if ((bx[i]-u[i])*(u[i]-cx[i]) > 0.0)
			 {
			   param[i] = u[i];	  
			   endState[i] = 1;
			 }
		       else 
			 {
			   if ((cx[i]-u[i])*(u[i]-ulim[i]) > 0.0) 
			     {
			       param[i] = u[i];	       
			       endState[i] = 2;
			     }		  	       
			   else
			     {
			       if ((u[i]-ulim[i])*(ulim[i]-cx[i]) >= 0.0) 
				 {
				   u[i] = ulim[i];
				   param[i] = u[i];	
				   endState[i] = 0;
				 }		  		
			       else 
				 {		  
				   u[i]=(cx[i])+MNBRAK_GOLD*(cx[i]-bx[i]);
				   param[i] = u[i];
				   endState[i] = 0;
				   if(param[i] > lim_sup) {param[i] = u[i] = lim_sup;}
				   if(param[i] < lim_inf) {param[i] = u[i] = lim_inf;}			  
				 }
			     }	  
			 }
		     }
		   break;
		 case 1:
		   endState[i] = 0;
		   break;
		 case 2:
		   endState[i] = 3;
		   break;
		 default:
		   assert(0);
		 }
	     }
	 }
       
       

       evaluateChange(tr, rateNumber, param, temp, converged, adef, whichFunction, numberOfModels, ll);

       for(i = 0; i < numberOfModels; i++)
	 {
	   if(!converged[i])
	     {
	       switch(endState[i])
		 {
		 case 0:
		   fu[i] = temp[i];
		   SHFT(ax[i],bx[i],cx[i],u[i]);
		   SHFT(fa[i],fb[i],fc[i],fu[i]);
		   state[i] = 0;
		   break;
		 case 1:
		   fu[i] = temp[i];
		   if (fu[i] < fc[i]) 
		     {
		       ax[i]=(bx[i]);
		       bx[i]=u[i];
		       fa[i]=(fb[i]);
		       fb[i]=fu[i]; 
		       converged[i] = TRUE;		      
		     } 
		   else 
		     {
		       if (fu[i] > fb[i]) 
			 {
			   cx[i]=u[i];
			   fc[i]=fu[i];
			   converged[i] = TRUE;			  
			 }
		       else
			 {		   
			   u[i]=(cx[i])+MNBRAK_GOLD*(cx[i]-bx[i]);
			   param[i] = u[i];
			   if(param[i] > lim_sup) {param[i] = u[i] = lim_sup;}
			   if(param[i] < lim_inf) {param[i] = u[i] = lim_inf;}	  
			   state[i] = 1;		 
			 }		  
		     }
		   break;
		 case 2: 
		   fu[i] = temp[i];
		   if (fu[i] < fc[i]) 
		     {
		       SHFT(bx[i],cx[i],u[i],cx[i]+MNBRAK_GOLD*(cx[i]-bx[i]));
		       state[i] = 2;
		     }	   
		   else
		     {
		       state[i] = 0;
		       SHFT(ax[i],bx[i],cx[i],u[i]);
		       SHFT(fa[i],fb[i],fc[i],fu[i]);
		     }
		   break;	   
		 case 3:
		   SHFT(fb[i],fc[i],fu[i], temp[i]);
		   SHFT(ax[i],bx[i],cx[i],u[i]);
		   SHFT(fa[i],fb[i],fc[i],fu[i]);
		   state[i] = 0;
		   break;
		 default:
		   assert(0);
		 }
	     }
	 }
    }
   

   assert(0);
   free(converged);
   free(ulim);
   free(u);
   free(r);
   free(q);
   free(fu);
   free(dum); 
   free(temp);
   free(state);   
   free(endState);

   return(0);
}









static void optInvar(tree *tr, double modelEpsilon, linkageList *ll)
{
  int 
    i,
    k,
    numberOfModels = ll->entries;
  double lim_inf = INVAR_MIN;
  double lim_sup = INVAR_MAX;
   double
    *startLH    = (double *)malloc(sizeof(double) * numberOfModels),
    *startInvar = (double *)malloc(sizeof(double) * numberOfModels),
    *endInvar   = (double *)malloc(sizeof(double) * numberOfModels),
    *_a     = (double *)malloc(sizeof(double) * numberOfModels),
    *_b     = (double *)malloc(sizeof(double) * numberOfModels),
    *_c     = (double *)malloc(sizeof(double) * numberOfModels),
    *_fa    = (double *)malloc(sizeof(double) * numberOfModels),
    *_fb    = (double *)malloc(sizeof(double) * numberOfModels),
    *_fc    = (double *)malloc(sizeof(double) * numberOfModels),
    *_param = (double *)malloc(sizeof(double) * numberOfModels),
    *result = (double *)malloc(sizeof(double) * numberOfModels),
    *_x     = (double *)malloc(sizeof(double) * numberOfModels);
#ifdef _USE_PTHREADS
   int revertModel = 0;
#endif

  evaluateGenericInitrav(tr, tr->start); 
#ifdef _USE_PTHREADS
  evaluateGeneric(tr, tr->start); 
  /* to avoid transferring traversal info further on */
#endif

  for(i = 0; i < numberOfModels; i++)
    {
      assert(ll->ld[i].valid);

      startInvar[i] = tr->partitionData[ll->ld[i].partitionList[0]].propInvariant;
      _a[i] = startInvar[i] + 0.1;
      _b[i] = startInvar[i] - 0.1;      
      if(_b[i] < lim_inf) 
	_b[i] = lim_inf;

      startLH[i] = 0.0;
      
      for(k = 0; k < ll->ld[i].partitions; k++)	
	{
	  startLH[i] += tr->perPartitionLH[ll->ld[i].partitionList[k]];
	  /* TODO need to fix the initialization for this assertion not to fail */
	  /* assert(tr->partitionData[ll->ld[i].partitionList[0]].propInvariant ==  tr->partitionData[ll->ld[i].partitionList[k]].propInvariant);*/
	}
    }	       

  brakGeneric(_param, _a, _b, _c, _fa, _fb, _fc, lim_inf, lim_sup, numberOfModels, -1, (analdef*)NULL, INVAR_F, tr, ll);
  brentGeneric(_a, _b, _c, _fb, modelEpsilon, _x, result, numberOfModels, INVAR_F, -1, (analdef*)NULL, tr, ll);

  for(i = 0; i < numberOfModels; i++)
    endInvar[i] = result[i];

  for(i = 0; i < numberOfModels; i++)
    {
      if(startLH[i] > endInvar[i])
	{    	  
	  for(k = 0; k < ll->ld[i].partitions; k++)	    
	    tr->partitionData[ll->ld[i].partitionList[k]].propInvariant = startInvar[i];	     			    
#ifdef _USE_PTHREADS	  
	  revertModel++;
#endif
	}      
    }

#ifdef _USE_PTHREADS
  if(revertModel > 0)
    masterBarrier(THREAD_COPY_INVAR, tr);	 
#endif
 

  free(startLH);
  free(startInvar);
  free(endInvar);
  free(result);
  free(_a);
  free(_b);
  free(_c);
  free(_fa);
  free(_fb);
  free(_fc);
  free(_param);
  free(_x);  
 
}






/**********************************************************************************************************/
/* ALPHA PARAM ********************************************************************************************/







static void optAlpha(tree *tr, double modelEpsilon, linkageList *ll)
{
  int 
    i, 
    k,
    numberOfModels = ll->entries;
  
  double 
    lim_inf     = ALPHA_MIN,
    lim_sup     = ALPHA_MAX;
  double
    *startLH    = (double *)malloc(sizeof(double) * numberOfModels),
    *startAlpha = (double *)malloc(sizeof(double) * numberOfModels),
    *endAlpha   = (double *)malloc(sizeof(double) * numberOfModels),
    *_a         = (double *)malloc(sizeof(double) * numberOfModels),
    *_b         = (double *)malloc(sizeof(double) * numberOfModels),
    *_c         = (double *)malloc(sizeof(double) * numberOfModels),
    *_fa        = (double *)malloc(sizeof(double) * numberOfModels),
    *_fb        = (double *)malloc(sizeof(double) * numberOfModels),
    *_fc        = (double *)malloc(sizeof(double) * numberOfModels),
    *_param     = (double *)malloc(sizeof(double) * numberOfModels),
    *result     = (double *)malloc(sizeof(double) * numberOfModels),
    *_x         = (double *)malloc(sizeof(double) * numberOfModels);   

#ifdef _USE_PTHREADS
   int revertModel = 0;
#endif   

  evaluateGenericInitrav(tr, tr->start);
   /* 
     at this point here every worker has the traversal data it needs for the 
     search, so we won't re-distribute it he he :-)
  */

  for(i = 0; i < numberOfModels; i++)
    {
      assert(ll->ld[i].valid);

      startAlpha[i] = tr->partitionData[ll->ld[i].partitionList[0]].alpha;
      _a[i] = startAlpha[i] + 0.1;
      _b[i] = startAlpha[i] - 0.1;      
      if(_b[i] < lim_inf) 
	_b[i] = lim_inf;

      startLH[i] = 0.0;
      
      for(k = 0; k < ll->ld[i].partitions; k++)	
	{
	  startLH[i] += tr->perPartitionLH[ll->ld[i].partitionList[k]];
	  assert(tr->partitionData[ll->ld[i].partitionList[0]].alpha ==  tr->partitionData[ll->ld[i].partitionList[k]].alpha);
	}
    }					  

  brakGeneric(_param, _a, _b, _c, _fa, _fb, _fc, lim_inf, lim_sup, numberOfModels, -1, (analdef*)NULL, ALPHA_F, tr, ll);       
  brentGeneric(_a, _b, _c, _fb, modelEpsilon, _x, result, numberOfModels, ALPHA_F, -1, (analdef*)NULL, tr, ll);

  for(i = 0; i < numberOfModels; i++)
    endAlpha[i] = result[i];
  
  for(i = 0; i < numberOfModels; i++)
    {
      if(startLH[i] > endAlpha[i])
	{    	  
	  for(k = 0; k < ll->ld[i].partitions; k++)
	    {	      
	      tr->partitionData[ll->ld[i].partitionList[k]].alpha = startAlpha[i];
	      makeGammaCats(tr->partitionData[ll->ld[i].partitionList[k]].alpha, tr->partitionData[ll->ld[i].partitionList[k]].gammaRates, 4); 		
	    }
#ifdef _USE_PTHREADS
	  revertModel++;
#endif
	}  
    }

#ifdef _USE_PTHREADS
  if(revertModel > 0)
    masterBarrier(THREAD_COPY_ALPHA, tr);
#endif

  
  free(startLH);
  free(startAlpha);
  free(endAlpha);
  free(result);
  free(_a);
  free(_b);
  free(_c);
  free(_fa);
  free(_fb);
  free(_fc);
  free(_param);
  free(_x);  

}


/*******************************************************************************************************************/
/*******************RATES ******************************************************************************************/



static void optRates(tree *tr, analdef *adef, double modelEpsilon, linkageList *ll, int numberOfModels, int states)
{
  int 
    i, 
    k, 
    j, 
    pos,
    numberOfRates = ((states * states - states) / 2) - 1;
    
  double lim_inf = RATE_MIN;
  double lim_sup = RATE_MAX;
  double 
    *startRates;
  double 
    *startLH= (double *)malloc(sizeof(double) * numberOfModels),
    *endLH  = (double *)malloc(sizeof(double) * numberOfModels),
    *_a     = (double *)malloc(sizeof(double) * numberOfModels),
    *_b     = (double *)malloc(sizeof(double) * numberOfModels),
    *_c     = (double *)malloc(sizeof(double) * numberOfModels),
    *_fa    = (double *)malloc(sizeof(double) * numberOfModels),
    *_fb    = (double *)malloc(sizeof(double) * numberOfModels),
    *_fc    = (double *)malloc(sizeof(double) * numberOfModels),
    *_param = (double *)malloc(sizeof(double) * numberOfModels),
    *result = (double *)malloc(sizeof(double) * numberOfModels),
    *_x     = (double *)malloc(sizeof(double) * numberOfModels); 
#ifdef _USE_PTHREADS
   int revertModel = 0;
#endif

   assert(states != -1);

  startRates = (double *)malloc(sizeof(double) * numberOfRates * numberOfModels);

  evaluateGenericInitrav(tr, tr->start);
  /* 
     at this point here every worker has the traversal data it needs for the 
     search 
  */

  for(i = 0, pos = 0; i < ll->entries; i++)
    {
      if(ll->ld[i].valid)
	{
	  endLH[pos] = unlikely;
	  startLH[pos] = 0.0;

	  for(j = 0; j < ll->ld[i].partitions; j++)
	    {
	      int index = ll->ld[i].partitionList[j];
	      
	      startLH[pos] += tr->perPartitionLH[index];
	      for(k = 0; k < numberOfRates; k++)
		startRates[pos * numberOfRates + k] = tr->partitionData[index].substRates[k];      
	    }
	  pos++;
	}
    }  

  assert(pos == numberOfModels);
  
  for(i = 0; i < numberOfRates; i++)
    {     
      for(k = 0, pos = 0; k < ll->entries; k++)
	{
	  if(ll->ld[k].valid)
	    {
	      int index = ll->ld[k].partitionList[0];
	      _a[pos] = tr->partitionData[index].substRates[i] + 0.1;
	      _b[pos] = tr->partitionData[index].substRates[i] - 0.1;
	      
	      if(_a[pos] < lim_inf) _a[pos] = lim_inf;
	      if(_a[pos] > lim_sup) _a[pos] = lim_sup;
	      
	      if(_b[pos] < lim_inf) _b[pos] = lim_inf;
	      if(_b[pos] > lim_sup) _b[pos] = lim_sup;    
	      pos++;
	    }
	}                       	     

      assert(pos == numberOfModels);

      brakGeneric(_param, _a, _b, _c, _fa, _fb, _fc, lim_inf, lim_sup, numberOfModels, i, adef, RATE_F, tr, ll);
      brentGeneric(_a, _b, _c, _fb, modelEpsilon, _x, result, numberOfModels, RATE_F, i, adef, tr,  ll);
	
      for(k = 0; k < numberOfModels; k++)
	endLH[k] = result[k];
	      
      for(k = 0, pos = 0; k < ll->entries; k++)
	{
#ifdef _USE_PTHREADS
	  revertModel = 0;
#endif
	  if(ll->ld[k].valid)
	    { 
	      if(startLH[pos] > endLH[pos])
		{
		  for(j = 0; j < ll->ld[k].partitions; j++)
		    {
		      int index = ll->ld[k].partitionList[j];
		      tr->partitionData[index].substRates[i] = startRates[pos * numberOfRates + i];	             	  
		      initReversibleGTR(tr, adef, index);
		    }
#ifdef _USE_PTHREADS		  
		  revertModel++;
#endif
		}
	      pos++;
	    }
	}

#ifdef _USE_PTHREADS
      if(revertModel > 0)
	masterBarrier(THREAD_COPY_RATES, tr);
#endif    
      assert(pos == numberOfModels);
    }

 
  free(startLH);
  free(endLH);
  free(result);
  free(_a);
  free(_b);
  free(_c);
  free(_fa);
  free(_fb);
  free(_fc);
  free(_param);
  free(_x);  
  free(startRates);
}

static boolean AAisGTR(tree *tr)
{
  int i, count = 0;

  for(i = 0; i < tr->NumberOfModels; i++)   
    {
      if(tr->partitionData[i].dataType == AA_DATA)
	{
	  count++;
	  if(tr->partitionData[i].protModels != GTR)
	    return FALSE;
	}
    }

  if(count == 0)
    return FALSE;

  return TRUE;
}

static void optRatesGeneric(tree *tr, analdef *adef, double modelEpsilon, linkageList *ll)
{
  int 
    i,
    dnaPartitions = 0,
    aaPartitions  = 0,
    secondaryPartitions = 0,
    secondaryModel = -1,
    states = -1;

  /* assumes homogeneous super-partitions, that either contain DNA or AA partitions !*/
  /* does not check whether AA are all linked */

  /* first do DNA */

  for(i = 0; i < ll->entries; i++)
    {
      switch(tr->partitionData[ll->ld[i].partitionList[0]].dataType)
	{
	case DNA_DATA:	
	  states = tr->partitionData[ll->ld[i].partitionList[0]].states;
	  ll->ld[i].valid = TRUE;
	  dnaPartitions++;  
	  break;
	case BINARY_DATA:
	case AA_DATA:
	case SECONDARY_DATA:
	case SECONDARY_DATA_6:
	case SECONDARY_DATA_7:
	case GENERIC_32:
	case GENERIC_64:
	  ll->ld[i].valid = FALSE;
	  break;
	default:
	  assert(0);
	}      
    }   

  if(dnaPartitions > 0)
    optRates(tr, adef, modelEpsilon, ll, dnaPartitions, states);
  

  /* then SECONDARY */

   for(i = 0; i < ll->entries; i++)
    {
      switch(tr->partitionData[ll->ld[i].partitionList[0]].dataType)
	{
	  /* we only have one type of secondary data models in one analysis */
	case SECONDARY_DATA_6:
	  states = tr->partitionData[ll->ld[i].partitionList[0]].states;
	  secondaryModel = SECONDARY_DATA_6;
	  ll->ld[i].valid = TRUE;
	  secondaryPartitions++;  
	  break;
	case SECONDARY_DATA_7: 
	  states = tr->partitionData[ll->ld[i].partitionList[0]].states;
	  secondaryModel = SECONDARY_DATA_7;
	  ll->ld[i].valid = TRUE;
	  secondaryPartitions++;  
	  break;
	case SECONDARY_DATA:
	  states = tr->partitionData[ll->ld[i].partitionList[0]].states;
	  secondaryModel = SECONDARY_DATA;
	  ll->ld[i].valid = TRUE;
	  secondaryPartitions++;  
	  break;
	case BINARY_DATA:
	case AA_DATA:	
	case DNA_DATA:
	case GENERIC_32:
	case GENERIC_64:
	  ll->ld[i].valid = FALSE;
	  break;
	default:
	  assert(0);
	}      
    }

  
   
   if(secondaryPartitions > 0)
     {
       assert(secondaryPartitions == 1);

       switch(secondaryModel)
	 {
	 case SECONDARY_DATA:
	   optRates(tr, adef, modelEpsilon, ll, secondaryPartitions, states);
	   break;
	 case SECONDARY_DATA_6:
	   optRates(tr, adef, modelEpsilon, ll, secondaryPartitions, states);
	   break;
	 case SECONDARY_DATA_7:
	   optRates(tr, adef, modelEpsilon, ll, secondaryPartitions, states);
	   break; 
	 default:
	   assert(0);
	 }
     }

  /* then AA */

  if(AAisGTR(tr))
    {
      for(i = 0; i < ll->entries; i++)
	{
	  switch(tr->partitionData[ll->ld[i].partitionList[0]].dataType)
	    {
	    case AA_DATA:
	      states = tr->partitionData[ll->ld[i].partitionList[0]].states;
	      ll->ld[i].valid = TRUE;
	      aaPartitions++;
	      break;
	    case DNA_DATA:	    
	    case BINARY_DATA:
	    case SECONDARY_DATA:	
	    case SECONDARY_DATA_6:
	    case SECONDARY_DATA_7:
	      ll->ld[i].valid = FALSE;
	      break;
	    default:
	      assert(0);
	    }	 
	}

      assert(aaPartitions == 1);     
      
      optRates(tr, adef, modelEpsilon, ll, aaPartitions, states);
    }
  
  /* then multi-state */

  /* 
     now here we have to be careful, because every multi-state partition can actually 
     have a distinct number of states, so we will go to every multi-state partition separately,
     parallel efficiency for this will suck, but what the hell .....
  */

  if(tr->multiStateModel == GTR_MULTI_STATE)
    {     
      for(i = 0; i < ll->entries; i++)
	{
	  switch(tr->partitionData[ll->ld[i].partitionList[0]].dataType)
	    {
	    case GENERIC_32:
	      {
		int k;
		
		states = tr->partitionData[ll->ld[i].partitionList[0]].states;			      

		ll->ld[i].valid = TRUE;
		
		for(k = 0; k < ll->entries; k++)
		  if(k != i)
		    ll->ld[k].valid = FALSE;
		
		optRates(tr, adef, modelEpsilon, ll, 1, states);
	      }
	      break;
	    case AA_DATA:	    
	    case DNA_DATA:	    
	    case BINARY_DATA:
	    case SECONDARY_DATA:	
	    case SECONDARY_DATA_6:
	    case SECONDARY_DATA_7:
	    case GENERIC_64:
	      break;
	    default:
	      assert(0);
	    }	 
	}           
    }

  for(i = 0; i < ll->entries; i++)
    ll->ld[i].valid = TRUE;
}





/*********************FUNCTIONS FOR APPROXIMATE MODEL OPTIMIZATION ***************************************/




static int catCompare(const void *p1, const void *p2)
{
 rateCategorize *rc1 = (rateCategorize *)p1;
 rateCategorize *rc2 = (rateCategorize *)p2;

  double i = rc1->accumulatedSiteLikelihood;
  double j = rc2->accumulatedSiteLikelihood;
  
  if (i > j)
    return (1);
  if (i < j)
    return (-1);
  return (0);
}


static void categorize(tree *tr, rateCategorize *rc)
{
  int i, k, found;
  double temp, diff, min;

  for (i = 0; i < tr->cdta->endsite; i++) 
      {
	temp = tr->cdta->patrat[i];
	found = 0;
	for(k = 0; k < tr->NumberOfCategories; k++)
	  {
	    if(temp == rc[k].rate || (fabs(temp - rc[k].rate) < 0.001))
	      {
		found = 1;
		tr->cdta->rateCategory[i] = k;				
		break;
	      }
	  }
	if(!found)
	  {
	    min = fabs(temp - rc[0].rate);
	    tr->cdta->rateCategory[i] = 0;

	    for(k = 1; k < tr->NumberOfCategories; k++)
	    {
	      diff = fabs(temp - rc[k].rate);
	      if(diff < min)
		{
		  min = diff;
		  tr->cdta->rateCategory[i] = k;
		}
	    }
	  }
      }

  for(k = 0; k < tr->NumberOfCategories; k++)
    tr->cdta->patrat[k] = rc[k].rate; 

}

#ifdef _USE_PTHREADS

void optRateCatPthreads(tree *tr, double lower_spacing, double upper_spacing, double *lhs, int n, int tid)
{
  int 
    model, 
    localIndex, 
    i;

  for(model = 0; model < tr->NumberOfModels; model++)
    {               
      for(i = tr->partitionData[model].lower, localIndex = 0;  i < tr->partitionData[model].upper; i++)
	{
	  if(i % n == tid)
	    {
	      
	      double initialRate, initialLikelihood, 
		leftLH, rightLH, leftRate, rightRate, v;
	      const double epsilon = 0.00001;
	      int k;	      
	      
	      tr->cdta->patrat[i] = tr->cdta->patratStored[i];     
	      initialRate = tr->cdta->patrat[i];
	      
	      initialLikelihood = evaluatePartialGeneric(tr, localIndex, initialRate, model); /* i is real i ??? */
	      
	      
	      leftLH = rightLH = initialLikelihood;
	      leftRate = rightRate = initialRate;
	      
	      k = 1;
	      
	      while((initialRate - k * lower_spacing > 0.0001) && 
		    ((v = evaluatePartialGeneric(tr, localIndex, initialRate - k * lower_spacing, model)) 
		     > leftLH) && 
		    (fabs(leftLH - v) > epsilon))  
		{	  
#ifndef WIN32
		  if(isnan(v))
		    assert(0);
#endif
		  
		  leftLH = v;
		  leftRate = initialRate - k * lower_spacing;
		  k++;	  
		}      
	      
	      k = 1;
	      
	      while(((v = evaluatePartialGeneric(tr, localIndex, initialRate + k * upper_spacing, model)) > rightLH) &&
		    (fabs(rightLH - v) > epsilon))    	
		{
#ifndef WIN32
		  if(isnan(v))
		    assert(0);
#endif     
		  rightLH = v;
		  rightRate = initialRate + k * upper_spacing;	 
		  k++;
		}           
	      
	      if(rightLH > initialLikelihood || leftLH > initialLikelihood)
		{
		  if(rightLH > leftLH)	    
		    {	     
		      tr->cdta->patrat[i] = rightRate;
		      lhs[i] = rightLH;
		    }
		  else
		    {	      
		      tr->cdta->patrat[i] = leftRate;
		      lhs[i] = leftLH;
		    }
		}
	      else
		lhs[i] = initialLikelihood;
	      
	      tr->cdta->patratStored[i] = tr->cdta->patrat[i];
	      localIndex++;
	    }
	}
      assert(localIndex == tr->partitionData[model].width);    
    }
}



#else


static void optRateCatModel(tree *tr, int model, double lower_spacing, double upper_spacing, double *lhs)
{
  int lower = tr->partitionData[model].lower;
  int upper = tr->partitionData[model].upper;
  int i;
  for(i = lower; i < upper; i++)
    {
      double initialRate, initialLikelihood, 
	leftLH, rightLH, leftRate, rightRate, v;
      const double epsilon = 0.00001;
      int k;
      
      tr->cdta->patrat[i] = tr->cdta->patratStored[i];     
      initialRate = tr->cdta->patrat[i];
      
      initialLikelihood = evaluatePartialGeneric(tr, i, initialRate, model); 
      
      
      leftLH = rightLH = initialLikelihood;
      leftRate = rightRate = initialRate;
      
      k = 1;
      
      while((initialRate - k * lower_spacing > 0.0001) && 
	    ((v = evaluatePartialGeneric(tr, i, initialRate - k * lower_spacing, model)) 
	     > leftLH) && 
	    (fabs(leftLH - v) > epsilon))  
	{	  
#ifndef WIN32
	  if(isnan(v))
	    assert(0);
#endif
	  
	  leftLH = v;
	  leftRate = initialRate - k * lower_spacing;
	  k++;	  
	}      
      
      k = 1;
      
      while(((v = evaluatePartialGeneric(tr, i, initialRate + k * upper_spacing, model)) > rightLH) &&
	    (fabs(rightLH - v) > epsilon))    	
	{
#ifndef WIN32
	  if(isnan(v))
	    assert(0);
#endif     
	  rightLH = v;
	  rightRate = initialRate + k * upper_spacing;	 
	  k++;
	}           
  
      if(rightLH > initialLikelihood || leftLH > initialLikelihood)
	{
	  if(rightLH > leftLH)	    
	    {	     
	      tr->cdta->patrat[i] = rightRate;
	      lhs[i] = rightLH;
	    }
	  else
	    {	      
	      tr->cdta->patrat[i] = leftRate;
	      lhs[i] = leftLH;
	    }
	}
      else
	lhs[i] = initialLikelihood;
      
      tr->cdta->patratStored[i] = tr->cdta->patrat[i];
    }

}


#endif

static void optimizeRateCategories(tree *tr, int _maxCategories)
{
  assert(_maxCategories > 0);

  if(_maxCategories > 1)
    {
      int i, k;
      double  temp, wtemp;   
      double lower_spacing, upper_spacing;
      int maxCategories = _maxCategories;
      double initialLH = tr->likelihood;
      double *oldRat =    (double *)malloc(sizeof(double) * tr->cdta->endsite);
      double *ratStored = (double *)malloc(sizeof(double) * tr->cdta->endsite);
      double *oldwr =     (double *)malloc(sizeof(double) * tr->cdta->endsite);
      double *oldwr2 =    (double *)malloc(sizeof(double) * tr->cdta->endsite);
      double *lhs =       (double *)malloc(sizeof(double) * tr->cdta->endsite);
      int *oldCategory =  (int *)malloc(sizeof(int) * tr->cdta->endsite);  
      int oldNumber;   
  
      assert(isTip(tr->start->number, tr->rdta->numsp));   

      if(tr->multiGene)
	determineFullTraversalMulti(tr->start, tr);
       else
	 determineFullTraversal(tr->start, tr);

      if(optimizeRateCategoryInvocations == 1)
	{
	  lower_spacing = 0.5 / ((double)optimizeRateCategoryInvocations);
	  upper_spacing = 1.0 / ((double)optimizeRateCategoryInvocations);
	}
      else
	{
	  lower_spacing = 0.05 / ((double)optimizeRateCategoryInvocations);
	  upper_spacing = 0.1 / ((double)optimizeRateCategoryInvocations);
	}
      
      if(lower_spacing < 0.001)
	lower_spacing = 0.001;
      
      if(upper_spacing < 0.001)
	upper_spacing = 0.001;
      
      optimizeRateCategoryInvocations++;
      
      oldNumber = tr->NumberOfCategories;
      
      for(i = 0; i < tr->cdta->endsite; i++)
	{    
	  oldCategory[i] = tr->cdta->rateCategory[i];
	  ratStored[i] = tr->cdta->patratStored[i];    
	  oldRat[i] = tr->cdta->patrat[i];
	  oldwr[i] =  tr->cdta->wr[i];
	  oldwr2[i] =  tr->cdta->wr2[i];
	}
      
#ifdef _USE_PTHREADS
      tr->lhs = lhs;
      tr->lower_spacing = lower_spacing;
      tr->upper_spacing = upper_spacing;
      masterBarrier(THREAD_RATE_CATS, tr);
#else
      {
	int m;
	for(m = 0; m < tr->NumberOfModels; m++)      
	  optRateCatModel(tr, m, lower_spacing, upper_spacing, lhs);
      }
#endif

     
  
       
      {     
	rateCategorize *rc = (rateCategorize *)malloc(sizeof(rateCategorize) * tr->cdta->endsite);
	int where;
	int found = 0;
	for (i = 0; i < tr->cdta->endsite; i++)
	  {
	    rc[i].accumulatedSiteLikelihood = 0;
	    rc[i].rate = 0;
	  }
	
	where = 1;   
	rc[0].accumulatedSiteLikelihood = lhs[0];
	rc[0].rate = tr->cdta->patrat[0];
	tr->cdta->rateCategory[0] = 0;
	
	for (i = 1; i < tr->cdta->endsite; i++) 
	  {
	    temp = tr->cdta->patrat[i];
	    found = 0;
	    for(k = 0; k < where; k++)
	      {
		if(temp == rc[k].rate || (fabs(temp - rc[k].rate) < 0.001))
		  {
		    found = 1;						
		    rc[k].accumulatedSiteLikelihood += lhs[i];	
		    break;
		  }
	      }
	    if(!found)
	      {	    
		rc[where].rate = temp;	    
		rc[where].accumulatedSiteLikelihood += lhs[i];	    
		where++;
	      }
	  }
	
	qsort(rc, where, sizeof(rateCategorize), catCompare);
	
	if(where < maxCategories)
	  {
	    tr->NumberOfCategories = where;
	    categorize(tr, rc);
	  }
	else
	  {
	    tr->NumberOfCategories = maxCategories;	
	    categorize(tr, rc);
	  }
	
	free(rc);
	
	for (i = 0; i < tr->cdta->endsite; i++) 
	  {	
	    temp = tr->cdta->patrat[tr->cdta->rateCategory[i]];
	    
	    tr->cdta->wr[i]  = wtemp = temp * tr->cdta->aliaswgt[i];
	    tr->cdta->wr2[i] = temp * wtemp;

	    if(tr->useFloat)
	      {
		tr->cdta->wr_FLOAT[i]  = ((float)tr->cdta->wr[i]);
		tr->cdta->wr2_FLOAT[i] = ((float)tr->cdta->wr2[i]);
	      }

	  }        
	
#ifdef _USE_PTHREADS
	masterBarrier(THREAD_COPY_RATE_CATS, tr);
#endif

	evaluateGenericInitrav(tr, tr->start);
	
	if(tr->likelihood < initialLH)
	  {	
	    tr->NumberOfCategories = oldNumber;
	    for (i = 0; i < tr->cdta->endsite; i++)
	      {
		tr->cdta->patratStored[i] = ratStored[i]; 
		tr->cdta->rateCategory[i] = oldCategory[i];
		tr->cdta->patrat[i] = oldRat[i];	    
		tr->cdta->wr[i]  = oldwr[i];
		tr->cdta->wr2[i] = oldwr2[i];
		
		if(tr->useFloat)
		  {
		    tr->cdta->wr_FLOAT[i]  = ((float)tr->cdta->wr[i]);
		    tr->cdta->wr2_FLOAT[i] = ((float)tr->cdta->wr2[i]);
		  }
	      }       
	    
#ifdef _USE_PTHREADS
	    masterBarrier(THREAD_COPY_RATE_CATS, tr);
#endif
	    evaluateGenericInitrav(tr, tr->start);
	  }
      }
      
      /*
	for(i = 0; i < tr->NumberOfCategories; i++)
	printf("%f ", tr->cdta->patrat[i]);
	printf("\n");
      */
      
      
      free(oldCategory);
      free(oldRat);
      free(ratStored);
      free(oldwr);
      free(oldwr2); 
      free(lhs); 
    }
}
  






/*****************************************************************************************************/

void resetBranches(tree *tr)
{
  nodeptr  p, q;
  int  nodes, i;
  
  nodes = tr->mxtips  +  3 * (tr->mxtips - 2);
  p = tr->nodep[1];
  while (nodes-- > 0) 
    {   
      for(i = 0; i < tr->numBranches; i++)
	p->z[i] = defaultz;
	
      q = p->next;
      while(q != p)
	{	
	  for(i = 0; i < tr->numBranches; i++)
	    q->z[i] = defaultz;	    
	  q = q->next;
	}
      p++;
    }
}


static void printAAmatrix(tree *tr, double epsilon)
{
  if(AAisGTR(tr))
    {
      int model;
      
      for(model = 0; model < tr->NumberOfModels; model++)
	{
	  if(tr->partitionData[model].dataType == AA_DATA) 
	    {
	      char gtrFileName[1024];
	      char epsilonStr[1024];
	      FILE *gtrFile;
	      double *rates = tr->partitionData[model].substRates;
	      double *f     = tr->partitionData[model].frequencies;
	      double q[20][20];
	      int    r = 0;
	      int i, j;

	      assert(tr->partitionData[model].protModels == GTR);

	      sprintf(epsilonStr, "%f", epsilon);

	      strcpy(gtrFileName, workdir);
	      strcat(gtrFileName, "RAxML_proteinGTRmodel.");
	      strcat(gtrFileName, run_id);
	      strcat(gtrFileName, "_");
	      strcat(gtrFileName, epsilonStr);

	      gtrFile = fopen(gtrFileName, "w");

	      for(i = 0; i < 20; i++)
		for(j = 0; j < 20; j++)
		  q[i][j] = 0.0;

	      for(i = 0; i < 19; i++)
		for(j = i + 1; j < 20; j++)
		  q[i][j] = rates[r++];

	      for(i = 0; i < 20; i++)
		for(j = 0; j <= i; j++)
		  {
		    if(i == j)
		      q[i][j] = 0.0;
		    else
		      q[i][j] = q[j][i];
		  }
	   
	      for(i = 0; i < 20; i++)
		{
		  for(j = 0; j < 20; j++)		
		    fprintf(gtrFile, "%1.80f ", q[i][j]);
		
		  fprintf(gtrFile, "\n");
		}
	      for(i = 0; i < 20; i++)
		fprintf(gtrFile, "%1.80f ", f[i]);
	      fprintf(gtrFile, "\n");

	      fclose(gtrFile);

	      printBothOpen("\nPrinted intermediate AA substitution matrix to file %s\n\n", gtrFileName);
	      
	      break;
	    }

	}	  
    }
}





void modOpt(tree *tr, analdef *adef, boolean resetModel, double likelihoodEpsilon)
{ 
  int i, model, catOpt = 0; 
  double 
    currentLikelihood,
    modelEpsilon = 0.0001;
  linkageList *alphaList;
  linkageList *invarList;
  linkageList *rateList; 
  /*
    int linkedAlpha[4] = {0, 0, 0, 0};   
    int linkedInvar[4] = {0, 0, 0, 0}; 
    int linkedRates[4] = {0, 0, 0, 0};
  */  
  int *unlinked = (int *)malloc(sizeof(int) * tr->NumberOfModels);

#ifdef _IPTOL
  writeCheckpoint();
#endif
  
  if(tr->useFloat)
    modelEpsilon = 0.0002;
  else 
    modelEpsilon = 0.0001;


  for(i = 0; i < tr->NumberOfModels; i++)
    unlinked[i] = i;

  alphaList = initLinkageList(unlinked, tr);
  invarList = initLinkageList(unlinked, tr);
  rateList  = initLinkageListGTR(tr);

  if(!(adef->mode == CLASSIFY_ML))
    tr->start = tr->nodep[1];

  if(resetModel)
    {
      initRateMatrix(tr);

      for(model = 0; model < tr->NumberOfModels; model++)
	{     	  
	  if(adef->useInvariant)
	    {
	      int lower, upper;
	      int count = 0;
	      int total = 0;
	      
	      lower = tr->partitionData[model].lower;
	      upper = tr->partitionData[model].upper;
	      
	      
	      for(i = lower; i < upper; i++)
		{
		  if(tr->invariant[i] < 4) 		
		    count += tr->cdta->aliaswgt[i];		  		
		  total += tr->cdta->aliaswgt[i];
		}
	      tr->partitionData[model].propInvariant = ((double)count)/((double) total);
	    }   
       
	  tr->partitionData[model].alpha = 1.0;     

	  initReversibleGTR(tr, adef, model);      

	  makeGammaCats(tr->partitionData[model].alpha, tr->partitionData[model].gammaRates, 4); 
	}
#ifdef _USE_PTHREADS     
      masterBarrier(THREAD_RESET_MODEL ,tr);    
#endif
   
      resetBranches(tr);
      
      evaluateGenericInitrav(tr, tr->start); 
     
      treeEvaluate(tr, 0.25);     
    }

 
  /* no need for individual models here, just an init on params equal for all partitions*/

  do
    {           
      currentLikelihood = tr->likelihood;
        
#ifdef _IPTOL
      writeCheckpoint();
#endif

      optRatesGeneric(tr, adef, modelEpsilon, rateList);
     
      onlyInitrav(tr, tr->start);         
      
      treeEvaluate(tr, 0.0625);                     	            
               
      switch(tr->rateHetModel)
	{
	 
	case GAMMA_I:
	  optAlpha(tr, modelEpsilon, alphaList);
	  optInvar(tr, modelEpsilon, invarList); 	      	    	   	 
	  treeEvaluate(tr, 0.1);    	 
	  break;
	case GAMMA:      
	  optAlpha(tr, modelEpsilon, alphaList); 
	  onlyInitrav(tr, tr->start); 	 	 
	  treeEvaluate(tr, 0.1);	  	 
	  break;
	case CAT:
	  if(catOpt < 3)
	    {	      	     	     
	      optimizeRateCategories(tr, adef->categories);	      	     	      	      
	      catOpt++;
	    }
	  break;	  
	default:
	  assert(0);
	}       

#ifdef _IPTOL
      writeCheckpoint();
#endif

      printAAmatrix(tr, fabs(currentLikelihood - tr->likelihood));    
    }
  while(fabs(currentLikelihood - tr->likelihood) > likelihoodEpsilon);  
  
  free(unlinked);
  freeLinkageList(alphaList);
  freeLinkageList(rateList);
  freeLinkageList(invarList);  
}




/*********************FUNCTIONS FOOR EXACT MODEL OPTIMIZATION UNDER GTRGAMMA ***************************************/



static double branchLength(int model, double *z, tree *tr)
{
  double x;
  
  x = z[model];
  assert(x > 0);
  if (x < zmin) 
    x = zmin; 

  assert(x <= zmax);
  if(!tr->multiBranch)
    x = -log(x) * tr->fracchange;
  else
    x = -log(x) * tr->fracchanges[model];

  return x;

}


static double treeLengthRec(nodeptr p, tree *tr, int model)
{  
  double x = branchLength(model, p->z, tr);

  if(isTip(p->number, tr->rdta->numsp))  
    return x;    
  else
    {
      double acc = 0;
      nodeptr q;                
     
      q = p->next;      

      while(q != p)
	{
	  acc += treeLengthRec(q->back, tr, model);
	  q = q->next;
	}

      return acc + x;
    }
}

double treeLength(tree *tr, int model)
{ 
  return treeLengthRec(tr->start->back, tr, model);
}





