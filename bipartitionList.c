/*  RAxML-HPC, a program for sequential and parallel estimation of phylogenetic trees 
 *  Copyright March 2006 by Alexandros Stamatakis
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
 *  stamatak@ics.forth.gr
 *
 *  When publishing work that is based on the results from RAxML-VI-HPC please cite:
 *  
 *  Alexandros Stamatakis: "An Efficient Program for phylogenetic Inference Using Simulated Annealing". 
 *  Proceedings of IPDPS2005,  Denver, Colorado, April 2005.
 *  
 *  AND
 *
 *  Alexandros Stamatakis:"RAxML-VI-HPC: maximum likelihood-based phylogenetic analyses with thousands of taxa and mixed models". 
 *  Bioinformatics 2006; doi: 10.1093/bioinformatics/btl446
 */


#ifndef WIN32  
#include <sys/times.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>  
#endif

#include <limits.h>
#include <math.h>
#include <time.h> 
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include "axml.h"

#ifdef __SIM_SSE3

#include <xmmintrin.h>
#include <pmmintrin.h>

#endif

extern FILE *INFILE;
extern char run_id[128];
extern char workdir[1024];
extern char bootStrapFile[1024];
extern char tree_file[1024];
extern char infoFileName[1024];
extern char resultFileName[1024];

extern double masterTime;

extern const int mask32[32];

extern volatile branchInfo      **branchInfos;
extern volatile int NumberOfThreads;
extern volatile int NumberOfJobs;

static void mre(hashtable *h, boolean icp, entry*** sbi, int* len, int which, int n, int vectorLength, boolean sortp);

static entry *initEntry(void)
{
  entry *e = (entry*)malloc(sizeof(entry));

  e->bitVector     = (unsigned int*)NULL;
  e->treeVector    = (unsigned int*)NULL;
  e->supportVector = (int*)NULL;
  e->bipNumber  = 0;
  e->bipNumber2 = 0;
  e->supportFromTreeset[0] = 0;
  e->supportFromTreeset[1] = 0;
  e->next       = (entry*)NULL;

  return e;
} 

hashtable *initHashTable(hashNumberType n)
{
  /* 
     init with primes 
    
     static const hashNumberType initTable[] = {53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317,
     196613, 393241, 786433, 1572869, 3145739, 6291469, 12582917, 25165843,
     50331653, 100663319, 201326611, 402653189, 805306457, 1610612741};
  */

  /* init with powers of two */

  static const  hashNumberType initTable[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
					      32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
					      4194304, 8388608, 16777216, 33554432, 67108864, 134217728,
					      268435456, 536870912, 1073741824, 2147483648U};
  
  hashtable *h = (hashtable*)malloc(sizeof(hashtable));
  
  hashNumberType
    tableSize,
    i,
    primeTableLength = sizeof(initTable)/sizeof(initTable[0]),
    maxSize = (hashNumberType)-1;    

  assert(n <= maxSize);

  i = 0;

  while(initTable[i] < n && i < primeTableLength)
    i++;

  assert(i < primeTableLength);

  tableSize = initTable[i];

  /* printf("Hash table init with size %u\n", tableSize); */

  h->table = (entry**)calloc(tableSize, sizeof(entry*));
  h->tableSize = tableSize;  
  h->entryCount = 0;  

  return h;
}




void freeHashTable(hashtable *h)
{
  hashNumberType
    i,
    entryCount = 0;
   

  for(i = 0; i < h->tableSize; i++)
    {
      if(h->table[i] != NULL)
	{
	  entry *e = h->table[i];
	  entry *previous;	 

	  do
	    {
	      previous = e;
	      e = e->next;

	      if(previous->bitVector)
		free(previous->bitVector);

	      if(previous->treeVector)
		free(previous->treeVector);

	      if(previous->supportVector)
		free(previous->supportVector);
	      
	      free(previous);	      
	      entryCount++;
	    }
	  while(e != NULL);	  
	}

    }

  assert(entryCount == h->entryCount);
 
  free(h->table);
}

#ifdef DEVELOPMENT
static void analyzeHashTable(hashtable *h)
{
  hashNumberType
    i, 
    entryCount     = 0,
    maxChainLength = 0,
    collisionCount = 0;

  for(i = 0; i < h->tableSize; i++)
    {
      if(h->table[i] != NULL)
	{
	  entry *e = h->table[i];	
	  hashNumberType count = 0;

	  do
	    {	   
	      e = e->next;
	      
	      count++;
	      entryCount++;
	    }
	  while(e != NULL);	  

	  if(count > 1)
	    {
	      collisionCount++;
	      if(count > maxChainLength)
		maxChainLength = count;
	    }
	}

    }

  printf("Hash Table Stats: \n max chain: %u \n collisions: %u \n entries: %u\n", maxChainLength, collisionCount, entryCount);
}
#endif

void cleanupHashTable(hashtable *h, int state)
{
  hashNumberType
    k,
    entryCount = 0,
    removeCount = 0;
 
  assert(state == 1 || state == 0);

  for(k = 0, entryCount = 0; k < h->tableSize; k++)	     
    {      
      if(h->table[k] != NULL)
	{
	  entry *e = h->table[k];
	  entry *start     = (entry*)NULL;
	  entry *lastValid = (entry*)NULL;
	  	  
	  do
	    {	   	 	      	
	      if(state == 0)
		{
		  e->treeVector[0] = e->treeVector[0] & 2;	
		  assert(!(e->treeVector[0] & 1));
		}
	      else
		{
		  e->treeVector[0] = e->treeVector[0] & 1;
		  assert(!(e->treeVector[0] & 2));
		}
	      
	      if(e->treeVector[0] != 0)
		{
		  if(!start)
		    start = e;
		  lastValid = e;
		  e = e->next;
		}	  
	      else
		{
		  entry *remove = e;
		  e = e->next;
		  
		  removeCount++;

		  if(lastValid)		    		    
		    lastValid->next = remove->next;

		  if(remove->bitVector)
		    free(remove->bitVector);
		  if(remove->treeVector)
		    free(remove->treeVector);
		  if(remove->supportVector)
		    free(remove->supportVector);
		  free(remove);		 
		}
	      
	      entryCount++;	     	     
	    }
	  while(e != NULL);	 

	  if(!start)
	    {
	      assert(!lastValid);
	      h->table[k] = NULL;
	    }
	  else
	    {
	      h->table[k] = start;
	    }	 	 
	}    
    }

  assert(entryCount ==  h->entryCount);  

  h->entryCount -= removeCount;
}





/*
  #define NN 312
  #define MM 156
  #define MATRIX_A 0xB5026F5AA96619E9LLU
  #define UM 0xFFFFFFFF80000000LLU 
  #define LM 0x7FFFFFFFLLU 
  
  static uint64_t mt[NN]; 
  static int mti=NN+1; 
  
  
  static void init_genrand64(uint64_t seed)
  {
  mt[0] = seed;
  for (mti=1; mti<NN; mti++) 
  mt[mti] =  (6364136223846793005LLU * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
  }  
  
  static void init_by_array64(uint64_t init_key[],
  uint64_t key_length)
  {
  uint64_t i, j, k;
  init_genrand64(19650218LLU);
  i=1; j=0;
  k = (NN>key_length ? NN : key_length);
  for (; k; k--) {
  mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 3935559000370003845LLU))
  + init_key[j] + j;
  i++; j++;
  if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
  if (j>=key_length) j=0;
  }
  for (k=NN-1; k; k--) {
  mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 2862933555777941757LLU))
  - i;
  i++;
  if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
  }
  
  mt[0] = 1LLU << 63;
  }
  
  
  static uint64_t genrand64_int64(void)
  {
  int i;
  uint64_t x;
  static uint64_t mag01[2]={0LLU, MATRIX_A};
  
  if (mti >= NN) {
  
  
  if (mti == NN+1) 
  init_genrand64(5489LLU); 
  
  for (i=0;i<NN-MM;i++) {
  x = (mt[i]&UM)|(mt[i+1]&LM);
  mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1LLU)];
  }
  for (;i<NN-1;i++) {
  x = (mt[i]&UM)|(mt[i+1]&LM);
  mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1LLU)];
  }
  x = (mt[NN-1]&UM)|(mt[0]&LM);
  mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1LLU)];
  
  mti = 0;
  }
  
  x = mt[mti++];
  
  x ^= (x >> 29) & 0x5555555555555555LLU;
  x ^= (x << 17) & 0x71D67FFFEDA60000LLU;
  x ^= (x << 37) & 0xFFF7EEE000000000LLU;
  x ^= (x >> 43);
  
  return x;
  }
*/






unsigned int **initBitVector(tree *tr, int *vectorLength)
{
  unsigned int **bitVectors = (unsigned int **)malloc(sizeof(unsigned int*) * 2 * tr->mxtips);
  int i;

  if(tr->mxtips % MASK_LENGTH == 0)
    *vectorLength = tr->mxtips / MASK_LENGTH;
  else
    *vectorLength = 1 + (tr->mxtips / MASK_LENGTH); 
  
  for(i = 1; i <= tr->mxtips; i++)
    {
      bitVectors[i] = (unsigned int *)calloc(*vectorLength, sizeof(unsigned int));
      bitVectors[i][(i - 1) / MASK_LENGTH] |= mask32[(i - 1) % MASK_LENGTH];
    }
  
  for(i = tr->mxtips + 1; i < 2 * tr->mxtips; i++) 
    bitVectors[i] = (unsigned int *)malloc(sizeof(unsigned int) * *vectorLength);

  return bitVectors;
}

void freeBitVectors(unsigned int **v, int n)
{
  int i;

  for(i = 1; i < n; i++)
    free(v[i]);
}





static void newviewBipartitions(unsigned int **bitVectors, nodeptr p, int numsp, int vectorLength)
{
  if(isTip(p->number, numsp))
    return;
  {
    nodeptr 
      q = p->next->back, 
      r = p->next->next->back;
    unsigned int       
      *vector = bitVectors[p->number],
      *left  = bitVectors[q->number],
      *right = bitVectors[r->number];
    int i;           

    while(!p->x)
      {	
	if(!p->x)
	  getxnode(p);
      }

    p->hash = q->hash ^ r->hash;

    if(isTip(q->number, numsp) && isTip(r->number, numsp))
      {		
	for(i = 0; i < vectorLength; i++)
	  vector[i] = left[i] | right[i];	  	
      }
    else
      {	
	if(isTip(q->number, numsp) || isTip(r->number, numsp))
	  {
	    if(isTip(r->number, numsp))
	      {	
		nodeptr tmp = r;
		r = q;
		q = tmp;
	      }	   
	    	    
	    while(!r->x)
	      {
		if(!r->x)
		  newviewBipartitions(bitVectors, r, numsp, vectorLength);
	      }	   

	    for(i = 0; i < vectorLength; i++)
	      vector[i] = left[i] | right[i];	    	 
	  }
	else
	  {	    
	    while((!r->x) || (!q->x))
	      {
		if(!q->x)
		  newviewBipartitions(bitVectors, q, numsp, vectorLength);
		if(!r->x)
		  newviewBipartitions(bitVectors, r, numsp, vectorLength);
	      }	   	    	    	    	   

	    for(i = 0; i < vectorLength; i++)
	      vector[i] = left[i] | right[i];	 
	  }

      }     
  }     
}

static void insertHash(unsigned int *bitVector, hashtable *h, int vectorLength, int bipNumber, hashNumberType position)
{
  entry *e = initEntry();

  e->bipNumber = bipNumber; 
  /*e->bitVector = (unsigned int*)calloc(vectorLength, sizeof(unsigned int)); */

  e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
  memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));
 
  memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);
  
  if(h->table[position] != NULL)
    {
      e->next = h->table[position];
      h->table[position] = e;           
    }
  else
    h->table[position] = e;

  h->entryCount =  h->entryCount + 1;
}



static int countHash(unsigned int *bitVector, hashtable *h, int vectorLength, hashNumberType position)
{ 
  if(h->table[position] == NULL)         
    return -1;
  {
    entry *e = h->table[position];     

    do
      {	 
	int i;

	for(i = 0; i < vectorLength; i++)
	  if(bitVector[i] != e->bitVector[i])
	    goto NEXT;
	   
	return (e->bipNumber);	 
      NEXT:
	e = e->next;
      }
    while(e != (entry*)NULL); 
     
    return -1;   
  }

}

static void insertHashAll(unsigned int *bitVector, hashtable *h, int vectorLength, int treeNumber,  hashNumberType position)
{    
  if(h->table[position] != NULL)
    {
      entry *e = h->table[position];     

      do
	{	 
	  int i;
	  
	  for(i = 0; i < vectorLength; i++)
	    if(bitVector[i] != e->bitVector[i])
	      break;
	  
	  if(i == vectorLength)
	    {
	      if(treeNumber == 0)
		e->bipNumber = 	e->bipNumber  + 1;
	      else
		e->bipNumber2 = e->bipNumber2 + 1;
	      return;
	    }
	  
	  e = e->next;	 
	}
      while(e != (entry*)NULL); 

      e = initEntry(); 
  
      /*e->bitVector  = (unsigned int*)calloc(vectorLength, sizeof(unsigned int)); */
      e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
      memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));


      memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);

      if(treeNumber == 0)	
	e->bipNumber  = 1;       	
      else		 
	e->bipNumber2 = 1;
	
      e->next = h->table[position];
      h->table[position] = e;              
    }
  else
    {
      entry *e = initEntry(); 
  
      /*e->bitVector  = (unsigned int*)calloc(vectorLength, sizeof(unsigned int)); */

      e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
      memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));

      memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);

      if(treeNumber == 0)	
	e->bipNumber  = 1;	  	
      else    
	e->bipNumber2 = 1;	

      h->table[position] = e;
    }

  h->entryCount =  h->entryCount + 1;
}




static void insertHashBootstop(unsigned int *bitVector, hashtable *h, int vectorLength, int treeNumber, int treeVectorLength, hashNumberType position)
{    
  if(h->table[position] != NULL)
    {
      entry *e = h->table[position];     

      do
	{	 
	  int i;
	  
	  for(i = 0; i < vectorLength; i++)
	    if(bitVector[i] != e->bitVector[i])
	      break;
	  
	  if(i == vectorLength)
	    {
	      e->treeVector[treeNumber / MASK_LENGTH] |= mask32[treeNumber % MASK_LENGTH];
	      return;
	    }
	  
	  e = e->next;
	}
      while(e != (entry*)NULL); 

      e = initEntry(); 

      e->bipNumber = h->entryCount;
       
      /*e->bitVector  = (unsigned int*)calloc(vectorLength, sizeof(unsigned int));*/
      e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
      memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));


      e->treeVector = (unsigned int*)calloc(treeVectorLength, sizeof(unsigned int));
      
      e->treeVector[treeNumber / MASK_LENGTH] |= mask32[treeNumber % MASK_LENGTH];
      memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);
     
      e->next = h->table[position];
      h->table[position] = e;          
    }
  else
    {
      entry *e = initEntry(); 

      e->bipNumber = h->entryCount;

      /*e->bitVector  = (unsigned int*)calloc(vectorLength, sizeof(unsigned int));*/

      e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
      memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));

      e->treeVector = (unsigned int*)calloc(treeVectorLength, sizeof(unsigned int));

      e->treeVector[treeNumber / MASK_LENGTH] |= mask32[treeNumber % MASK_LENGTH];
      memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);     

      h->table[position] = e;
    }

  h->entryCount =  h->entryCount + 1;
}

static void insertHashRF(unsigned int *bitVector, hashtable *h, int vectorLength, int treeNumber, int treeVectorLength, hashNumberType position, int support, 
			 boolean computeWRF)
{     
  if(h->table[position] != NULL)
    {
      entry *e = h->table[position];     

      do
	{	 
	  int i;
	  
	  for(i = 0; i < vectorLength; i++)
	    if(bitVector[i] != e->bitVector[i])
	      break;
	  
	  if(i == vectorLength)
	    {
	      e->treeVector[treeNumber / MASK_LENGTH] |= mask32[treeNumber % MASK_LENGTH];
	      if(computeWRF)
		{
		  e->supportVector[treeNumber] = support;
		 
		  assert(0 <= treeNumber && treeNumber < treeVectorLength * MASK_LENGTH);
		}
	      return;
	    }
	  
	  e = e->next;
	}
      while(e != (entry*)NULL); 

      e = initEntry(); 
       
      /*e->bitVector  = (unsigned int*)calloc(vectorLength, sizeof(unsigned int));*/
      e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
      memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));


      e->treeVector = (unsigned int*)calloc(treeVectorLength, sizeof(unsigned int));
      if(computeWRF)
	e->supportVector = (int*)calloc(treeVectorLength * MASK_LENGTH, sizeof(int));

      e->treeVector[treeNumber / MASK_LENGTH] |= mask32[treeNumber % MASK_LENGTH];
      if(computeWRF)
	{
	  e->supportVector[treeNumber] = support;
	 
	  assert(0 <= treeNumber && treeNumber < treeVectorLength * MASK_LENGTH);
	}

      memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);
     
      e->next = h->table[position];
      h->table[position] = e;          
    }
  else
    {
      entry *e = initEntry(); 
       
      /*e->bitVector  = (unsigned int*)calloc(vectorLength, sizeof(unsigned int)); */

      e->bitVector = (unsigned int*)malloc_aligned(vectorLength * sizeof(unsigned int));
      memset(e->bitVector, 0, vectorLength * sizeof(unsigned int));

      e->treeVector = (unsigned int*)calloc(treeVectorLength, sizeof(unsigned int));
      if(computeWRF)	
	e->supportVector = (int*)calloc(treeVectorLength * MASK_LENGTH, sizeof(int));


      e->treeVector[treeNumber / MASK_LENGTH] |= mask32[treeNumber % MASK_LENGTH];
      if(computeWRF)
	{
	  e->supportVector[treeNumber] = support;
	 
	  assert(0 <= treeNumber && treeNumber < treeVectorLength * MASK_LENGTH);
	}

      memcpy(e->bitVector, bitVector, sizeof(unsigned int) * vectorLength);     

      h->table[position] = e;
    }

  h->entryCount =  h->entryCount + 1;
}



void bitVectorInitravSpecial(unsigned int **bitVectors, nodeptr p, int numsp, int vectorLength, hashtable *h, int treeNumber, int function, branchInfo *bInf, 
			     int *countBranches, int treeVectorLength, boolean traverseOnly, boolean computeWRF)
{
  if(isTip(p->number, numsp))
    return;
  else
    {
      nodeptr q = p->next;          

      do 
	{
	  bitVectorInitravSpecial(bitVectors, q->back, numsp, vectorLength, h, treeNumber, function, bInf, countBranches, treeVectorLength, traverseOnly, computeWRF);
	  q = q->next;
	}
      while(q != p);
           
      newviewBipartitions(bitVectors, p, numsp, vectorLength);
      
      assert(p->x);

      if(traverseOnly)
	{
	  if(!(isTip(p->back->number, numsp)))
	    *countBranches =  *countBranches + 1;
	  return;
	}

      if(!(isTip(p->back->number, numsp)))
	{
	  unsigned int *toInsert  = bitVectors[p->number];
	  hashNumberType position = p->hash % h->tableSize;
	 
	  assert(!(toInsert[0] & 1));	 

	  switch(function)
	    {
	    case BIPARTITIONS_ALL:	      
	      insertHashAll(toInsert, h, vectorLength, treeNumber, position);
	      *countBranches =  *countBranches + 1;	
	      break;
	    case GET_BIPARTITIONS_BEST:	   	     
	      insertHash(toInsert, h, vectorLength, *countBranches, position);	     
	      
	      p->bInf            = &bInf[*countBranches];
	      p->back->bInf      = &bInf[*countBranches];        
	      p->bInf->support   = 0;	  	 
	      p->bInf->oP = p;
	      p->bInf->oQ = p->back;
	      
	      *countBranches =  *countBranches + 1;		
	      break;
	    case DRAW_BIPARTITIONS_BEST:	     
	      {
		int found = countHash(toInsert, h, vectorLength, position);
		if(found >= 0)
		  bInf[found].support =  bInf[found].support + 1;
		*countBranches =  *countBranches + 1;
	      }	      
	      break;
	    case BIPARTITIONS_BOOTSTOP:	      
	      insertHashBootstop(toInsert, h, vectorLength, treeNumber, treeVectorLength, position);
	      *countBranches =  *countBranches + 1;
	      break;
	    case BIPARTITIONS_RF:
	      if(computeWRF)
		assert(p->support == p->back->support);
	      insertHashRF(toInsert, h, vectorLength, treeNumber, treeVectorLength, position, p->support, computeWRF);
	      *countBranches =  *countBranches + 1;
	      break;
	    default:
	      assert(0);
	    }	  	  
	}
      
    }
}

static void linkBipartitions(nodeptr p, tree *tr, branchInfo *bInf, int *counter, int numberOfTrees)
{
  if(isTip(p->number, tr->mxtips))    
    {
      assert(p->bInf == (branchInfo*) NULL && p->back->bInf == (branchInfo*) NULL);      
      return;
    }
  else
    {
      nodeptr q;          
      
      q = p->next;

      while(q != p)
	{
	  linkBipartitions(q->back, tr, bInf, counter, numberOfTrees);	
	  q = q->next;
	}
     
      if(!(isTip(p->back->number, tr->mxtips)))
	{
	  double support;

	  p->bInf       = &bInf[*counter];
	  p->back->bInf = &bInf[*counter]; 

	  support = ((double)(p->bInf->support)) / ((double) (numberOfTrees));
	  p->bInf->support = (int)(0.5 + support * 100.0);	 	       	  

	  assert(p->bInf->oP == p);
	  assert(p->bInf->oQ == p->back);
	  
	  *counter = *counter + 1;
	}


      return;
    }
}


static void loadTreeFileToMemory(tree *tr, char *fileName, analdef *adef)
{
  FILE 
    *treeFile;

  long
    readSize,
    sizeCleaned,
    size = 0,
    i = 0;    

  int         
    numberOfTrees = 0,
    numberOfTrees2 = 0;
    
  boolean
    foundEnd;

  char 
    *buffer,
    *treeBuffer,
    **treeStarts;
                

  assert(tr->numberOfTrees == -1);
  assert(!(tr->treeStarts));
  assert(!(tr->treeBuffer));

  treeFile = myfopen(fileName, "r");           
  
  fseek(treeFile, 0, SEEK_END);
  size = ftell(treeFile);  

  treeBuffer = (char *)malloc((size+1) * sizeof(char));
  treeBuffer[size] = 0; 

  buffer = (char *)malloc((size+1) * sizeof(char));
  buffer[size] = 0; 
  
  fseek(treeFile, 0, SEEK_SET);       
  readSize = fread(buffer, sizeof(char), size, treeFile);  
  assert(readSize == size);
  
  fclose(treeFile);

  for(i = 0, sizeCleaned = 0, numberOfTrees = 0; i < size; i++)
    {
      char ch = buffer[i];
      
      /* remove all white chars */
      
      if(!(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r'))
	treeBuffer[sizeCleaned++] = ch;

      /* count trees */
      
      if(ch == ';')
	numberOfTrees++;
    }
  
  treeBuffer[sizeCleaned] = 0;

  size = sizeCleaned;
  
  free(buffer); 	   	   
      
  if(!adef->allInOne)   
    printBothOpen("\n\nFound %d trees in File %s\n\n", numberOfTrees, fileName);

  treeStarts = (char **)malloc(numberOfTrees * sizeof(char *));
       	
  treeStarts[0] = &treeBuffer[0];      

  for(i = 1, foundEnd = FALSE, numberOfTrees2 = 1; i < size; i++)
    {
      if(treeBuffer[i] == ';')
	foundEnd = TRUE;
      else
	{
	  if(foundEnd && treeBuffer[i] == '(')
	    {
	      treeStarts[numberOfTrees2++] = &treeBuffer[i];
	      foundEnd = FALSE;
	    }
	}
    }    
                 
  assert(numberOfTrees == numberOfTrees2);

  tr->numberOfTrees = numberOfTrees;
  tr->treeStarts = treeStarts;
  tr->treeBuffer = treeBuffer;
}


static void freeTreeFileInMemory(tree *tr)
{
  tr->numberOfTrees = -1;
  free(tr->treeStarts);
  tr->treeStarts = (char**)NULL;
  free(tr->treeBuffer);
  tr->treeBuffer = (char*)NULL;
}

static void readSingleTree(tree *tr, char *fileName, analdef *adef, boolean readBranches, boolean readNodeLabels)
{
  int 
    position = 0;

  loadTreeFileToMemory(tr, fileName, adef);
  assert(tr->numberOfTrees == 1);
  treeReadTopologyString(tr->treeStarts[0], tr, readBranches, readNodeLabels, &position, TRUE, adef);
  freeTreeFileInMemory(tr);
}

void calcBipartitions(tree *tr, analdef *adef, char *bestTreeFileName, char *bootStrapFileName)
{  
  branchInfo 
    *bInf;
  
  int 
    branchCounter = 0,
    counter = 0,
    vLength,
    numberOfTrees = 0,
    i;

  unsigned int 
    **bitVectors = initBitVector(tr, &vLength);
  
  hashtable *h = 
    initHashTable(tr->mxtips * 10);
  
  readSingleTree(tr, bestTreeFileName, adef, FALSE, FALSE);    
  
  bInf = (branchInfo*)malloc(sizeof(branchInfo) * (tr->mxtips - 3));

  bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vLength, h, 0, GET_BIPARTITIONS_BEST, bInf, &branchCounter, 0, FALSE, FALSE);   
 
  assert((int)h->entryCount == (tr->mxtips - 3));  
  assert(branchCounter == (tr->mxtips - 3));

  loadTreeFileToMemory(tr, bootStrapFileName, adef);
  numberOfTrees = tr->numberOfTrees;

#ifdef _USE_PTHREADS              
  tr->h = h;            
      
  NumberOfJobs = tr->numberOfTrees;
    
 
  masterBarrier(THREAD_DRAW_BIPARTITIONS, tr);        
  
    
  for(i = 0; i < (tr->mxtips - 3); i++)
    {
      int k;
      
      bInf[i].support = 0;
      
      for(k = 0; k < NumberOfThreads; k++)
	bInf[i].support += branchInfos[k][i].support;
    }
    
  masterBarrier(THREAD_FREE_DRAW_BIPARTITIONS, tr);    
#else  
  for(i = 0; i < numberOfTrees; i++)
    {                
      int 
	bCount = 0,
	position = 0;
	  	 
      treeReadTopologyString(tr->treeStarts[i], tr, FALSE, FALSE, &position, TRUE, adef);     
      assert(tr->ntips == tr->mxtips);
	 
      bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vLength, h, 0, DRAW_BIPARTITIONS_BEST, bInf, &bCount, 0, FALSE, FALSE);      
      assert(bCount == tr->mxtips - 3);      
    }  
#endif 
  

  freeTreeFileInMemory(tr);
   
  readSingleTree(tr, bestTreeFileName, adef, TRUE, FALSE); 
   
  linkBipartitions(tr->nodep[1]->back, tr, bInf, &counter, numberOfTrees);

  assert(counter == branchCounter);

  printBipartitionResult(tr, adef, TRUE);    

  freeBitVectors(bitVectors, 2 * tr->mxtips);
  free(bitVectors);
  freeHashTable(h);
  free(h); 

  free(bInf);
}

/*************************************************************/

static double testFreq(double *vect1, double *vect2, int n);



void compareBips(tree *tr, char *bootStrapFileName, analdef *adef)
{
  int 
    numberOfTreesAll = 0, 
    numberOfTreesStop = 0,
    i; 
  unsigned int k, entryCount;
  double *vect1, *vect2, p, avg1 = 0.0, avg2 = 0.0, scaleAll, scaleStop;
  int 
    bipAll = 0,
    bipStop = 0;
  char bipFileName[1024];
  FILE *outf; 
  int vLength;
  unsigned int **bitVectors = initBitVector(tr, &vLength);
  hashtable *h = initHashTable(tr->mxtips * 100);    
  unsigned long int c1 = 0;
  unsigned long int c2 = 0;   


  /*********************************************************************************************************/
  
  loadTreeFileToMemory(tr, bootStrapFileName, adef);
  
  numberOfTreesAll = tr->numberOfTrees;
              
  for(i = 0; i < numberOfTreesAll; i++)
    { 
      int 
	bCounter = 0,
	position = 0;
      
      treeReadTopologyString(tr->treeStarts[i], tr, FALSE, FALSE, &position, TRUE, adef);                
      assert(tr->mxtips == tr->ntips); 

      bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vLength, h, 0, BIPARTITIONS_ALL, (branchInfo*)NULL, &bCounter, 0, FALSE, FALSE);
      assert(bCounter == tr->mxtips - 3);      
    }
	  
  freeTreeFileInMemory(tr);


  /*********************************************************************************************************/   

  loadTreeFileToMemory(tr, tree_file, adef);
  
  numberOfTreesStop = tr->numberOfTrees;   
       
  for(i = 0; i < numberOfTreesStop; i++)
    {              
      int 
	bCounter = 0,
	position = 0;


      treeReadTopologyString(tr->treeStarts[i], tr, FALSE, FALSE, &position, TRUE, adef);      
      assert(tr->mxtips == tr->ntips);
      
      bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vLength, h, 1, BIPARTITIONS_ALL, (branchInfo*)NULL, &bCounter, 0, FALSE, FALSE);
      assert(bCounter == tr->mxtips - 3);     
    }
	  
  freeTreeFileInMemory(tr);  

  /***************************************************************************************************/
      
  vect1 = (double *)malloc(sizeof(double) * h->entryCount);
  vect2 = (double *)malloc(sizeof(double) * h->entryCount);

  strcpy(bipFileName,         workdir);  
  strcat(bipFileName,         "RAxML_bipartitionFrequencies.");
  strcat(bipFileName,         run_id);

  outf = myfopen(bipFileName, "w");


  scaleAll  = 1.0 / (double)numberOfTreesAll;
  scaleStop = 1.0 / (double)numberOfTreesStop;

  for(k = 0, entryCount = 0; k < h->tableSize; k++)	     
    {
      
      if(h->table[k] != NULL)
	{
	  entry *e = h->table[k];

	  do
	    {
	      c1 += e->bipNumber;
	      c2 += e->bipNumber2;
	      vect1[entryCount] = ((double)e->bipNumber) * scaleAll;
	      if(vect1[entryCount] > 0)
		bipAll++;
	      vect2[entryCount] = ((double)e->bipNumber2) * scaleStop;
	      if(vect2[entryCount] > 0)
		bipStop++;
	      fprintf(outf, "%f %f\n", vect1[entryCount], vect2[entryCount]);
	      entryCount++;
	      e = e->next;
	    }
	  while(e != NULL);
	}     
    }
  
  printBothOpen("%ld %ld\n", c1, c2);

  assert(entryCount == h->entryCount);

  fclose(outf);

  p = testFreq(vect1, vect2, h->entryCount);

  for(k = 0; k < h->entryCount; k++)
    {
      avg1 += vect1[k];
      avg2 += vect2[k];
    }

  avg1 /= ((double)h->entryCount);
  avg2 /= ((double)h->entryCount);
  
  
  printBothOpen("Average [%s]: %1.40f [%s]: %1.40f\n", bootStrapFileName, avg1, tree_file, avg2);
  printBothOpen("Pearson: %f Bipartitions in [%s]: %d Bipartitions in [%s]: %d Total Bipartitions: %d\n", p, bootStrapFileName, bipAll, tree_file, bipStop, h->entryCount);

  printBothOpen("\nFile containing pair-wise bipartition frequencies written to %s\n\n", bipFileName);
  
  freeBitVectors(bitVectors, 2 * tr->mxtips);
  free(bitVectors);
  freeHashTable(h);
  free(h);

  free(vect1);
  free(vect2);

  exit(0);
}

/*************************************************************************************************************/

void computeRF(tree *tr, char *bootStrapFileName, analdef *adef)
{
  int     
    treeVectorLength, 
    numberOfTrees = 0, 
    i,
    j, 
    *rfMat,
    *wrfMat,
    *wrf2Mat,
    *presentList,
    vLength; 

  unsigned int 
    k, 
    entryCount,    
    **bitVectors = initBitVector(tr, &vLength);
  
  char rfFileName[1024];

  boolean computeWRF = FALSE;

  double 
    maxRF, 
    avgRF,
    avgWRF,
    avgWRF2;

  FILE *outf;

  hashtable *h = (hashtable *)NULL;   

  loadTreeFileToMemory(tr, bootStrapFileName, adef);
  numberOfTrees = tr->numberOfTrees;
  
  h = initHashTable(tr->mxtips * 2 * numberOfTrees); 

  if(numberOfTrees % MASK_LENGTH == 0)
    treeVectorLength = numberOfTrees / MASK_LENGTH;
  else
    treeVectorLength = 1 + (numberOfTrees / MASK_LENGTH);

  presentList = (int*)malloc(numberOfTrees * sizeof(int));
  rfMat = (int*)calloc(numberOfTrees * numberOfTrees, sizeof(int));
  wrfMat = (int*)calloc(numberOfTrees * numberOfTrees, sizeof(int));
  wrf2Mat = (int*)calloc(numberOfTrees * numberOfTrees, sizeof(int));

  for(i = 0; i < numberOfTrees; i++)
    { 
      int 
	bCounter = 0,
	position = 0,
	lcount   = 0;
      
      lcount = treeReadTopologyString(tr->treeStarts[i], tr, FALSE, TRUE, &position, TRUE, adef); 
            
      assert(tr->mxtips == tr->ntips); 
      
      if(i == 0)
	{
	  assert(lcount == 0 || lcount == tr->mxtips - 3);
	  if(lcount == 0)
	    computeWRF = FALSE;
	  else
	    computeWRF = TRUE;
	}
     
      if(computeWRF)
	assert(lcount == tr->mxtips - 3);     	

      bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vLength, h, i, BIPARTITIONS_RF, (branchInfo *)NULL,
			      &bCounter, treeVectorLength, FALSE, computeWRF);
     
      assert(bCounter == tr->mxtips - 3);          
    }
	  
  freeTreeFileInMemory(tr);  

  for(k = 0, entryCount = 0; k < h->tableSize; k++)	     
    {    
      if(h->table[k] != NULL)
	{
	  entry *e = h->table[k];

	  do
	    {
	      unsigned int *vector = e->treeVector;
	      	    	      
	      if(!computeWRF)
		{	
		  i = 0;
		  while(i < numberOfTrees)
		    {
		      if(vector[i / MASK_LENGTH] == 0)
			i += MASK_LENGTH;
		      else
			{		     
			  if((vector[i / MASK_LENGTH] & mask32[i % MASK_LENGTH]) > 0)
			    {			
			      int *r = &rfMat[i * numberOfTrees];

			      for(j = 0; j < numberOfTrees; j++)
				if((vector[j / MASK_LENGTH] & mask32[j % MASK_LENGTH]) == 0)
				  r[j]++;			     			    
			    }
			  i++;
			}
		    }	 		  		  		  
		}
	      else
		{
		  int *supportVector = e->supportVector;

		  i = 0;

		  while(i < numberOfTrees)
		    {
		      if(vector[i / MASK_LENGTH] == 0)
			i += MASK_LENGTH;
		      else
			{		     
			  if((vector[i / MASK_LENGTH] & mask32[i % MASK_LENGTH]) > 0)
			    {			
			      int 
				*r = &rfMat[i * numberOfTrees],
				*w   = &wrfMat[i * numberOfTrees],
				*w2  = &wrf2Mat[i * numberOfTrees],
				support = supportVector[i];
			      
			      

			      for(j = 0; j < numberOfTrees; j++)
				{
				  if((vector[j / MASK_LENGTH] & mask32[j % MASK_LENGTH]) == 0)
				    {
				      r[j]++;			     			    
				      w[j] += ABS(support - supportVector[j]);
				      w2[j] += ABS(support - supportVector[j]);
				    }
				  else
				    {
				      w2[j] += ABS(support - supportVector[j]);
				    }

				}
			    }
			  i++;
			}
		    }			  
		}      
	      
	      entryCount++;
	      e = e->next;
	    }
	  while(e != NULL);
	}

     
    }
  assert(entryCount == h->entryCount);  


  strcpy(rfFileName,         workdir);  
  strcat(rfFileName,         "RAxML_RF-Distances.");
  strcat(rfFileName,         run_id);

  outf = myfopen(rfFileName, "w");
  
  maxRF  = ((double)(2 * (tr->mxtips - 3)));
  avgRF  = 0.0;
  avgWRF = 0.0;
  avgWRF2 = 0.0;

  if(!computeWRF)
    {
      for(i = 0; i < numberOfTrees; i++)    
	for(j = i + 1; j < numberOfTrees; j++)
	  rfMat[i * numberOfTrees + j] += rfMat[j * numberOfTrees + i];
    }
  else
    {
      for(i = 0; i < numberOfTrees; i++)    
	for(j = i + 1; j < numberOfTrees; j++)
	  {
	    rfMat[i * numberOfTrees + j] += rfMat[j * numberOfTrees + i];
	    wrfMat[i * numberOfTrees + j] += wrfMat[j * numberOfTrees + i];
	    wrf2Mat[i * numberOfTrees + j] += wrf2Mat[j * numberOfTrees + i];
	  }
    }

  for(i = 0; i < numberOfTrees; i++)
    for(j = i + 1; j < numberOfTrees; j++)
      {
	int    rf = rfMat[i * numberOfTrees + j];
	double rrf = (double)rf / maxRF;
	if(computeWRF)
	  {
	    double wrf = wrfMat[i * numberOfTrees + j] / 100.0;
	    double rwrf = wrf / maxRF;
	    double wrf2 = wrf2Mat[i * numberOfTrees + j] / 100.0;
	    double rwrf2 = wrf2 / maxRF;
	
	    fprintf(outf, "%d %d: %d %f, %f %f, %f %f\n", i, j, rf, rrf, wrf, rwrf, wrf2, rwrf2);
	    avgWRF  += rwrf;
	    avgWRF2 += rwrf2; 
	  }
	else
	  fprintf(outf, "%d %d: %d %f\n", i, j, rf, rrf);
	
	avgRF += rrf;
      }
  
  fclose(outf);

  
  printBothOpen("\n\nAverage relative RF in this set: %f\n", avgRF / ((double)((numberOfTrees * numberOfTrees - numberOfTrees) / 2)));
  if(computeWRF)
    {
      printBothOpen("\n\nAverage relative WRF in this set: %f\n", avgWRF / ((double)((numberOfTrees * numberOfTrees - numberOfTrees) / 2)));
      printBothOpen("\n\nAverage relative WRF2 in this set: %f\n", avgWRF2 / ((double)((numberOfTrees * numberOfTrees - numberOfTrees) / 2)));
      printBothOpen("\nFile containing all %d pair-wise RF, WRF and WRF2 distances written to file %s\n\n", (numberOfTrees * numberOfTrees - numberOfTrees) / 2, rfFileName);
    }    
  else
    printBothOpen("\nFile containing all %d pair-wise RF distances written to file %s\n\n", (numberOfTrees * numberOfTrees - numberOfTrees) / 2, rfFileName);

  free(rfMat);
  free(wrfMat);
  free(wrf2Mat);
  freeBitVectors(bitVectors, 2 * tr->mxtips);
  free(bitVectors);
  freeHashTable(h);
  free(h);  

  exit(0);
}

double convergenceCriterion(hashtable *h, int mxtips)
{
  int      
    rf = 0; 

  unsigned int 
    k = 0, 
    entryCount = 0;
  
  double    
    rrf;  

  for(k = 0, entryCount = 0; k < h->tableSize; k++)	     
    {      
      if(h->table[k] != NULL)
	{
	  entry *e = h->table[k];

	  do
	    {
	      unsigned int *vector = e->treeVector;	     
	      if(((vector[0] & 1) > 0) + ((vector[0] & 2) > 0) == 1)
		rf++;	     
	      
	      entryCount++;
	      e = e->next;
	    }
	  while(e != NULL);
	}     
    }

  assert(entryCount == h->entryCount);  
      
  rrf = (double)rf/((double)(2 * (mxtips - 3)));  
 
  return rrf;
}




/*************************************************************************************************************/

static void permute(unsigned int *perm, unsigned int n, long *seed)
{
  unsigned int  i, j, k;
 
  for (i = 0; i < n; i++) 
    {
      k =  (int)((double)(n - i) * randum(seed));
      j        = perm[i];    
      perm[i]     = perm[i + k];
      perm[i + k] = j; 
      /*assert(i + k < n);*/
    }
}





static double testFreq(double *vect1, double *vect2, int n)
{
  int 
    i;
  
  boolean 
    allEqual = TRUE;

  double
    avg1 = 0.0, 
    avg2 = 0.0,
    sum_xy = 0.0, 
    sum_x  = 0.0, 
    sum_y  = 0.0,
    corr   = 0.0;
 
  for(i = 0; i < n; i++)
    {	     
      allEqual = allEqual && (vect1[i] == vect2[i]);

      avg1 += vect1[i];
      avg2 += vect2[i];
    }
      
  avg1 /= ((double)n);
  avg2 /= ((double)n); 
      
  for(i = 0; i < n; i++)
    {
      sum_xy += ((vect1[i] - avg1) * (vect2[i] - avg2));
      sum_x  += ((vect1[i] - avg1) * (vect1[i] - avg1));
      sum_y  += ((vect2[i] - avg2) * (vect2[i] - avg2));	 
    }       

  if(allEqual)
    return 1.0;

  if(sum_x == 0.0 || sum_y == 0.0) 
    return 0.0;

  corr = sum_xy / (sqrt(sum_x) * sqrt(sum_y));
   
  /*
    #ifndef WIN32
    if(isnan(corr))
    {
    printf("Numerical Error pearson correlation is not a number\n");
    assert(0);
    }
    #endif
  */

  return corr;
}

static double frequencyCriterion(int numberOfTrees, hashtable *h, int *countBetter)
{
  int 
    k, 
    l;
    
  long 
    seed = 12345;

  double 
    t,
    result, 
    avg = 0, 
    *vect1, 
    *vect2; 

  unsigned int
    *perm =  (unsigned int *)malloc(sizeof(unsigned int) * numberOfTrees),
    j;

  assert(*countBetter == 0);
	  
 
	  
  for(j = 0; j < (unsigned int)numberOfTrees; j++)
    perm[j] = j;
	  
  for(k = 0; k < BOOTSTOP_PERMUTATIONS; k++)
    {   	      		      	      
      unsigned int entryCount = 0;

      permute(perm, numberOfTrees, &seed);
      
      t = gettime();

      vect1 = (double *)calloc(h->entryCount, sizeof(double));
      vect2 = (double *)calloc(h->entryCount, sizeof(double));	     

        
      
      for(j = 0; j < h->tableSize; j++)
	{		
	  if(h->table[j] != NULL)
	    {		
	      entry *e = h->table[j];
	      
	      do
		{
		  unsigned int *set = e->treeVector;       
		  
		  for(l = 0; l < numberOfTrees; l++)
		    {			     
		      if((set[l / MASK_LENGTH] != 0) && (set[l / MASK_LENGTH] & mask32[l % MASK_LENGTH]))
			{
			  if(perm[l] % 2 == 0)
			    vect1[entryCount] = vect1[entryCount] + 1.0;
			  else			
			    vect2[entryCount] = vect2[entryCount] + 1.0;
			}
		    }
		  entryCount++;
		  e = e->next;
		}
	      while(e != NULL);
	    }			     
	}		    	
      
      
      
      
      assert(entryCount == h->entryCount);
      
      

      result = testFreq(vect1, vect2, entryCount);
	  
          
      
      if(result >= FC_LOWER)
	*countBetter = *countBetter + 1;
	      
      avg += result;
	      
      free(vect1);		  
      free(vect2);		 
    }

  free(perm);
	  
  avg /= BOOTSTOP_PERMUTATIONS;
	
      

  return avg;
}




static double wcCriterion(int numberOfTrees, hashtable *h, int *countBetter, double *wrf_thresh_avg, double *wrf_avg, tree *tr, int vectorLength)
{
  int 
    k, 
    l,   
    wrf,
    mr_thresh = ((double)numberOfTrees/4.0);
   
  unsigned int 
    *perm =  (unsigned int *)malloc(sizeof(unsigned int) * numberOfTrees),
    j;

  long seed = 12345;  
  
  double 
    wrf_thresh = 0.0,
    pct_avg = 0.0;

  assert(*countBetter == 0 && *wrf_thresh_avg == 0.0 && *wrf_avg == 0.0);
	   	  
  for(j = 0; j < (unsigned int)numberOfTrees; j++)
    perm[j] = j;
	  
  for(k = 0; k < BOOTSTOP_PERMUTATIONS; k++)
    {   	      		           
      int mcnt1 = 0;			  
      int mcnt2 = 0;
      unsigned int entryCount = 0;
      double halfOfConsideredBips = 0.0;

      entry ** sortedByKeyA;
      entry ** sortedByKeyB;
      int lenA, lenB;
      boolean ignoreCompatibilityP;

      int iA, iB;
      wrf = 0;
      	      
      permute(perm, numberOfTrees, &seed);      
    
      for(j = 0; j < h->tableSize; j++)
	{		
	  if(h->table[j] != NULL)
	    {
	      entry *e = h->table[j];

	        do
		  {
		    int cnt1 = 0;
		    int cnt2 = 0;

		    unsigned int *set = e->treeVector;

		    for(l = 0; l < numberOfTrees; l++)
		      {			     
			if((set[l / MASK_LENGTH] != 0) && (set[l / MASK_LENGTH] & mask32[l % MASK_LENGTH]))
			  {			    
			    if(perm[l] % 2 == 0)
			      cnt1++;
			    else			
			      cnt2++;
			  }			     
		      }
		    
		    switch(tr->bootStopCriterion)
		      {
		      case MR_STOP:
			if(cnt1 <= mr_thresh)			      
			  cnt1 = 0;
		       
			if(cnt2 <= mr_thresh)	    
			  cnt2 = 0;

			if(cnt1 > 0)			      
			  mcnt1++;
			
			if(cnt2 > 0)			      
			  mcnt2++;

			wrf += ((cnt1 > cnt2) ? cnt1 - cnt2 : cnt2 - cnt1);
			break;
		      case MRE_STOP:
		      case MRE_IGN_STOP:
			e->supportFromTreeset[0] = cnt1;
			e->supportFromTreeset[1] = cnt2;
			break;
		      default:
			assert(0);
		      }

		    entryCount++;
		    e = e->next;
		  }
		while(e != NULL);
	    }	  	  	  	
	}	
      
      assert(entryCount == h->entryCount);

      if((tr->bootStopCriterion == MRE_STOP) || (tr->bootStopCriterion == MRE_IGN_STOP))
	{
	
	  
	  if (tr->bootStopCriterion == MRE_IGN_STOP)
	    ignoreCompatibilityP = TRUE;
	  else
	    ignoreCompatibilityP = FALSE;
	    
	 
	 
	  mre(h, ignoreCompatibilityP, &sortedByKeyA, &lenA, 0, tr->mxtips, vectorLength, TRUE);
	  mre(h, ignoreCompatibilityP, &sortedByKeyB, &lenB, 1, tr->mxtips, vectorLength, TRUE);
	  
	   
	  mcnt1 = lenA;
	  mcnt2 = lenB;
	  
	  iA = iB = 0;
	  
	  while(iA < mcnt1 || iB < mcnt2)
	    {	    
	      if( iB == mcnt2 || (iA < mcnt1 && sortedByKeyA[iA] < sortedByKeyB[iB]) )
		{
		  wrf += sortedByKeyA[iA]->supportFromTreeset[0];
		  iA++;
		}
	      else
		{
		  if( iA == mcnt1 || (iB < mcnt2 && sortedByKeyB[iB] < sortedByKeyA[iA]) )
		    {
		      wrf += sortedByKeyB[iB]->supportFromTreeset[1];
		      iB++;
		    }
		  else
		    {
		      int cnt1, cnt2;
		      
		      assert (sortedByKeyA[iA] == sortedByKeyB[iB]);
		      
		      cnt1 = sortedByKeyA[iA]->supportFromTreeset[0];
		      cnt2 = sortedByKeyB[iB]->supportFromTreeset[1];
		      
		      wrf += ((cnt1 > cnt2) ? cnt1 - cnt2 : cnt2 - cnt1);
		      
		      iA++; 
		      iB++;
		    }
		}
	    }

	  free(sortedByKeyA);
	  free(sortedByKeyB);
	  
	  assert (iA == mcnt1);
	  assert (iB == mcnt2);	  
	}

      halfOfConsideredBips = ( ((((double)numberOfTrees/2.0) * (double)mcnt1)) + ((((double)numberOfTrees/2.0) * (double)mcnt2)) );

      /* 
	 wrf_thresh is the 'custom' threshold computed for this pair
	 of majority rules trees (i.e. one of the BS_PERMS splits),
	 and simply takes into account the resolution of the two trees
      */

      wrf_thresh = (tr->wcThreshold) * halfOfConsideredBips;    
      
      /*
	we count this random split as 'succeeding' when
	 the wrf between maj rules trees is exceeded
	 by its custom threshold
      */

      if((double)wrf <= wrf_thresh)			        
	*countBetter = *countBetter + 1;

      /* 
	 here we accumulate outcomes and thresholds, because
	 we're not going to stop until the avg dist is less
	 than the avg threshold
      */

      pct_avg += (double)wrf / halfOfConsideredBips  * 100.0;
      *wrf_avg += (double)wrf;
      *wrf_thresh_avg += wrf_thresh;
    }
 
  free(perm);

  pct_avg /= (double)BOOTSTOP_PERMUTATIONS; 
  *wrf_avg /= (double)BOOTSTOP_PERMUTATIONS; 
  *wrf_thresh_avg /= (double)BOOTSTOP_PERMUTATIONS;   

  /*printf("%d \t\t %f \t\t %d \t\t\t\t %f\n", numberOfTrees, *wrf_avg, *countBetter, *wrf_thresh_avg);	  	      */

  return pct_avg; 
}	  






void computeBootStopOnly(tree *tr, char *bootStrapFileName, analdef *adef)
{
  int numberOfTrees = 0, i;
  boolean stop = FALSE;
  double avg;
  int checkEvery;
  int treesAdded = 0;
  hashtable *h = initHashTable(tr->mxtips * FC_INIT * 10); 
  int 
    treeVectorLength, 
    vectorLength;
  unsigned int **bitVectors = initBitVector(tr, &vectorLength);
  
  

  double tt = 0.0, tc = 0.0;

  assert((FC_SPACING % 2 == 0) && (FC_THRESHOLD < BOOTSTOP_PERMUTATIONS));

  
  loadTreeFileToMemory(tr, bootStrapFileName, adef);
  numberOfTrees = tr->numberOfTrees;
  /*
    INFILE = myfopen(bootStrapFileName, "r");       
    numberOfTrees = countTrees(INFILE); 
  */
  
  printBothOpen("\n\nFound %d trees in File %s\n\n", numberOfTrees, bootStrapFileName);
  
  assert(sizeof(unsigned char) == 1);
  
  if(numberOfTrees % MASK_LENGTH == 0)
    treeVectorLength = numberOfTrees / MASK_LENGTH;
  else
    treeVectorLength = 1 + (numberOfTrees / MASK_LENGTH);  
 
  checkEvery = FC_SPACING;
        
  switch(tr->bootStopCriterion)
    {
    case FREQUENCY_STOP:
      printBothOpen("# Trees \t Average Pearson Coefficient \t # Permutations: pearson >= %f\n", 
		    FC_LOWER);
      break;
    case MR_STOP:
    case MRE_STOP:
    case MRE_IGN_STOP:
      printBothOpen("# Trees \t Avg WRF in %s \t # Perms: wrf <= %1.2f %s\n","%", 100.0 * tr->wcThreshold, "%");
      break;
    default:    
      assert(0);
    }
  
  for(i = 1; i <= numberOfTrees && !stop; i++)
    {                  
      int 
	bCount = 0,
	position = 0;
      
      double t = gettime();
     

      treeReadTopologyString(tr->treeStarts[i - 1], tr, FALSE, FALSE, &position, TRUE, adef); 
      assert(tr->mxtips == tr->ntips);
      
      bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vectorLength, h, (i - 1), BIPARTITIONS_BOOTSTOP, (branchInfo *)NULL,
			      &bCount, treeVectorLength, FALSE, FALSE);
      tt += gettime() - t;
      assert(bCount == tr->mxtips - 3);
                 
      treesAdded++;	
            
      if(i > START_BSTOP_TEST && i % checkEvery == 0)
	{ 
	  int countBetter = 0;
	  
	  t = gettime();
	  switch(tr->bootStopCriterion)
	    {
	    case FREQUENCY_STOP:
	      avg = frequencyCriterion(i, h, &countBetter);	  	  	  
	      printBothOpen("%d \t\t\t %f \t\t\t\t %d\n", i, avg, countBetter);
	  	  
	      stop = (countBetter >= FC_THRESHOLD && avg >= FC_LOWER);	  	 
	      break;
	    case MR_STOP:
	    case MRE_STOP:
	    case MRE_IGN_STOP:
	      {
		double 
		  wrf_thresh_avg = 0.0,
		  wrf_avg = 0.0;
		avg = wcCriterion(i, h, &countBetter, &wrf_thresh_avg, &wrf_avg, tr, vectorLength);
		printBothOpen("%d \t\t %1.2f \t\t\t %d\n", i, avg, countBetter);	       
		
		stop = (countBetter >= FC_THRESHOLD && wrf_avg <= wrf_thresh_avg);
	      }
	      break;
	    default:
	      assert(0);
	    }
	  tc += gettime() - t;
	}	 	   
      
    }
  
 

  if(stop)              
    printBothOpen("Converged after %d replicates\n", treesAdded);           
  else    
    printBothOpen("Bootstopping test did not converge after %d trees\n", treesAdded);     

  freeTreeFileInMemory(tr); 
  
  freeBitVectors(bitVectors, 2 * tr->mxtips);
  free(bitVectors);
  freeHashTable(h);
  free(h);

  

 
  exit(0);
}

boolean bootStop(tree *tr, hashtable *h, int numberOfTrees, double *pearsonAverage, unsigned int **bitVectors, int treeVectorLength, int vectorLength)
{
  int 
    n = numberOfTrees + 1,
    bCount = 0;

  assert((FC_SPACING % 2 == 0) && (FC_THRESHOLD < BOOTSTOP_PERMUTATIONS));
  assert(tr->mxtips == tr->rdta->numsp);

  bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vectorLength, h, numberOfTrees, BIPARTITIONS_BOOTSTOP, (branchInfo *)NULL,
			  &bCount, treeVectorLength, FALSE, FALSE);
  assert(bCount == tr->mxtips - 3); 

  if((n > START_BSTOP_TEST) && (n % FC_SPACING == 0))
    {     
      int countBetter = 0;

      switch(tr->bootStopCriterion)
	{
	case FREQUENCY_STOP:
	  *pearsonAverage = frequencyCriterion(n, h, &countBetter);	  	        	       

	  if(countBetter >= FC_THRESHOLD)
	    return TRUE;
	  else
	    return FALSE;
	  break;
	case MR_STOP:
	case MRE_STOP:
	case MRE_IGN_STOP:
	 {	   
	   double 
	     wrf_thresh_avg = 0.0,
	     wrf_avg = 0.0;
	   
	   *pearsonAverage = wcCriterion(n, h, &countBetter, &wrf_thresh_avg, &wrf_avg, tr, vectorLength);
	  
	   if(countBetter >= FC_THRESHOLD && wrf_avg <= wrf_thresh_avg)
	     return TRUE;
	   else
	     return FALSE;
	 } 
	default:
	  assert(0);
	}
    }
  else
    return FALSE;
}




/* consensus stuff */

static boolean compatible(entry* e1, entry* e2, int bvlen)
{
  int i;

  unsigned int 
    *A = e1->bitVector,
    *C = e2->bitVector;
  
  for(i = 0; i < bvlen; i++)        
    if(A[i] & C[i])
      break;
          
  if(i == bvlen)
    return TRUE;
  
  for(i = 0; i < bvlen; i++)
    if(A[i] & ~C[i])
      break;
   
  if(i == bvlen)  
    return TRUE;  
  
  for(i = 0; i < bvlen; i++)
    if(~A[i] & C[i])
      break;
  
  if(i == bvlen)
    return TRUE;  
  else
    return FALSE;
}



static int sortByWeight(const void *a, const void *b, int which)
{
  //recall, we want to sort descending, instead of ascending
  int 
    ca,
    cb;
  ca = ((*((entry **)a))->supportFromTreeset)[which];
  cb = ((*((entry **)b))->supportFromTreeset)[which];
  
  if (ca == cb) 
    return 0;
  
  return ((ca<cb)?1:-1);
}

static int sortByIndex(const void *a, const void *b)
{
  if ( (*((entry **)a)) == (*((entry **)b)) ) return 0;
  return (( (*((entry **)a)) < (*((entry **)b)) )?-1:1);
}

static int _sortByWeight0(const void *a, const void *b)
{
  return sortByWeight(a,b,0);
}

static int _sortByWeight1(const void *a, const void *b)
{
  return sortByWeight(a,b,1);
}

static void mre(hashtable *h, boolean icp, entry*** sbi, int* len, int which, int n, int vectorLength, boolean sortp)
{
  entry **sbw;
  unsigned int 
    i = 0,
    j = 0,
    k = 0;  
 
  sbw = (entry **) calloc(h->entryCount, sizeof(entry *));  

  for(i = 0; i < h->tableSize; i++)
    {		
      if(h->table[i] != NULL)
	{
	  entry *e = h->table[i];
	  do
	    {
	      sbw[j] = e;
	      j++;
	      e = e->next;
	    }
	  while(e != NULL);
	}
    }

  assert(j == h->entryCount);

  if(which == 0)    
    qsort(sbw, h->entryCount, sizeof(entry *), _sortByWeight0);      
  else    
    qsort(sbw, h->entryCount, sizeof(entry *), _sortByWeight1);      

  *sbi = (entry **)calloc(n - 3, sizeof(entry *));

  *len = 0;

  if(icp == FALSE)
    {     
      for(i = 0; i < h->entryCount && (*len) < n-3; i++)
	{	
	  boolean compatflag = TRUE;

	  assert(*len < n-3);		  
	
	  /*  for(k = 0; k < (unsigned int)(*len); k++)	  */
	  /*if(sbw[i]->supportFromTreeset[which] <= mr_thresh) */
	    for(k = ((unsigned int)(*len)); k > 0; k--)
	      {
		//k indexes sbi
		//j indexes sbw
		//need to compare the two
		
		if(!compatible((*sbi)[k-1], sbw[i], vectorLength))		
		  {
		    compatflag = FALSE;	    
		    break;
		  }
	      }
	  
	  if(compatflag)
	    {	      
	      (*sbi)[*len] = sbw[i];
	      (*len)++;
	    }         
	}
    }
  else 
    {      
      for(i = 0; i < (unsigned int)(n-3); i++)
	{
	  (*sbi)[i] = sbw[i];
	  (*len)++;
	}
    }

  free(sbw);

  if (sortp == TRUE)
    qsort(*sbi, (*len), sizeof(entry *), sortByIndex);    

 

  return;
}



//is A a subset of B?
static boolean issubset(unsigned int* bipA, unsigned int* bipB, int vectorLen)
{
  int i;

  for (i = 0; i < vectorLen; i++)    
    /*if((bipA[i] & bipB[i]) != bipA[i])    */
    /* code below seems to be slightly faster */
    if((bipA[i] & ~bipB[i]) != 0)
      return FALSE;
        
  return TRUE;
}



static void printBip(entry *curBip, entry **consensusBips, const int consensusBipLen, const int numtips, const int vectorLen, 
		     boolean *processed, tree *tr, FILE *outf, const int numberOfTrees, boolean topLevel)
{
  int
    branchLabel,
    i,
    j,    
    printed = 0;

  unsigned int *subBip = (unsigned int *)calloc(vectorLen, sizeof(unsigned int));

  double 
    support = 0.0;

  for(i = 0; i < consensusBipLen; i++)
    {
      if((!processed[i]) && issubset(consensusBips[i]->bitVector, curBip->bitVector, vectorLen))
	{
	  boolean processThisRound = TRUE;
	  
	  for (j = 0; j < consensusBipLen; j++)	    
	    if(j != i && !processed[j] && issubset(consensusBips[i]->bitVector, consensusBips[j]->bitVector, vectorLen))		
	      processThisRound = FALSE;			   
	  
	  if(processThisRound == TRUE)
	    {
	      processed[i] = TRUE;

	      for(j = 0; j < vectorLen; j++)
		subBip[j] |= consensusBips[i]->bitVector[j];
	      
	      if(printed == 0 && !topLevel)		
		fprintf(outf, "(");		
	      else		
		fprintf(outf, ",");
		
	      printBip(consensusBips[i], consensusBips, consensusBipLen, numtips, vectorLen, processed, tr, outf, numberOfTrees, FALSE);

	      printed += 1;
	    }
	}
    }	
  
  for(i = 0; i < numtips; i++)
    {
      if((((curBip->bitVector[i/MASK_LENGTH] & mask32[i%MASK_LENGTH]) > 0) && ((subBip[i/MASK_LENGTH] & mask32[i%MASK_LENGTH]) == 0) ) == TRUE)
	{
	  if(printed == 0 && !topLevel)	    
	    fprintf(outf,"(");	   
	  else	    
	    fprintf(outf,",");
	   
	  fprintf(outf,"%s", tr->nameList[i+1]);
	  printed += 1;
	}
    }

  free(subBip);

  support = ((double)(curBip->supportFromTreeset[0])) / ((double) (numberOfTrees));
  branchLabel = (int)(0.5 + support * 100.0);
  
  if(!topLevel)
    fprintf(outf,"):1.0[%d]", branchLabel);
}

void computeConsensusOnly(tree *tr, char *treeSetFileName, analdef *adef)
{        
  hashtable 
    *h = initHashTable(tr->mxtips * FC_INIT * 10); 

  hashNumberType
    entries = 0;
  
  int  
    numberOfTrees = 0, 
    i, 
    j, 
    l,
    treeVectorLength, 
    vectorLength,
    consensusBipsLen,
    mr_thresh;

  unsigned int 
    **bitVectors = initBitVector(tr, &vectorLength),
    *topBip;

  entry 
    topBipE,
    **consensusBips;
 
  boolean 
    *processed;

  char 
    consensusFileName[1024];
  
  FILE 
    *outf;  

 
  
  loadTreeFileToMemory(tr, treeSetFileName, adef);
  numberOfTrees = tr->numberOfTrees; 
  
  mr_thresh = ((double)numberOfTrees / 2.0);  
  
  assert(sizeof(unsigned char) == 1);
  
  if(numberOfTrees % MASK_LENGTH == 0)
    treeVectorLength = numberOfTrees / MASK_LENGTH;
  else
    treeVectorLength = 1 + (numberOfTrees / MASK_LENGTH);  

  for(i = 1; i <= numberOfTrees; i++)
    {                  
      int 
	bCount = 0,
	position = 0;
      
      treeReadTopologyString(tr->treeStarts[i - 1], tr, FALSE, FALSE, &position, TRUE, adef);               
      
      assert(tr->mxtips == tr->ntips);
      
      bitVectorInitravSpecial(bitVectors, tr->nodep[1]->back, tr->mxtips, vectorLength, h, (i - 1), BIPARTITIONS_BOOTSTOP, (branchInfo *)NULL,
			      &bCount, treeVectorLength, FALSE, FALSE);
      
      assert(bCount == tr->mxtips - 3);                     
    }

 

  if(tr->consensusType == MR_CONSENSUS || tr->consensusType == STRICT_CONSENSUS)
    {
      consensusBips = (entry **)calloc(tr->mxtips - 3, sizeof(entry *));
      consensusBipsLen = 0;
    }

  for(j = 0; j < (int)h->tableSize; j++)
    {		
      if(h->table[j] != NULL)
	{
	  entry *e = h->table[j];
	  
	  do
	    {	
	      int cnt = 0;
	      
	      unsigned int *set = e->treeVector;
	      
	      for(l = 0; l < numberOfTrees; l++)					     
		if((set[l / MASK_LENGTH] != 0) && (set[l / MASK_LENGTH] & mask32[l % MASK_LENGTH]))		    			    
		  cnt++;		    		     		
	      
	      if(tr->consensusType == MR_CONSENSUS)
		{
		  if(cnt > mr_thresh)			      
		    {
		      consensusBips[consensusBipsLen] = e;
		      consensusBipsLen++;
		    }
		}

	      if(tr->consensusType == STRICT_CONSENSUS)
		{
		  if(cnt == numberOfTrees)
		    {
		      consensusBips[consensusBipsLen] = e;
		      consensusBipsLen++;
		    }
		}

	      e->supportFromTreeset[0] = cnt;
	      e = e->next;
	      entries++;
	    }
	  while(e != NULL);		
	}	  	        
    }	

  freeTreeFileInMemory(tr); 
  assert(entries == h->entryCount);
  
  if(tr->consensusType == MR_CONSENSUS || tr->consensusType == STRICT_CONSENSUS)
    assert(consensusBipsLen <= (tr->mxtips - 3));

  if(tr->consensusType == MRE_CONSENSUS)    
    mre(h, FALSE, &consensusBips, &consensusBipsLen, 0, tr->mxtips, vectorLength, FALSE);    

 

  processed = (boolean *) calloc(consensusBipsLen, sizeof(boolean));

  topBip = (unsigned int *) calloc(vectorLength, sizeof(unsigned int));
  
  for(i = 1; i < tr->mxtips; i++)
    topBip[i / MASK_LENGTH] |= mask32[i % MASK_LENGTH];  

  topBipE.bitVector = topBip;
  topBipE.supportFromTreeset[0] = numberOfTrees;

  strcpy(consensusFileName,         workdir);  
  
  switch(tr->consensusType)
    {
    case MR_CONSENSUS:
      strcat(consensusFileName,         "RAxML_MajorityRuleConsensusTree.");
      break;
    case MRE_CONSENSUS:
      strcat(consensusFileName,         "RAxML_MajorityRuleExtendedConsensusTree.");
      break;
    case STRICT_CONSENSUS:
      strcat(consensusFileName,         "RAxML_StrictConsensusTree.");
      break;
    default:
      assert(0);
    }
  
  strcat(consensusFileName,         run_id);

  outf = myfopen(consensusFileName, "w");

  fprintf(outf, "(%s", tr->nameList[1]);
  printBip(&topBipE, consensusBips, consensusBipsLen, tr->mxtips, vectorLength, processed, tr, outf, numberOfTrees, TRUE);  
  fprintf(outf, ");\n");

  

  fclose(outf);
  
  switch(tr->consensusType)
    {
    case MR_CONSENSUS:
      printBothOpen("RAxML Majority Rule consensus tree written to file: %s\n", consensusFileName);
      break;
    case MRE_CONSENSUS:
      printBothOpen("RAxML extended Majority Rule consensus tree written to file: %s\n", consensusFileName);
      break;
    case STRICT_CONSENSUS:
      printBothOpen("RAxML strict consensus tree written to file: %s\n", consensusFileName);
      break;
    default:
      assert(0);
    }
  
  
  free(topBip);
  free(processed);

  freeBitVectors(bitVectors, 2 * tr->mxtips);
  free(bitVectors);
  freeHashTable(h);
  free(h);
  free(consensusBips); 

  
 
  /*fclose(INFILE);*/

  
  exit(0);   
}
