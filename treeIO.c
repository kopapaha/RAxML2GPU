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

#include "axml.h"

extern FILE *INFILE;
extern char infoFileName[1024];
extern char tree_file[1024];
extern char *likelihood_key;
extern char *ntaxa_key;
extern char *smoothed_key;
extern int partCount;
extern double masterTime;





stringHashtable *initStringHashTable(hashNumberType n)
{
  /* 
     init with primes 
  */
    
  static const hashNumberType initTable[] = {53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317,
					     196613, 393241, 786433, 1572869, 3145739, 6291469, 12582917, 25165843,
					     50331653, 100663319, 201326611, 402653189, 805306457, 1610612741};
 

  /* init with powers of two

  static const  hashNumberType initTable[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
					      32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
					      4194304, 8388608, 16777216, 33554432, 67108864, 134217728,
					      268435456, 536870912, 1073741824, 2147483648U};
  */
  
  stringHashtable *h = (stringHashtable*)malloc(sizeof(stringHashtable));
  
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

  h->table = (stringEntry**)calloc(tableSize, sizeof(stringEntry*));
  h->tableSize = tableSize;    

  return h;
}


static hashNumberType  hashString(char *p, hashNumberType tableSize)
{
  hashNumberType h = 0;
  
  for(; *p; p++)
    h = 31 * h + *p;
  
  return (h % tableSize);
}

 

void addword(char *s, stringHashtable *h, int nodeNumber)
{
  hashNumberType position = hashString(s, h->tableSize);
  stringEntry *p = h->table[position];
  
  for(; p!= NULL; p = p->next)
    {
      if(strcmp(s, p->word) == 0)		 
	return;	  	
    }

  p = (stringEntry *)malloc(sizeof(stringEntry));

  assert(p);
  
  p->nodeNumber = nodeNumber;
  p->word = (char *)malloc((strlen(s) + 1) * sizeof(char));

  strcpy(p->word, s);
  
  p->next =  h->table[position];
  
  h->table[position] = p;
}

static int lookupWord(char *s, stringHashtable *h)
{
  hashNumberType position = hashString(s, h->tableSize);
  stringEntry *p = h->table[position];
  
  for(; p!= NULL; p = p->next)
    {
      if(strcmp(s, p->word) == 0)		 
	return p->nodeNumber;	  	
    }

  return -1;
}


static int countTips(nodeptr p, int numsp)
{
  if(isTip(p->number, numsp))  
    return 1;    
  {
    nodeptr q;
    int tips = 0;

    q = p->next;
    while(q != p)
      { 
	tips += countTips(q->back, numsp);
	q = q->next;
      } 
    
    return tips;
  }
}



static char *Tree2StringREC(char *treestr, tree *tr, nodeptr p, boolean printBranchLengths, boolean printNames, 
			    boolean printLikelihood, boolean rellTree, boolean finalPrint, int perGene, boolean branchLabelSupport)
{
  double  x, z;
  char  *nameptr;            
      
  if(isTip(p->number, tr->rdta->numsp)) 
    {	       	  
      if(printNames)
	{
	  nameptr = tr->nameList[p->number];     
	  sprintf(treestr, "%s", nameptr);
	}
      else
	sprintf(treestr, "%d", p->number);    
	
      while (*treestr) treestr++;
    }
  else 
    {                 	 
      *treestr++ = '(';
      treestr = Tree2StringREC(treestr, tr, p->next->back, printBranchLengths, printNames, printLikelihood, rellTree, 
			       finalPrint, perGene, branchLabelSupport);
      *treestr++ = ',';
      treestr = Tree2StringREC(treestr, tr, p->next->next->back, printBranchLengths, printNames, printLikelihood, rellTree, 
			       finalPrint, perGene, branchLabelSupport);
      if(p == tr->start->back) 
	{
	  *treestr++ = ',';
	  treestr = Tree2StringREC(treestr, tr, p->back, printBranchLengths, printNames, printLikelihood, rellTree, 
				   finalPrint, perGene, branchLabelSupport);
	}
      *treestr++ = ')';                    
    }

  if(p == tr->start->back) 
    {	      	 
      if(printBranchLengths && !rellTree)
	sprintf(treestr, ":0.0;\n");
      else
	sprintf(treestr, ";\n");	 	  	
    }
  else 
    {                   
      if(rellTree ||  branchLabelSupport)
	{	 	 
	  if(( !isTip(p->number, tr->rdta->numsp)) && 
	     ( !isTip(p->back->number, tr->rdta->numsp)))
	    {			      
	      assert(p->bInf != (branchInfo *)NULL);
	      if(rellTree)
		sprintf(treestr, "%d:%8.20f", p->bInf->support, p->z[0]);
	      else
		sprintf(treestr, ":%8.20f[%d]", p->z[0], p->bInf->support);
	    }
	  else		
	    sprintf(treestr, ":%8.20f", p->z[0]);	
	}
      else
	{
	  if(printBranchLengths)
	    {
	      assert(perGene != NO_BRANCHES);
	      
	      if(!tr->multiBranch)
		{
		  assert(tr->fracchange != -1.0);
		  z = p->z[0];
		  if (z < zmin) z = zmin;      	 
		  x = -log(z) * tr->fracchange;
		  sprintf(treestr, ":%8.20f", x);	
		}
	      else
		{
		  if(perGene == SUMMARIZE_LH)
		    {
		      int i;
		      double avgX = 0;
		      
		      for(i = 0; i < tr->numBranches; i++)
			{
			  assert(tr->partitionContributions[i] != -1.0);
			  assert(tr->fracchanges[i] != -1.0);
			  z = p->z[i];
			  if (z < zmin) z = zmin;      	 
			  x = -log(z) * tr->fracchanges[i];
			  avgX += x * tr->partitionContributions[i];
			}
		      sprintf(treestr, ":%8.20f", avgX);
		    }
		  else
		    {	
		      assert(tr->fracchanges[perGene] != -1.0);
		      assert(perGene >= 0 && perGene < tr->numBranches);
		      z = p->z[perGene];
		      if (z < zmin) z = zmin;      	 
		      x = -log(z) * tr->fracchanges[perGene];
		      sprintf(treestr, ":%8.20f", x);
		    }
		}
	    }
	  else
	    sprintf(treestr, "%s", "\0");	    
	}      
    }
  
  while (*treestr) treestr++;
  return  treestr;
}


static void collectSubtrees(tree *tr, nodeptr *subtrees, int *count, int ogn)
{
  int i;
  for(i = tr->mxtips + 1; i <= tr->mxtips + tr->mxtips - 2; i++)
    {
      nodeptr p, q;
      p = tr->nodep[i];
      if(countTips(p, tr->rdta->numsp) == ogn)
	{
	  subtrees[*count] = p;
	  *count = *count + 1;
	}
      q = p->next;
      while(q != p)
	{
	  if(countTips(q, tr->rdta->numsp) == ogn)
	    {
	      subtrees[*count] = q;
	      *count = *count + 1;
	    }
	  q = q->next;
	}
    }
}

static void checkOM(nodeptr p, int *n, int *c, tree *tr)
{
  if(isTip(p->number, tr->rdta->numsp))
    {
      n[*c] = p->number;
      *c = *c + 1;     
    }
  else
    {
      nodeptr q = p->next;

      while(q != p)
	{
	  checkOM(q->back, n, c, tr);
	  q = q->next;
	}
    }
}
    
static char *rootedTreeREC(char *treestr, tree *tr, nodeptr p, boolean printBranchLengths, boolean printNames, 
			   boolean printLikelihood, boolean rellTree, 
			   boolean finalPrint, analdef *adef, int perGene, boolean branchLabelSupport)
{
  double  x, z;
  char  *nameptr;

  if(isTip(p->number, tr->rdta->numsp)) 
    {	     
      if(printNames)
	{
	  nameptr = tr->nameList[p->number];     
	  sprintf(treestr, "%s", nameptr);
	}
      else
	sprintf(treestr, "%d", p->number);
      
      while (*treestr) treestr++;
    }
  else 
    {
      *treestr++ = '(';
      treestr = rootedTreeREC(treestr, tr, p->next->back, printBranchLengths, printNames, printLikelihood, 
			      rellTree, finalPrint, adef, perGene, branchLabelSupport);
      *treestr++ = ',';
      treestr = rootedTreeREC(treestr, tr, p->next->next->back, printBranchLengths, printNames, printLikelihood, 
			      rellTree, finalPrint, adef, perGene, branchLabelSupport);      
      *treestr++ = ')';            
      }

  if(rellTree ||  branchLabelSupport)
    {           
      if(!isTip(p->number, tr->rdta->numsp) && !isTip(p->back->number, tr->rdta->numsp))
	{	  
	  assert(p->bInf != (branchInfo *)NULL);
	  if(rellTree)		
	    sprintf(treestr, "%d:%8.20f", p->bInf->support, p->z[0]);		
	  else
	    sprintf(treestr, ":%8.20f[%d]", p->z[0], p->bInf->support);
	}
      else     
	sprintf(treestr, ":%8.20f", p->z[0]);	
    }
  else
    {
      if(printBranchLengths)
	{	  
	  assert(perGene != NO_BRANCHES);

	  if(!tr->multiBranch)
	    {
	      assert(tr->fracchange != -1.0);
	      z = p->z[0];
	      if (z < zmin) z = zmin;      	 
	      x = -log(z) * tr->fracchange;
	      sprintf(treestr, ":%8.20f", x);	
	    }
	  else
	    {
	      if(perGene == SUMMARIZE_LH)
		{
		  int i;
		  double avgX = 0;
		  
		  for(i = 0; i < tr->numBranches; i++)
		    {
		      assert(tr->fracchanges[i] != -1.0);
		      assert(tr->partitionContributions[i] != -1.0);
		      z = p->z[i];
		      if (z < zmin) z = zmin;      	 
		      x = -log(z) * tr->fracchanges[i];
		      avgX += x * tr->partitionContributions[i];
		    }
		  sprintf(treestr, ":%8.20f", avgX);
		}
	      else
		{
		  assert(tr->fracchanges[perGene] != -1.0);
		  assert(perGene >= 0 && perGene < tr->numBranches);
		  z = p->z[perGene];
		  if (z < zmin) z = zmin;      	 
		  x = -log(z) * tr->fracchanges[perGene];
		  sprintf(treestr, ":%8.20f", x);
		}
	    }	  
	}
      else
	sprintf(treestr, "%s", "\0");
    }

    while (*treestr) treestr++;
    return  treestr;

}

static char *rootedTree(char *treestr, tree *tr, nodeptr p, boolean printBranchLengths, boolean printNames, 
			boolean printLikelihood, boolean rellTree, 
			boolean finalPrint, analdef *adef, int perGene, boolean branchLabelSupport)
{
  double oldz[NUM_BRANCHES];
  int i;
  
  for(i = 0; i < tr->numBranches; i++)
    oldz[i] = p->z[i];

  if(rellTree)    
    {    
      p->z[0] = p->back->z[0] = oldz[0] * 0.5;
      /*printf("%f\n",  p->z[0]);*/
    }
  else
    {
      if(printBranchLengths)
	{
	  double rz, z;
	  assert(perGene != NO_BRANCHES);

	  if(!tr->multiBranch)
	    {
	      assert(tr->fracchange != -1.0);
	      z = -log(p->z[0]) * tr->fracchange;
	      rz = exp(-(z * 0.5)/ tr->fracchange);
	      p->z[0] = p->back->z[0] = rz;
	    }
	  else
	    {
	      if(perGene == SUMMARIZE_LH)
		{				  		
		  int i;	      
		  
		  for(i = 0; i < tr->numBranches; i++)
		    {	
		      assert(tr->fracchanges[i] != -1.0);
		      z    = -log(p->z[i]) * tr->fracchanges[i];	    	      
		      rz   = exp(-(z * 0.5)/ tr->fracchanges[i]);
		      p->z[i] = p->back->z[i] = rz;		    
		    }		 
		}	     	     
	      else
		{		
		  assert(tr->fracchanges[perGene] != -1.0);
		  assert(perGene >= 0 && perGene < tr->numBranches);
		  z = -log(p->z[perGene]) * tr->fracchanges[perGene];
		  rz = exp(-(z * 0.5)/ tr->fracchanges[perGene]);
		  p->z[perGene] = p->back->z[perGene] = rz;	       	      	      
		}
	    }
	}
    }

  *treestr = '(';
  treestr++;
  treestr = rootedTreeREC(treestr, tr, p,  printBranchLengths, printNames, printLikelihood, rellTree, finalPrint, adef, perGene, branchLabelSupport);
  *treestr = ',';
  treestr++;
  treestr = rootedTreeREC(treestr, tr, p->back,  printBranchLengths, printNames, printLikelihood, rellTree, finalPrint, adef, perGene, branchLabelSupport);
  sprintf(treestr, ");\n");
  while (*treestr) treestr++;


  for(i = 0; i < tr->numBranches; i++)
    p->z[i] = p->back->z[i] = oldz[i];  
    
  return  treestr;
}



char *Tree2String(char *treestr, tree *tr, nodeptr p, boolean printBranchLengths, boolean printNames, boolean printLikelihood, 
		  boolean rellTree, 
		  boolean finalPrint, analdef *adef, int perGene, boolean branchLabelSupport)
{ 
  if(finalPrint && adef->outgroup)
    {
      nodeptr startNode = tr->start;

      if(tr->numberOfOutgroups > 1)
	{
	  nodeptr root;
	  nodeptr *subtrees = (nodeptr *)malloc(sizeof(nodeptr) * tr->mxtips);
	  int i, k, count = 0;
	  int *nodeNumbers = (int*)malloc(sizeof(int) * tr->numberOfOutgroups);
	  int *foundVector = (int*)malloc(sizeof(int) * tr->numberOfOutgroups);
	  boolean monophyletic = FALSE;

	  collectSubtrees(tr, subtrees, &count, tr->numberOfOutgroups);

	  /*printf("Found %d subtrees of size  %d\n", count, tr->numberOfOutgroups);*/
	  
	  for(i = 0; (i < count) && (!monophyletic); i++)
	    {
	      int l, sum, nc = 0;
	      for(k = 0; k <  tr->numberOfOutgroups; k++)
		{
		  nodeNumbers[k] = -1;
		  foundVector[k] = 0;
		}

	      checkOM(subtrees[i], nodeNumbers, &nc, tr);	      
	      
	      for(l = 0; l < tr->numberOfOutgroups; l++)
		for(k = 0; k < tr->numberOfOutgroups; k++)
		  {
		    if(nodeNumbers[l] == tr->outgroupNums[k])
		      foundVector[l] = 1;
		  }
	      
	      sum = 0;
	      for(l = 0; l < tr->numberOfOutgroups; l++)
		sum += foundVector[l];
	      
	      if(sum == tr->numberOfOutgroups)
		{	       		  
		  root = subtrees[i];
		  tr->start = root;		
		  /*printf("outgroups are monphyletic!\n");*/
		  monophyletic = TRUE;		  
		}
	      else
		{
		  if(sum > 0)
		    {
		      /*printf("outgroups are NOT monophyletic!\n");*/
		      monophyletic = FALSE;
		    }	     
		}	
	    }
	  
	  if(!monophyletic)
	    {
	      printf("WARNING, outgroups are not monophyletic, using first outgroup \"%s\"\n", tr->nameList[tr->outgroupNums[0]]);
	      printf("from the list to root the tree!\n");
	     
#ifndef PARALLEL
	      {
		FILE *infoFile = myfopen(infoFileName, "a");

		fprintf(infoFile, "\nWARNING, outgroups are not monophyletic, using first outgroup \"%s\"\n", tr->nameList[tr->outgroupNums[0]]);
		fprintf(infoFile, "from the list to root the tree!\n");
		
		fclose(infoFile);
	      }
#endif 

	      tr->start = tr->nodep[tr->outgroupNums[0]];
	      
	      rootedTree(treestr, tr, tr->start->back, printBranchLengths, printNames, printLikelihood, rellTree, finalPrint, adef, perGene, branchLabelSupport);
	    }
	  else
	    {	     
	      if(isTip(tr->start->number, tr->rdta->numsp))
		{
		  printf("Outgroup-Monophyly ERROR; tr->start is a tip \n");
		  errorExit(-1);
		}
	      if(isTip(tr->start->back->number, tr->rdta->numsp))
	      	{
		  printf("Outgroup-Monophyly ERROR; tr->start is a tip \n");
		  errorExit(-1);
		}
	      	      
	      rootedTree(treestr, tr, tr->start->back, printBranchLengths, printNames, printLikelihood, rellTree, finalPrint, adef, perGene, branchLabelSupport);
	    }
	  
	  free(foundVector);
	  free(nodeNumbers);
	  free(subtrees);
	}
      else
	{	  
	  tr->start = tr->nodep[tr->outgroupNums[0]];	 
	  rootedTree(treestr, tr, tr->start->back, printBranchLengths, printNames, printLikelihood, rellTree, finalPrint, adef, perGene, branchLabelSupport);
	}      

      tr->start = startNode;
    }
  else
    {
      Tree2StringREC(treestr, tr, p, printBranchLengths, printNames, printLikelihood, rellTree, finalPrint, perGene, branchLabelSupport);  
    }
  while (*treestr) treestr++;
  return treestr;
}


void printTreePerGene(tree *tr, analdef *adef, char *fileName, char *permission)
{  
  FILE *treeFile;
  char extendedTreeFileName[1024];
  char buf[16];
  int i;

  assert(adef->perGeneBranchLengths);
     
  for(i = 0; i < tr->numBranches; i++)	
    {
      strcpy(extendedTreeFileName, fileName);
      sprintf(buf,"%d", i);
      strcat(extendedTreeFileName, ".PARTITION.");
      strcat(extendedTreeFileName, buf);
      /*printf("Partitiuon %d file %s\n", i, extendedTreeFileName);*/
      Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, TRUE, adef, i, FALSE);
      treeFile = myfopen(extendedTreeFileName, permission);
      fprintf(treeFile, "%s", tr->tree_string);
      fclose(treeFile);
    }  
    
}



/*=======================================================================*/
/*                         Read a tree from a file                       */
/*=======================================================================*/


/*  1.0.A  Processing of quotation marks in comment removed
 */

static int treeFinishCom (FILE *fp, char **strp)
{
  int  ch;
  
  while ((ch = getc(fp)) != EOF && ch != ']') {
    if (strp != NULL) *(*strp)++ = ch;    /* save character  */
    if (ch == '[') {                      /* nested comment; find its end */
      if ((ch = treeFinishCom(fp, strp)) == EOF)  break;
      if (strp != NULL) *(*strp)++ = ch;  /* save closing ]  */
    }
  }
  
  if (strp != NULL) **strp = '\0';        /* terminate string  */
  return  ch;
} /* treeFinishCom */


static int treeGetCh (FILE *fp)         /* get next nonblank, noncomment character */
{ /* treeGetCh */
  int  ch;

  while ((ch = getc(fp)) != EOF) {
    if (whitechar(ch)) ;
    else if (ch == '[') {                   /* comment; find its end */
      if ((ch = treeFinishCom(fp, (char **) NULL)) == EOF)  break;
    }
    else  break;
  }
  
  return  ch;
} /* treeGetCh */


static boolean treeLabelEnd (int ch)
{
  switch (ch) 
    {
    case EOF:  
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
      return TRUE;
    default:
      break;
    }
  return FALSE;
} 


static boolean  treeGetLabel (FILE *fp, char *lblPtr, int maxlen)
{
  int      ch;
  boolean  done, quoted, lblfound;

  if (--maxlen < 0) 
    lblPtr = (char *) NULL; 
  else 
    if (lblPtr == NULL) 
      maxlen = 0;

  ch = getc(fp);
  done = treeLabelEnd(ch);

  lblfound = ! done;
  quoted = (ch == '\'');
  if (quoted && ! done) 
    {
      ch = getc(fp); 
      done = (ch == EOF);
    }

  while (! done) 
    {
      if (quoted) 
	{
	  if (ch == '\'') 
	    {
	      ch = getc(fp); 
	      if (ch != '\'') 
		break;
	    }
        }
      else 
	if (treeLabelEnd(ch)) break;     

      if (--maxlen >= 0) *lblPtr++ = ch;
      ch = getc(fp);
      if (ch == EOF) break;
    }

  if (ch != EOF)  (void) ungetc(ch, fp);

  if (lblPtr != NULL) *lblPtr = '\0';

  return lblfound;
}


static boolean  treeFlushLabel (FILE *fp)
{ 
  return  treeGetLabel(fp, (char *) NULL, (int) 0);
} 




static int treeFindTipByLabelString(char  *str, tree *tr)                    
{
  int lookup = lookupWord(str, tr->nameHash);

  if(lookup > 0)
    {
      assert(! tr->nodep[lookup]->back);
      return lookup;
    }
  else
    { 
      printf("ERROR: Cannot find tree species: %s\n", str);
      return  0;
    }
}


static int treeFindTipName(FILE *fp, tree *tr)
{
  char    str[nmlngth+2];
  int      n;

  if(treeGetLabel(fp, str, nmlngth+2))
    n = treeFindTipByLabelString(str, tr);
  else
    n = 0;
   

  return  n;
} 



static void  treeEchoContext (FILE *fp1, FILE *fp2, int n)
{ /* treeEchoContext */
  int      ch;
  boolean  waswhite;
  
  waswhite = TRUE;
  
  while (n > 0 && ((ch = getc(fp1)) != EOF)) {
    if (whitechar(ch)) {
      ch = waswhite ? '\0' : ' ';
      waswhite = TRUE;
    }
    else {
      waswhite = FALSE;
    }
    
    if (ch > '\0') {putc(ch, fp2); n--;}
  }
} /* treeEchoContext */


static boolean treeProcessLength (FILE *fp, double *dptr)
{
  int  ch;
  
  if ((ch = treeGetCh(fp)) == EOF)  return FALSE;    /*  Skip comments */
  (void) ungetc(ch, fp);
  
  if (fscanf(fp, "%lf", dptr) != 1) {
    printf("ERROR: treeProcessLength: Problem reading branch length\n");
    treeEchoContext(fp, stdout, 40);
    printf("\n");
    return  FALSE;
  }
  
  return  TRUE;
}


static int treeFlushLen (FILE  *fp)
{
  double  dummy;  
  int     ch;
  
  ch = treeGetCh(fp);
  
  if (ch == ':') 
    {
      ch = treeGetCh(fp);
      
      ungetc(ch, fp);
      if(!treeProcessLength(fp, & dummy)) return 0;
      return 1;	  
    }
  
  
  
  if (ch != EOF) (void) ungetc(ch, fp);
  return 1;
} 





static boolean treeNeedCh (FILE *fp, int c1, char *where)
{
  int  c2;
  
  if ((c2 = treeGetCh(fp)) == c1)  return TRUE;
  
  printf("ERROR: Expecting '%c' %s tree; found:", c1, where);
  if (c2 == EOF) 
    {
      printf("End-of-File");
    }
  else 
    {      	
      ungetc(c2, fp);
      treeEchoContext(fp, stdout, 40);
    }
  putchar('\n');

  if(c1 == ':')    
    printf("RAxML may be expecting to read a tree that contains branch lengths\n");

  return FALSE;
} 



static boolean addElementLen (FILE *fp, tree *tr, nodeptr p, boolean readBranchLengths, boolean readNodeLabels, int *lcount)
{   
  nodeptr  q;
  int      n, ch, fres;
  
  if ((ch = treeGetCh(fp)) == '(') 
    { 
      n = (tr->nextnode)++;
      if (n > 2*(tr->mxtips) - 2) 
	{
	  if (tr->rooted || n > 2*(tr->mxtips) - 1) 
	    {
	      printf("ERROR: Too many internal nodes.  Is tree rooted?\n");
	      printf("       Deepest splitting should be a trifurcation.\n");
	      return FALSE;
	    }
	  else 
	    {
	      assert(!readNodeLabels);
	      tr->rooted = TRUE;
	    }
	}
      
      q = tr->nodep[n];

      if (! addElementLen(fp, tr, q->next, readBranchLengths, readNodeLabels, lcount))        return FALSE;
      if (! treeNeedCh(fp, ',', "in"))             return FALSE;
      if (! addElementLen(fp, tr, q->next->next, readBranchLengths, readNodeLabels, lcount))  return FALSE;
      if (! treeNeedCh(fp, ')', "in"))             return FALSE;
      
      if(readNodeLabels)
	{
	  char label[64];
	  int support;

	  if(treeGetLabel (fp, label, 10))
	    {	
	      int val = sscanf(label, "%d", &support);
      
	      assert(val == 1);

	      /*printf("LABEL %s Number %d\n", label, support);*/
	      p->support = q->support = support;
	      /*printf("%d %d %d %d\n", p->support, q->support, p->number, q->number);*/
	      assert(p->number > tr->mxtips && q->number > tr->mxtips);
	      *lcount = *lcount + 1;
	    }
	}
      else	
	(void) treeFlushLabel(fp);
    }
  else 
    {   
      ungetc(ch, fp);
      if ((n = treeFindTipName(fp, tr)) <= 0)          return FALSE;
      q = tr->nodep[n];
      if (tr->start->number > n)  tr->start = q;
      (tr->ntips)++;
    }
  
  if(readBranchLengths)
    {
      double branch;
      if (! treeNeedCh(fp, ':', "in"))                 return FALSE;
      if (! treeProcessLength(fp, &branch))            return FALSE;
      
      /*printf("Branch %8.20f %d\n", branch, tr->numBranches);*/
      hookup(p, q, &branch, tr->numBranches);
    }
  else
    {
      fres = treeFlushLen(fp);
      if(!fres) return FALSE;
      
      hookupDefault(p, q, tr->numBranches);
    }
  return TRUE;          
} 











static nodeptr uprootTree (tree *tr, nodeptr p, boolean readBranchLengths, boolean readConstraint)
{
  nodeptr  q, r, s, start;
  int      n, i;              

  for(i = tr->mxtips + 1; i < 2 * tr->mxtips - 1; i++)
    assert(i == tr->nodep[i]->number);

  
  if(isTip(p->number, tr->mxtips) || p->back) 
    {
      printf("ERROR: Unable to uproot tree.\n");
      printf("       Inappropriate node marked for removal.\n");
      assert(0);
    }
  
  assert(p->back == (nodeptr)NULL);
  
  tr->nextnode = tr->nextnode - 1;

  assert(tr->nextnode < 2 * tr->mxtips);
  
  n = tr->nextnode;               
  
  assert(tr->nodep[tr->nextnode]);

  if (n != tr->mxtips + tr->ntips - 1) 
    {
      printf("ERROR: Unable to uproot tree.  Inconsistent\n");
      printf("       number of tips and nodes for rooted tree.\n");
      assert(0);
    }

  q = p->next->back;                  /* remove p from tree */
  r = p->next->next->back;
  assert(p->back == (nodeptr)NULL);
    
  if(readBranchLengths)
    {
      double b[NUM_BRANCHES];
      int i;
      for(i = 0; i < tr->numBranches; i++)
	b[i] = (r->z[i] + q->z[i]);
      hookup (q, r, b, tr->numBranches);
    }
  else    
    hookupDefault(q, r, tr->numBranches);    

  if(readConstraint && tr->grouped)
    {    
      if(tr->constraintVector[p->number] != 0)
	{
	  printf("Root node to remove should have top-level grouping of 0\n");
	  assert(0);
	}
    }  
 
  assert(!(isTip(r->number, tr->mxtips) && isTip(q->number, tr->mxtips))); 

  assert(p->number > tr->mxtips);

  if(tr->ntips > 2 && p->number != n) 
    {    	
      q = tr->nodep[n];            /* transfer last node's conections to p */
      r = q->next;
      s = q->next->next;
      
      if(readConstraint && tr->grouped)	
	tr->constraintVector[p->number] = tr->constraintVector[q->number];       
      
      hookup(p,             q->back, q->z, tr->numBranches);   /* move connections to p */
      hookup(p->next,       r->back, r->z, tr->numBranches);
      hookup(p->next->next, s->back, s->z, tr->numBranches);           
      
      q->back = q->next->back = q->next->next->back = (nodeptr) NULL;
    }
  else    
    p->back = p->next->back = p->next->next->back = (nodeptr) NULL;
  
  assert(tr->ntips > 2);
  
  start = findAnyTip(tr->nodep[tr->mxtips + 1], tr->mxtips);
  
  assert(isTip(start->number, tr->mxtips));
  tr->rooted = FALSE;
  return  start;
}


boolean treeReadLen (FILE *fp, tree *tr, analdef *adef)
{
  nodeptr  p;
  int      i, ch, dummy; 

  for (i = 1; i <= tr->mxtips; i++) 
    tr->nodep[i]->back = (node *) NULL;

 

  for(i = tr->mxtips + 1; i < 2 * tr->mxtips; i++)
    {
      tr->nodep[i]->back = (nodeptr)NULL;
      tr->nodep[i]->next->back = (nodeptr)NULL;
      tr->nodep[i]->next->next->back = (nodeptr)NULL;
      tr->nodep[i]->number = i;
      tr->nodep[i]->next->number = i;
      tr->nodep[i]->next->next->number = i;
    }
    
  
    
  tr->start       = tr->nodep[1];
  tr->ntips       = 0;
  tr->nextnode    = tr->mxtips + 1;      
 
  for(i = 0; i < tr->numBranches; i++)
    tr->partitionSmoothed[i] = FALSE;
  
  tr->rooted      = FALSE;     

  p = tr->nodep[(tr->nextnode)++]; 
  
  while((ch = treeGetCh(fp)) != '(');
      

  
       
  if (! addElementLen(fp, tr, p, FALSE, FALSE, &dummy))                 return FALSE;
  if (! treeNeedCh(fp, ',', "in"))                return FALSE;
  if (! addElementLen(fp, tr, p->next, FALSE, FALSE, &dummy))           return FALSE;
  if (! tr->rooted) 
    {
      if ((ch = treeGetCh(fp)) == ',') 
	{ 
	  if (! addElementLen(fp, tr, p->next->next, FALSE, FALSE, &dummy)) return FALSE;	    
	}
      else 
	{                                    /*  A rooted format */
	  tr->rooted = TRUE;
	  if (ch != EOF)  (void) ungetc(ch, fp);
	}	
    }
  else 
    {
      p->next->next->back = (nodeptr) NULL;
    }
  if (! treeNeedCh(fp, ')', "in"))                return FALSE;
  (void) treeFlushLabel(fp);
  if (! treeFlushLen(fp))                         return FALSE;
 
  if (! treeNeedCh(fp, ';', "at end of"))       return FALSE;   
  
  if (tr->rooted) 
    {     
      p->next->next->back = (nodeptr) NULL;      
      tr->start = uprootTree(tr, p->next->next, FALSE, FALSE);      
      if (! tr->start)                              
	{
	  printf("FATAL ERROR UPROOTING TREE\n");
	  exit(-1);	  
	}    
    }
  else 
    {
      /*
	tr->start = p->next->next->back; 
	This is start used by treeString 
      */   
      tr->start = findAnyTip(p, tr->rdta->numsp);
    }
  
  setupPointerMesh(tr);


      

  if(tr->ntips < tr->mxtips)
    {
      if(adef->computeDistance)
	{
	  printf("Error: pairwise distance computation only allows for complete, i.e., containing all taxa\n");
	  printf("bifurcating starting trees\n");
	  exit(-1);
	}     
      if(adef->mode == CLASSIFY_ML)
	{	 
	  printf("RAxML classifier Algo: You provided a reference tree with %d taxa; alignmnet has %d taxa\n", tr->ntips, tr->mxtips);
	  printf("%d query taxa will be classifed under ML\n", tr->mxtips - tr->ntips);
	  classifyML(tr, adef);	  
	}
      else
	{
	  printf("You provided an incomplete starting tree %d alignmnet has %d taxa\n", tr->ntips, tr->mxtips);	  
	  makeParsimonyTreeIncomplete(tr, adef);	 		 
	}    
    }
  else
    {
      if(adef->mode == PARSIMONY_ADDITION)
	{
	  printf("Error you want to add sequences to a trees via MP stepwise addition, but \n");
	  printf("you have provided an input tree that already contains all taxa\n");
	  exit(-1);
	}
      if(adef->mode == CLASSIFY_ML)
	{
	  printf("Error you want to classify query sequences into a tree via ML, but \n");
	  printf("you have provided an input tree that already contains all taxa\n");
	  exit(-1);
	}
    }
  
 
  onlyInitrav(tr, tr->start);
 
  
  return TRUE;
}



/********************************MULTIFURCATIONS************************************************/


static boolean  addElementLenMULT (FILE *fp, tree *tr, nodeptr p, int partitionCounter)
{ 
  nodeptr  q, r, s;
  int      n, ch, fres, rn;
  double randomResolution;
  int old;
    
  tr->constraintVector[p->number] = partitionCounter; 

  if ((ch = treeGetCh(fp)) == '(') 
    {
      partCount++;
      old = partCount;       
      
      n = (tr->nextnode)++;
      if (n > 2*(tr->mxtips) - 2) 
	{
	  if (tr->rooted || n > 2*(tr->mxtips) - 1) 
	    {
	      printf("ERROR: Too many internal nodes.  Is tree rooted?\n");
	      printf("       Deepest splitting should be a trifurcation.\n");
	      return FALSE;
	    }
	  else 
	    {
	      tr->rooted = TRUE;	    
	    }
	}
      q = tr->nodep[n];
      tr->constraintVector[q->number] = partCount;
      if (! addElementLenMULT(fp, tr, q->next, old))        return FALSE;
      if (! treeNeedCh(fp, ',', "in"))             return FALSE;
      if (! addElementLenMULT(fp, tr, q->next->next, old))  return FALSE;
                 
      hookupDefault(p, q, tr->numBranches);

      while((ch = treeGetCh(fp)) == ',')
	{ 
	  n = (tr->nextnode)++;
	  if (n > 2*(tr->mxtips) - 2) 
	    {
	      if (tr->rooted || n > 2*(tr->mxtips) - 1) 
		{
		  printf("ERROR: Too many internal nodes.  Is tree rooted?\n");
		  printf("       Deepest splitting should be a trifurcation.\n");
		  return FALSE;
		}
	      else 
		{
		  tr->rooted = TRUE;
		}
	    }
	  r = tr->nodep[n];
	  tr->constraintVector[r->number] = partCount;	  

	  rn = randomInt(10000);
	  if(rn == 0) 
	    randomResolution = 0;
	  else 
	    randomResolution = ((double)rn)/10000.0;
	   	  
	   if(randomResolution < 0.5)
	    {	    
	      s = q->next->back;	      
	      r->back = q->next;
	      q->next->back = r;	      
	      r->next->back = s;
	      s->back = r->next;	      
	      addElementLenMULT(fp, tr, r->next->next, old);	     
	    }
	  else
	    {	  
	      s = q->next->next->back;	      
	      r->back = q->next->next;
	      q->next->next->back = r;	      
	      r->next->back = s;
	      s->back = r->next;	      
	      addElementLenMULT(fp, tr, r->next->next, old);	     
	    }	    	  	  
	}       

      if(ch != ')')
	{
	  printf("Missing /) in treeReadLenMULT\n");
	  exit(-1);	        
	}
	


      (void) treeFlushLabel(fp);
    }
  else 
    {                             
      ungetc(ch, fp);
      if ((n = treeFindTipName(fp, tr)) <= 0)          return FALSE;
      q = tr->nodep[n];      
      tr->constraintVector[q->number] = partitionCounter;

      if (tr->start->number > n)  tr->start = q;
      (tr->ntips)++;
      hookupDefault(p, q, tr->numBranches);
    }
  
  fres = treeFlushLen(fp);
  if(!fres) return FALSE;
    
  return TRUE;          
} 





boolean treeReadLenMULT (FILE *fp, tree *tr, analdef *adef)
{
  nodeptr  p, r, s;
  int      i, ch, n, rn;
  int partitionCounter = 0;
  double randomResolution;

  srand((unsigned int) time(NULL));
  
  for(i = 0; i < 2 * tr->mxtips; i++)
    tr->constraintVector[i] = -1;

  for (i = 1; i <= tr->mxtips; i++) 
    tr->nodep[i]->back = (node *) NULL;

  for(i = tr->mxtips + 1; i < 2 * tr->mxtips; i++)
    {
      tr->nodep[i]->back = (nodeptr)NULL;
      tr->nodep[i]->next->back = (nodeptr)NULL;
      tr->nodep[i]->next->next->back = (nodeptr)NULL;
      tr->nodep[i]->number = i;
      tr->nodep[i]->next->number = i;
      tr->nodep[i]->next->next->number = i;
    }


  tr->start       = tr->nodep[tr->mxtips];
  tr->ntips       = 0;
  tr->nextnode    = tr->mxtips + 1;
 
  for(i = 0; i < tr->numBranches; i++)
    tr->partitionSmoothed[i] = FALSE;

  tr->rooted      = FALSE;
 
  p = tr->nodep[(tr->nextnode)++]; 
  while((ch = treeGetCh(fp)) != '(');
      
  if (! addElementLenMULT(fp, tr, p, partitionCounter))                 return FALSE;
  if (! treeNeedCh(fp, ',', "in"))                return FALSE;
  if (! addElementLenMULT(fp, tr, p->next, partitionCounter))           return FALSE;
  if (! tr->rooted) 
    {
      if ((ch = treeGetCh(fp)) == ',') 
	{       
	  if (! addElementLenMULT(fp, tr, p->next->next, partitionCounter)) return FALSE;

	  while((ch = treeGetCh(fp)) == ',')
	    { 
	      n = (tr->nextnode)++;
	      assert(n <= 2*(tr->mxtips) - 2);
	
	      r = tr->nodep[n];	
	      tr->constraintVector[r->number] = partitionCounter;	   
	      
	      rn = randomInt(10000);
	      if(rn == 0) 
		randomResolution = 0;
	      else 
		randomResolution = ((double)rn)/10000.0;


	      if(randomResolution < 0.5)
		{	
		  s = p->next->next->back;		  
		  r->back = p->next->next;
		  p->next->next->back = r;		  
		  r->next->back = s;
		  s->back = r->next;		  
		  addElementLenMULT(fp, tr, r->next->next, partitionCounter);	
		}
	      else
		{
		  s = p->next->back;		  
		  r->back = p->next;
		  p->next->back = r;		  
		  r->next->back = s;
		  s->back = r->next;		  
		  addElementLenMULT(fp, tr, r->next->next, partitionCounter);
		}
	    }	  	  	      	  

	  if(ch != ')')
	    {
	      printf("Missing /) in treeReadLenMULT\n");
	      exit(-1);	        	      	      
	    }
	  else
	    ungetc(ch, fp);
	}
      else 
	{ 
	  tr->rooted = TRUE;
	  if (ch != EOF)  (void) ungetc(ch, fp);
	}       
    }
  else 
    {
      p->next->next->back = (nodeptr) NULL;
    }
    
  if (! treeNeedCh(fp, ')', "in"))                return FALSE;
  (void) treeFlushLabel(fp);
  if (! treeFlushLen(fp))                         return FALSE;
   
  if (! treeNeedCh(fp, ';', "at end of"))       return FALSE;
  

  if (tr->rooted) 
    {        
      p->next->next->back = (nodeptr) NULL;
      tr->start = uprootTree(tr, p->next->next, FALSE, TRUE);
      if (! tr->start)                              return FALSE;
    }
  else 
    {     
      tr->start = findAnyTip(p, tr->rdta->numsp);
    }

  
  

  if(tr->ntips < tr->mxtips)         
    makeParsimonyTreeIncomplete(tr, adef);          

  setupPointerMesh(tr);

  if(!adef->rapidBoot)
    onlyInitrav(tr, tr->start);
  return TRUE; 
}






void getStartingTree(tree *tr, analdef *adef)
{
  tr->likelihood = unlikely;
  
  if(adef->restart) 
    {	 	     	     
      INFILE = myfopen(tree_file, "r");	
                 		
      if(!adef->grouping)
	{
	  if (! treeReadLen(INFILE, tr, adef))
	    exit(-1);
	}
      else
	{
	  partCount = 0;
	  if (! treeReadLenMULT(INFILE, tr, adef))
	    exit(-1);
	}                                                               
     
     

      if(adef->mode == PARSIMONY_ADDITION)
	return; 

      evaluateGenericInitrav(tr, tr->start); 

      /*printf("%1.40f \n", tr->likelihood); */
           
      treeEvaluate(tr, 1);  
     
      /*printf("%1.40f \n", tr->likelihood);*/
               
      fclose(INFILE);
    }
  else
    { 
      assert(adef->mode != PARSIMONY_ADDITION &&
	     adef->mode != MORPH_CALIBRATOR && 
	     adef->mode != MORPH_CALIBRATOR_PARSIMONY);

      if(adef->randomStartingTree)	  
	makeRandomTree(tr, adef);       	   	 	   	  
      else
	makeParsimonyTree(tr, adef);	   	    	      		      	
      
      if(adef->startingTreeOnly)
	{
	  printStartingTree(tr, adef, TRUE);
	  exit(0);
	}
      else   	         
	printStartingTree(tr, adef, FALSE);     	         
            
      setupPointerMesh(tr);	  
      
      evaluateGenericInitrav(tr, tr->start);                                       	 
      
      treeEvaluate(tr, 1);        	     
    }         

  tr->start = tr->nodep[1];
}


/************************* string based *****************************************************/







static boolean treeNeedString(const char *fp, char c1, int *position)
{
  char 
    c2 = fp[(*position)++];
  
  if(c2 == c1)  
    return TRUE;
  else  
    {   
      int 
	lower = MAX(0, *position - 20),
	upper = *position + 20;
      
      printf("Tree Parsing ERROR: Expecting '%c', found: '%c'\n", c1, c2); 
      printf("Context: \n");
      
      while(lower < upper && fp[lower])
	printf("%c", fp[lower++]);
      
      printf("\n");

      return FALSE;
  }
} 



static boolean treeLabelEndString (char ch)
{
  switch(ch) 
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
      return TRUE;
    default:
      break;
    }
  
  return FALSE;
} 

static boolean  treeGetLabelString (const char *fp, char *lblPtr, int maxlen, int *position)
{
  char 
    ch;
  
  boolean  
    done, 
    lblfound;

  if (--maxlen < 0) 
    lblPtr = (char *)NULL; 
  else 
    if(lblPtr == NULL) 
      maxlen = 0;

  ch = fp[(*position)++];
  
  done = treeLabelEndString(ch);

  lblfound = !done;  

  while(!done) 
    {      
      if(treeLabelEndString(ch)) 
	break;     

      if(--maxlen >= 0) 
	*lblPtr++ = ch;
      
      ch = fp[(*position)++];      
    }
  
  (*position)--; 

  if (lblPtr != NULL) 
    *lblPtr = '\0';

  return lblfound;
}

static boolean  treeFlushLabelString(const char *fp, int *position)
{ 
  return  treeGetLabelString(fp, (char *) NULL, (int) 0, position);
} 


static boolean treeProcessLengthString (const char *fp, double *dptr, int *position)
{ 
  (*position)++;
  
  if(sscanf(&fp[*position], "%lf", dptr) != 1) 
    {
      printf("ERROR: treeProcessLength: Problem reading branch length\n");     
      assert(0);
    }

  while(fp[*position] != ',' && fp[*position] != ')' && fp[*position] != ';')
    *position = *position + 1;
  
  return  TRUE;
}

static int treeFlushLenString (const char *fp, int *position)
{
  double  
    dummy;  
  
  char     
    ch;

  ch = fp[(*position)++];
 
  if(ch == ':') 
    {     
      if(!treeProcessLengthString(fp, &dummy, position)) 
	return 0;
      return 1;	  
    }
    
  (*position)--;

  return 1;
} 

static int treeFindTipNameString (const char *fp, tree *tr, int *position)
{
  char    str[nmlngth+2];
  int      n;

  if(treeGetLabelString(fp, str, nmlngth+2, position))
    n = treeFindTipByLabelString(str, tr);
  else
    n = 0;
   
  return  n;
} 

static boolean addElementLenString(const char *fp, tree *tr, nodeptr p, boolean readBranchLengths, boolean readNodeLabels, int *lcount, int *position)
{
  nodeptr  
    q;
  
  int      
    n, 
    fres;

  char 
    ch;
  
  if ((ch = fp[(*position)++]) == '(') 
    { 
      n = (tr->nextnode)++;
      if (n > 2*(tr->mxtips) - 2) 
	{
	  if (tr->rooted || n > 2*(tr->mxtips) - 1) 
	    {
	      printf("ERROR: Too many internal nodes.  Is tree rooted?\n");
	      printf("       Deepest splitting should be a trifurcation.\n");
	      return FALSE;
	    }
	  else 
	    {
	      assert(!readNodeLabels);
	      tr->rooted = TRUE;
	    }
	}
      
      q = tr->nodep[n];

      if (!addElementLenString(fp, tr, q->next, readBranchLengths, readNodeLabels, lcount, position))        
	return FALSE;
      if (!treeNeedString(fp, ',', position))             
	return FALSE;
      if (!addElementLenString(fp, tr, q->next->next, readBranchLengths, readNodeLabels, lcount, position))  
	return FALSE;
      if (!treeNeedString(fp, ')', position))             
	return FALSE;
      
      if(readNodeLabels)
	{
	  char label[64];
	  int support;

	  if(treeGetLabelString(fp, label, 10, position))
	    {	
	      int val = sscanf(label, "%d", &support);
      
	      assert(val == 1);
	     
	      p->support = q->support = support;
	     
	      assert(p->number > tr->mxtips && q->number > tr->mxtips);
	      *lcount = *lcount + 1;
	    }
	}
      else	
	treeFlushLabelString(fp, position);
    }
  else 
    {   
      (*position)--;
     
      if ((n = treeFindTipNameString(fp, tr, position)) <= 0)          
	return FALSE;
      q = tr->nodep[n];
      
      if (tr->start->number > n)  tr->start = q;
      (tr->ntips)++;
    }
  
  if(readBranchLengths)
    {
      double branch;
      if (! treeNeedString(fp, ':', position))                 
	return FALSE;
      if (! treeProcessLengthString(fp, &branch, position))            
	return FALSE;            
      hookup(p, q, &branch, tr->numBranches);
    }
  else
    {      
      fres = treeFlushLenString(fp, position);
      if(!fres) 
	return FALSE;
      
      hookupDefault(p, q, tr->numBranches);
    }

  return TRUE;          
}

int treeReadTopologyString(const char *fp, tree *tr, boolean readBranches, boolean readNodeLabels, int *position, boolean topologyOnly, analdef *adef)
{ 
  nodeptr  
    p;
  
  int   
    lcount = 0,
    i;
  
  char 
    ch;       

  for (i = 1; i <= tr->mxtips; i++)
    {
      tr->nodep[i]->back = (node *)NULL;
      if(topologyOnly)
	tr->nodep[i]->support = -1;
    }
  
  for(i = tr->mxtips + 1; i < 2 * tr->mxtips; i++)
    {
      tr->nodep[i]->back = (nodeptr)NULL;
      tr->nodep[i]->next->back = (nodeptr)NULL;
      tr->nodep[i]->next->next->back = (nodeptr)NULL;
      tr->nodep[i]->number = i;
      tr->nodep[i]->next->number = i;
      tr->nodep[i]->next->next->number = i;
      
      if(topologyOnly)
	{
	  tr->nodep[i]->support = -2;
	  tr->nodep[i]->next->support = -2;
	  tr->nodep[i]->next->next->support = -2;
	}
    }
    
  if(topologyOnly)
    tr->start       = tr->nodep[tr->mxtips];
  else
    tr->start       = tr->nodep[1];

  tr->ntips       = 0;
  tr->nextnode    = tr->mxtips + 1;
  
  for(i = 0; i < tr->numBranches; i++)
    tr->partitionSmoothed[i] = FALSE;
  
  tr->rooted      = FALSE;      
  
  p = tr->nodep[(tr->nextnode)++]; 
  
  if(fp[(*position)++] != '(')
    assert(0);

  if(!topologyOnly)
    assert(readBranches == FALSE && readNodeLabels == FALSE);
    
  if (! addElementLenString(fp, tr, p, readBranches, readNodeLabels, &lcount, position))                 
    assert(0);
  
  if (! treeNeedString(fp, ',', position))                
    assert(0);
   
  if (! addElementLenString(fp, tr, p->next, readBranches, readNodeLabels, &lcount, position))           
    assert(0);

  if (! tr->rooted) 
    {
      if ((ch = fp[(*position)++]) == ',') 
	{ 
	  if (! addElementLenString(fp, tr, p->next->next, readBranches, readNodeLabels, &lcount, position)) 
	    assert(0);	 
	}
      else 
	{                                    
	  tr->rooted = TRUE;
	  (*position)--;
	}	
    }
  else     
    p->next->next->back = (nodeptr) NULL;
    
  
  if (! treeNeedString(fp, ')', position))                
    assert(0);
  
  if(topologyOnly)
    assert(!(tr->rooted && readNodeLabels));

  treeFlushLabelString(fp, position);
  
  if (!treeFlushLenString(fp, position))                         
    assert(0);
  
  if (!treeNeedString(fp, ';', position))       
    assert(0);
    
  if (tr->rooted) 
    {            
      assert(!readNodeLabels);
      
      p->next->next->back = (nodeptr) NULL;      
      tr->start = uprootTree(tr, p->next->next, readBranches, FALSE);
      if (! tr->start)                              
	{
	  printf("FATAL ERROR UPROOTING TREE\n");
	  assert(0);	  
	}      
    }
  else           
    tr->start = findAnyTip(p, tr->mxtips);       
  
  if(!topologyOnly)
    {
      setupPointerMesh(tr); 

      assert(tr->ntips <= tr->mxtips);

      if(tr->ntips < tr->mxtips)
	{
	  if(adef->computeDistance)
	    {
	      printf("Error: pairwise distance computation only allows for complete, i.e., containing all taxa\n");
	      printf("bifurcating starting trees\n");
	      exit(-1);
	    }     
	  if(adef->mode == CLASSIFY_ML)
	    {	 
	      printf("RAxML classifier Algo: You provided a reference tree with %d taxa; alignmnet has %d taxa\n", tr->ntips, tr->mxtips);
	      printf("%d query taxa will be classifed under ML\n", tr->mxtips - tr->ntips);
	      classifyML(tr, adef);	  
	    }
	  else
	    {
	      printf("You provided an incomplete starting tree %d alignmnet has %d taxa\n", tr->ntips, tr->mxtips);	  
	      makeParsimonyTreeIncomplete(tr, adef);	 		 
	    }    
	}
      else
	{
	  if(adef->mode == PARSIMONY_ADDITION)
	    {
	      printf("Error you want to add sequences to a trees via MP stepwise addition, but \n");
	      printf("you have provided an input tree that already contains all taxa\n");
	      exit(-1);
	    }
	  if(adef->mode == CLASSIFY_ML)
	    {
	      printf("Error you want to classify query sequences into a tree via ML, but \n");
	      printf("you have provided an input tree that already contains all taxa\n");
	      exit(-1);
	    }
	}
  
      onlyInitrav(tr, tr->start);
    }

  return lcount;
} 

