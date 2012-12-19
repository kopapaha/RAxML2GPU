#include <stdio.h>

#include "axml.h"
#include "kernel.h"
#define DIMBLOCKX 32




/*
__device__ __inline__ int ld_gbl_cg(int *addr) {
  int return_value;
  asm("ld.global.cg.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}
 */

//ld_gbl_cg((int *) &b.sense)
/*
 *
 * global barrier
 * GTX 480 max dims: 120x128 (blocks, threads/block)
 *  
 */
__device__ void globalBarrier()
{
 //int ti = dimBlock.x*blockIdx.x + threadIdx.x;
 int nofBlocks = gridDim.x*gridDim.y*gridDim.z;
 
 if (threadIdx.x == 0){
     
      //int mySense = !(ld_gbl_cg((int *) &b.sense));
     int mySense = !b.sense;
     
      int old = atomicAdd((int *)&b.blockFinish, 1);

      if (old == nofBlocks-1){
            b.blockFinish = 0;
            b.sense = !b.sense;
      }
      else
      {
          while (mySense != b.sense);
      }

 }
 __syncthreads();

}




__device__ int isTip(int number, int maxTips)
{
  assert(number > 0);

  if(number <= maxTips)
    return TRUE;
  else
    return FALSE;
}





__device__ void hookup (nodeptr p, nodeptr q, double *z, int numBranches)
{
  int i;

  p->back = q;
  q->back = p;

  for(i = 0; i < numBranches; i++)
    p->z[i] = q->z[i] = z[i];
}





__device__ boolean allSmoothed(tree *tr)
{
  int i;
  boolean result = TRUE;
  
  for(i = 0; i < tr->numBranches; i++)
    {
      if(tr->partitionSmoothed[i] == FALSE)
	result = FALSE;
      else
	tr->partitionConverged[i] = TRUE;
    }

  return result;
}

/*
 * makeP_FLOAT(qz, rz, tr->cdta->patrat,   
                        tr->partitionData[model].EI, tr->partitionData[model].EIGN,
                        tr->NumberOfCategories, left_FLOAT, right_FLOAT, DNA_DATA);
 */
/*
 * makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
			tr->partitionData[model].EI, tr->partitionData[model].EIGN,
			4, left_FLOAT, right_FLOAT, DNA_DATA);
 */


/*
static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
{
  int i, j, k;
//NumberOfCategories 1 2 25

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

*/



/*
 * newviewGTRCAT_FLOAT(tInfo->tipCase,  tr->partitionData[model].EV_FLOAT, tr->partitionData[model].rateCategory,
					       x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
					       ex3, tipX1, tipX2,
					       width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling
					       );
 *               newviewGTRCAT_FLOAT(tInfo->tipCase, EV, rateCategory,
					       x1_start, x2_start, x3_start, tipVector,
					       tipX1, tipX2,
					       left, right, wgt
					       );
 */
__device__ void newviewGTRCAT_FLOAT( int tipCase,  float *EV,  int *cptr,
				 float *x1_start,  float *x2_start,  float *x3_start,  float *tipVector,
				 unsigned char *tipX1, unsigned char *tipX2,
				 float *left, float *right, int *wgt, int *scalerIncrement)
{
  float
    *le,
    *ri,
    *x1, *x2, *x3;
  float
    ump_x1, ump_x2; 
          
  int i, j, k, scale, addScale = 0;

  float x1px2_s[4];
  
  switch(tipCase)
    {
    case TIP_TIP:
      {
	//for (i = 0; i < n; i++)
	  //{
          i = blockDim.x*blockIdx.x + threadIdx.x;
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
		x1px2_s[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 4; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 4; j++)
	      for(k = 0; k < 4; k++)
		x3[k] += x1px2_s[j] * EV[j * 4 + k];	    
	  //}
      }
      break;
    case TIP_INNER:
      {
	//for (i = 0; i < n; i++)
	  //{
          i = blockDim.x*blockIdx.x + threadIdx.x;
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
		x1px2_s[j] = ump_x1 * ump_x2;
	      }

	    for(j = 0; j < 4; j++)
	      x3[j] = 0.0;

	    for(j = 0; j < 4; j++)
	      for(k = 0; k < 4; k++)
		x3[k] +=  x1px2_s[j] *  EV[4 * j + k];	    

	    scale = 1;

	    for(j = 0; j < 4 && scale; j++)
            {
                if (x3[j] < minlikelihood_FLOAT && x3[j] > minusminlikelihood_FLOAT)
                {
                    scale = 1;
                }
                else
                {
                    scale = 0;
                }
            }
	      //scale = (x3[j] < minlikelihood_FLOAT && x3[j] > minusminlikelihood_FLOAT);


	    if(scale)
	      {
		for(j = 0; j < 4; j++)
		  x3[j] *= twotothe256_FLOAT;

		//if(useFastScaling)
		  addScale = addScale + wgt[i];
		//else
		 // ex3[i]  += 1;	      

	      }
	  //}
      }
      break;
    case INNER_INNER:
      //for (i = 0; i < n; i++)
	//{
        i = blockDim.x*blockIdx.x + threadIdx.x;
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
	      x1px2_s[j] = ump_x1 * ump_x2;
	    }

	  for(j = 0; j < 4; j++)
	    x3[j] = 0.0;

	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      x3[k] +=  x1px2_s[j] *  EV[4 * j + k];	  

	  scale = 1;

	  for(j = 0; j < 4 && scale; j++)
          {
              if (x3[j] < minlikelihood_FLOAT && x3[j] > minusminlikelihood_FLOAT)
              {
                  scale = 1;
              }
              else
              {
                  scale = 0;
              }
          }
	    //scale = (x3[j] < minlikelihood_FLOAT && x3[j] > minusminlikelihood_FLOAT);


	  if(scale)
	    {
	      for(j = 0; j < 4; j++)
		x3[j] *= twotothe256_FLOAT;


	      //if(useFastScaling)
		addScale += wgt[i];
	     // else
		//ex3[i]  += 1;	     

	    }
	//}
      break;
    default:
      assert(0);
    }


  //if(useFastScaling)
  //scalerIncrement[i] = addScale;
  //atomicAdd(&scalerIncrement, addScale);
  //atomicAdd(&scaleNum, 1);
    *scalerIncrement = addScale;


}









/*
 * newviewGTRGAMMA_FLOAT(tInfo->tipCase,
                        x1_start_FLOAT, x2_start_FLOAT, x3_start_FLOAT, tr->partitionData[model].EV_FLOAT, tr->partitionData[model].tipVector_FLOAT,
			ex3, tipX1, tipX2,
			width, left_FLOAT, right_FLOAT, wgt, &scalerIncrement, tr->useFastScaling);
 */
__device__ void newviewGTRGAMMA_FLOAT_shared(int tipCase,
                                  float *x1_start, float *x2_start, float *x3_start,
                                  float *EV, float *tipVector,
                                  unsigned char *tipX1, unsigned char *tipX2,
                                  float *left, float *right, int *wgt, int *scalerIncrement
                                  )
{

  float
    *x1, *x2, *x3;
  float
    buf,
    ump_x1,
    ump_x2;
  int i, j, k, l, scale, addScale = 0;

  __shared__ float  umpX1_s[256], umpX2_s[256];
  float *uX1, *uX2, x1px2_s[4]; //aftr1
  //float *uX1, *umpX1, *uX2, *umpX2;
  //umpX1 = d_umpX1 + threadIdx.x*256;
  //umpX2 = d_umpX2 + threadIdx.x*256;
          
  boolean step;
  int thi;
  
  switch(tipCase)
    {
    case TIP_TIP:
      {

        //float *uX1, umpX1[256], *uX2, umpX2[256]; //bfr1        
        
        step = (240%32!=0)?1:0;
        for(i=0; i < 240/32+step; i++){
            thi = i*32 + threadIdx.x;
            if(thi < 240){
                umpX1_s[thi+16] = 0.0;
                umpX2_s[thi+16] = 0.0;
                for(j=0; j<4; j++){
                    umpX1_s[thi+16] += tipVector[(thi/16+1)*4 + j]*left[(thi*4+j)%64];
                    umpX2_s[thi+16] += tipVector[(thi/16+1)*4 + j]*right[(thi*4+j)%64];
                }
            }
        }
        __syncthreads();
          
          
/*
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
*/

        //for (i = 0; i < n; i++)
          //{
        i = blockDim.x*blockIdx.x + threadIdx.x; //for alignLength threads in a grid
        if (i >= alignLength_d)
           return;
        
            x3 = &x3_start[i * 16];
            
            uX1 = &umpX1_s[16 * tipX1[i]];
            uX2 = &umpX2_s[16 * tipX2[i]];

            for(j = 0; j < 16; j++)
              x3[j] = 0.0;

            for (j = 0; j < 4; j++)
              {

                for (k = 0; k < 4; k++)
                  {
                    buf = uX1[j*4 + k] * uX2[j*4 + k];

                    for (l=0; l<4; l++)
                      x3[j * 4 + l] +=  buf * EV[4 * k + l];

                  }

              }            
          //}
      }
      break;
    case TIP_INNER:
      {
	
        //float *uX1, umpX1[256]; //bfr1      

        step = (240%32!=0)?1:0;
        for(i=0; i < 240/32+step; i++){
            thi = i*32 + threadIdx.x;
            if(thi < 240){
                umpX1_s[thi+16] = 0.0;
                for(j=0; j<4; j++){
                    umpX1_s[thi+16] += tipVector[(thi/16+1)*4 + j]*left[(thi*4+j)%64];
                }
            }
        }
        __syncthreads();          
          
          

/*
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
*/

         //for (i = 0; i < n; i++)
           //{
           i = blockDim.x*blockIdx.x + threadIdx.x; //for 2048 threads in a grid
        
           if (i >= alignLength_d)
               return;
             x2 = &x2_start[i * 16];
             x3 = &x3_start[i * 16];

             uX1 = &umpX1_s[16 * tipX1[i]];

             for(j = 0; j < 16; j++)
               x3[j] = 0.0;

             for (j = 0; j < 4; j++)
               {


                 for (k = 0; k < 4; k++)
                   {
                     ump_x2 = 0.0;

                     for (l=0; l<4; l++)
                       ump_x2 += x2[j*4 + l] * right[j* 16 + k*4 + l];
                     x1px2_s[k] = uX1[j * 4 + k] * ump_x2;
                   }

                 for(k = 0; k < 4; k++)
                   for (l=0; l<4; l++)
                     x3[j * 4 + l] +=  x1px2_s[k] * EV[4 * k + l];

               }             




             scale = 1;
             for(l = 0; scale && (l < 16); l++)
             {
                 if (ABS(x3[l]) <  minlikelihood_FLOAT)
                     scale = 1;
                 else
                     scale = 0;
             }
               //scale = (ABS(x3[l]) <  minlikelihood_FLOAT);

             if(scale)
               {
                 for (l=0; l<16; l++)
                   x3[l] *= twotothe256_FLOAT;

		 //if(useFastScaling)
		   addScale += wgt[i];
		 //else
		   //ex3[i]  += 1;             
               }




           //}
      }
      break;
    case INNER_INNER:    
     //for (i = 0; i < n; i++)
       //{
         i = blockDim.x*blockIdx.x + threadIdx.x; //for 2048 threads in a grid
        if (i >= alignLength_d)
           return;
         
         x1 = &x1_start[i * 16];
         x2 = &x2_start[i * 16];
         x3 = &x3_start[i * 16];

         for(j = 0; j < 16; j++)
           x3[j] = 0.0;

         for (j = 0; j < 4; j++)
           {
             for (k = 0; k < 4; k++)
               {
                 ump_x1 = 0.0;
                 ump_x2 = 0.0;

                 for (l=0; l<4; l++)
                   {
                     ump_x1 += x1[j*4 + l] * left[j*16 + k*4 +l];
                     ump_x2 += x2[j*4 + l] * right[j*16 + k*4 +l];
                   }

                 x1px2_s[k] = ump_x1 * ump_x2;
               }




             for(k = 0; k < 4; k++)
               for (l=0; l<4; l++)
                 x3[j * 4 + l] +=  x1px2_s[k] * EV[4 * k + l];



           }

        
         scale = 1;




         for(l = 0; scale && (l < 16); l++)
         {
            if (ABS(x3[l]) <  minlikelihood_FLOAT)
                scale = 1;
            else
                scale = 0;
         }             
           //scale = (ABS(x3[l]) <  minlikelihood_FLOAT);

         if(scale)
           {
             for (l=0; l<16; l++)
               x3[l] *= twotothe256_FLOAT;

	     //if(useFastScaling)
	       addScale += wgt[i];
	     //else
	      // ex3[i]  += 1;            
           }
        
       //}
     break;
    default:
      assert(0);
    }

  //if(useFastScaling)
  //scalerIncrement[i] = addScale;
    *scalerIncrement = addScale;
  //atomicAdd(&scalerIncrement, addScale);
  //atomicAdd(&scaleNum, 1);
  
}



__device__ void newviewGTRGAMMA_FLOAT(int tipCase,
                                  float *x1_start, float *x2_start, float *x3_start,
                                  float *EV, float *tipVector,
                                  unsigned char *tipX1, unsigned char *tipX2,
                                  float *left, float *right, int *wgt, int *scalerIncrement
                                  )
{

  float
    *x1, *x2, *x3;
  float
    buf,
    ump_x1,
    ump_x2;
  int i, j, k, l, scale, addScale = 0;

  float x1px2[4], umpX1[256], umpX2[256];
  float *uX1, *uX2; //aftr1
  //float *uX1, *umpX1, *uX2, *umpX2;
  //umpX1 = d_umpX1 + threadIdx.x*256;
  //umpX2 = d_umpX2 + threadIdx.x*256;
          

  
  switch(tipCase)
    {
    case TIP_TIP:
      {

        //float *uX1, umpX1[256], *uX2, umpX2[256]; //bfr1        
        
          
          

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


        //for (i = 0; i < n; i++)
          //{
        i = blockDim.x*blockIdx.x + threadIdx.x; //for alignLength threads in a grid
            x3 = &x3_start[i * 16];
            
            uX1 = &umpX1[16 * tipX1[i]];
            uX2 = &umpX2[16 * tipX2[i]];

            for(j = 0; j < 16; j++)
              x3[j] = 0.0;

            for (j = 0; j < 4; j++)
              {

                for (k = 0; k < 4; k++)
                  {
                    buf = uX1[j*4 + k] * uX2[j*4 + k];

                    for (l=0; l<4; l++)
                      x3[j * 4 + l] +=  buf * EV[4 * k + l];

                  }

              }            
          //}
      }
      break;
    case TIP_INNER:
      {
	
        //float *uX1, umpX1[256]; //bfr1      

          


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


         //for (i = 0; i < n; i++)
           //{
           i = blockDim.x*blockIdx.x + threadIdx.x; //for 2048 threads in a grid

             x2 = &x2_start[i * 16];
             x3 = &x3_start[i * 16];

             uX1 = &umpX1[16 * tipX1[i]];

             for(j = 0; j < 16; j++)
               x3[j] = 0.0;

             for (j = 0; j < 4; j++)
               {


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

               }             




             scale = 1;
             for(l = 0; scale && (l < 16); l++)
             {
                 if (ABS(x3[l]) <  minlikelihood_FLOAT)
                     scale = 1;
                 else
                     scale = 0;
             }
               //scale = (ABS(x3[l]) <  minlikelihood_FLOAT);

             if(scale)
               {
                 for (l=0; l<16; l++)
                   x3[l] *= twotothe256_FLOAT;

		 //if(useFastScaling)
		   addScale += wgt[i];
		 //else
		   //ex3[i]  += 1;             
               }




           //}
      }
      break;
    case INNER_INNER:    
     //for (i = 0; i < n; i++)
       //{
         i = blockDim.x*blockIdx.x + threadIdx.x; //for 2048 threads in a grid
        
         x1 = &x1_start[i * 16];
         x2 = &x2_start[i * 16];
         x3 = &x3_start[i * 16];

         for(j = 0; j < 16; j++)
           x3[j] = 0.0;

         for (j = 0; j < 4; j++)
           {
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



           }

        
         scale = 1;




         for(l = 0; scale && (l < 16); l++)
         {
            if (ABS(x3[l]) <  minlikelihood_FLOAT)
                scale = 1;
            else
                scale = 0;
         }             
           //scale = (ABS(x3[l]) <  minlikelihood_FLOAT);

         if(scale)
           {
             for (l=0; l<16; l++)
               x3[l] *= twotothe256_FLOAT;

	     //if(useFastScaling)
	       addScale += wgt[i];
	     //else
	      // ex3[i]  += 1;            
           }
        
       //}
     break;
    default:
      assert(0);
    }

  //if(useFastScaling)
  //scalerIncrement[i] = addScale;
    *scalerIncrement = addScale;
  //atomicAdd(&scalerIncrement, addScale);
  //atomicAdd(&scaleNum, 1);
  
}






__device__ void calcDiagptable_FLOAT(double z, int numberOfCategories, double *rptr, double *EIGN, float *diagptable)
{
  int i;
  double lz;

  if (z < zmin) 
    lz = log(zmin);
  else
    lz = log(z);
  

        double lz1, lz2, lz3;
	lz1 = EIGN[0] * lz;
	lz2 = EIGN[1] * lz;
	lz3 = EIGN[2] * lz;

	for(i = 0; i <  numberOfCategories; i++)
	  {		 
	    diagptable[4 * i] = 1.0;
	    diagptable[4 * i + 1] = (float)(exp(rptr[i] * lz1));
	    diagptable[4 * i + 2] = (float)(exp(rptr[i] * lz2));
	    diagptable[4 * i + 3] = (float)(exp(rptr[i] * lz3));	   
	  }
      

}



//evaluateGTRCAT_FLOAT(rateCategory, wgt, x1_start, x2_start, tipVector, tip, left);
__device__ double evaluateGTRCAT_FLOAT(
        volatile int *cptr, volatile int *wptr, 
        volatile float *x1_start, volatile float *x2_start, 
        volatile float *tipVector, volatile unsigned char *tipX1, volatile float *diagptable_start)
{
  volatile float  sum = 0.0, term;       
  volatile int     i, j;  
  volatile float  *diagptable, *x1, *x2;
  
  volatile float x1j, x2j, dj;
  volatile unsigned char tX1;
  
  i = blockDim.x*blockIdx.x + threadIdx.x;
  
  
  
  
  
  
if(tipX1)
    {          
      //for (i = 0; i < n; i++) 
	//{
          tX1 = tipX1[i];
	  x1 = &(tipVector[4 * tX1]);
	  x2 = &x2_start[4 * i];
	  
	  diagptable = &diagptable_start[4 * cptr[i]];	    	    	  
	  
	  for(j = 0, term = 0.0; j < 4; j++)
          {
              x1j = x1[j];
	      x2j = x2[j];
              //x2j = x2_start[4 * i + j];
              dj = diagptable[j];
              term += x1j*x2j*dj;
          }
	  //if(fastScaling)	   	       
	    term = logf(term);
	  //else
	    //term = LOGF(term) + (ex2[i] * LOGF(minlikelihood_FLOAT));	   	    	   	 	  	  	 
	  
	  sum += wptr[i] * term;
	//}	
    }               
  else
    {
      //for (i = 0; i < n; i++) 
	//{	 	           	
	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];
	  
	  diagptable = &diagptable_start[4 * cptr[i]];		  
	  
	  for(j = 0, term = 0.0; j < 4; j++)
	    term += x1[j] * x2[j] * diagptable[j];     	
	  
	  //if(fastScaling)	   	       
	    term = logf(term);
	  //else
	    //term = LOGF(term) + ((ex1[i] + ex2[i]) * LOGF(minlikelihood_FLOAT));	  

	  sum += wptr[i] * term;
	//}    
    }        

  return  ((double)sum);  
    
}



//evaluateGTRGAMMA_FLOAT(wgt, x1_start, x2_start, tipVector, tip, left);
__device__ double evaluateGTRGAMMA_FLOAT(
        volatile int *wptr, volatile float *x1_start, volatile float *x2_start, 
	volatile float *tipVector, volatile unsigned char *tipX1, volatile float *diagptable)
{
  volatile float   sum = 0.0, term;    
  volatile int     i, j, k;
  volatile float  *x1, *x2;    
  
  i = blockDim.x*blockIdx.x + threadIdx.x;


  if(tipX1)
    {         
     // for (i = 0; i < n; i++)
	//{
	  x1 = &(tipVector[4 * tipX1[i]]);	 
	  x2 = &x2_start[16 * i];	          	  	
	  
	  for(j = 0, term = 0.0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      term += x1[k] * x2[j * 4 + k] * diagptable[j * 4 + k];	          	  	  	    	    	    
	  
	  //if(fastScaling)
	    term = logf(0.25 * term);
	  //else
	    //term = LOGF(0.25 * term) + ex2[i] * LOGF(minlikelihood_FLOAT);	 
	  
	  sum += wptr[i] * term;
	//}     
    }
  else
    {         
      //for (i = 0; i < n; i++) 
	//{	  	 	  	  
	  x1 = &x1_start[16 * i];
	  x2 = &x2_start[16 * i];	  	  
	  
	  for(j = 0, term = 0.0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      term += x1[j * 4 + k] * x2[j * 4 + k] * diagptable[j * 4 + k];
	  
	  //if(fastScaling)
	    term = logf(0.25 * term);
	  //else
	    //term = LOGF(0.25 * term) + (ex1[i] + ex2[i]) * LOGF(minlikelihood_FLOAT);

	  sum += wptr[i] * term;
	//}                      	
    }

  return ((double)sum);
    
}







/*
 * kernel version only for -m GTRCAT_FLOAT model
 */

/*
 *   startKernel<<<dimGrid, dimBlock>>>(nofSpecies, countr, pIsTip, qIsTip, 
 * 
 * d_partitionLikelihood, d_scalerThread, d_EV, d_tipVector, d_gammaRates, d_globalScaler, 
 * d_left, d_right, d_patrat, d_ei, d_eign, d_rateCategory, d_wgt, d_tree, d_ti, d_xVector, d_yVector);
*/
    __global__ void startKernelnewViewEvaluate(
    int execModel,
        int countr,
        int pIsTip,
        int qIsTip)
{
    
    //traversalInfo *ti = d_ti;
        double p1, e1, ex1;
   int i;
   int j, k, l;
   double d1[3], d2[3];
   float left[400], right[400];
   //float *left, *right;
   //left = d_left + threadIdx.x*400;
   //right = d_right + threadIdx.x*400;
   
   float
        *x1_start,
	*x2_start,
	*x3_start;  
    
   //int
	//states = tr->partitionData[model].states,
	int scalerIncrement;
	//*wgt = (int*)NULL,	       
	//*ex3 = (int*)NULL;
    
   unsigned char
	*tipX1,
	*tipX2;
  
   double qz, rz;
   //int width =  tr->partitionData[model].width;    
    
   //traversalInfo *ti   = tr->td[0].ti; //ok
   traversalInfo *tInfo;
   //int tx = threadIdx.x;
   
   int thi = blockDim.x*blockIdx.x + threadIdx.x;
   if (thi >= alignLength_d)
       return;
   if (initScalerThread==0 && countr==0)
   {
   //if (tr->td[0].count>1){
     for(j=0; j<2*nofSpecies_d; j++)
     {
         scalerThread[thi*2*nofSpecies_d + j] =0;// globalScaler[j];
     }
     if (thi==0)
         initScalerThread = 1;
   //}
   }
    
   
   for(i = 1; i < tr->td[0].count; i++)
   {
       tInfo = &ti[i];
       
       
        
        x1_start = (float*)NULL;
	x2_start = (float*)NULL;
	x3_start = (float*)NULL;  
    
	scalerIncrement = 0;

	tipX1 = (unsigned char *)NULL;
	tipX2 = (unsigned char *)NULL;
       
       
       
       
       switch(tInfo->tipCase)
       {
        case TIP_TIP:
           tipX1    = yVector[tInfo->qNumber];
           tipX2    = yVector[tInfo->rNumber];
           
           x3_start = xVector[tInfo->pNumber - tr->mxtips - 1];
           break;
        case TIP_INNER:
           tipX1    =  yVector[tInfo->qNumber];

           x2_start = xVector[tInfo->rNumber - tr->mxtips - 1];		 		    		 
	   x3_start = xVector[tInfo->pNumber - tr->mxtips - 1];
           break;
        case INNER_INNER:
	   x1_start = xVector[tInfo->qNumber - tr->mxtips - 1];		  
	   x2_start = xVector[tInfo->rNumber - tr->mxtips - 1];		 
	   x3_start = xVector[tInfo->pNumber - tr->mxtips - 1];		  
	   break;
	default:
           assert(0);
       }
       
       //left_FLOAT = tr->partitionData[model].left_FLOAT; //OK
       //right_FLOAT = tr->partitionData[model].right_FLOAT;  //OK
       // (*tInfo).qz[0];  (*tInfo).*(qz + 0);
       qz = tInfo->qz[0];
       rz = tInfo->rz[0];      

       //case DNA_DATA:
       
       switch(tr->rateHetModel)
       {
           case CAT:
              /*makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);*/
               
               //NumberOfCategories 1 2 25
               
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               //every thread compute its private left and right vectors
                  for(l = 0; l < tr->NumberOfCategories; l++)
                  {

                    for(j = 0; j < 3; j++)
                      {
                        p1 = patrat[l];
                        e1 = eign[j];
                        ex1 = exp(p1 * e1 * qz);
                        d1[j] = ex1;
                        d2[j] = exp(patrat[l] * eign[j] * rz);
                      }


                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * ei[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * ei[3 * j + k]));
                            //left[l * 16 + j * 4 + k + 1]  = ((float)( ei[3 * j + k]));
                            //right[l * 16 + j * 4 + k + 1] = ((float)( ei[3 * j + k]));
                          }

                      }

                  }
                  
                

#ifndef notdef
              newviewGTRCAT_FLOAT(tInfo->tipCase, EV, rateCategory,
					       x1_start, x2_start, x3_start, tipVector,
					       tipX1, tipX2,
					       left, right, wgt ,&scalerIncrement
					       );
#endif
              break;

           case GAMMA:
           case GAMMA_I:
/*
               makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, DNA_DATA);
*/
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               
                  for(l = 0; l < 4; l++)
                  {

                    for(j = 0; j < 3; j++)
                      {
                        d1[j] = exp(gammaRates[l] * eign[j] * qz);
                        d2[j] = exp(gammaRates[l] * eign[j] * rz);
                      }


                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * ei[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * ei[3 * j + k]));
                            //left[l * 16 + j * 4 + k + 1]  = ((float)( ei[3 * j + k]));
                            //right[l * 16 + j * 4 + k + 1] = ((float)( ei[3 * j + k]));
                          }

                      }

                  }               

#ifndef notdef
                  
               newviewGTRGAMMA_FLOAT(tInfo->tipCase,
						x1_start, x2_start, x3_start, EV, tipVector,
						tipX1, tipX2,
						left, right, wgt, &scalerIncrement);			
#endif
               break;
           default:
               assert(0);
       }


#ifndef notdef
       //if(tr->useFastScaling)
        //{
       /*
       if(thi == 0)
       {
           globalScaler[tInfo->pNumber] =
                   globalScaler[tInfo->qNumber] + 
                   globalScaler[tInfo->rNumber] +
                   scalerIncrement[thi]; //!!!
           assert(globalScaler[tInfo->pNumber] < INT_MAX);
       }
       else
       {
           atomicAdd(&(globalScaler[tInfo->pNumber]), scalerIncrement[thi]);
       }
       */
       /*ver2 wrong*/

           scalerThread[thi*2*nofSpecies_d + tInfo->pNumber] =
                   scalerThread[thi*2*nofSpecies_d +tInfo->qNumber] + 
                   scalerThread[thi*2*nofSpecies_d +tInfo->rNumber] +
                   scalerIncrement; //!!!       
       
       /* ver1 wrong
       if(scaleNum == 2048)
       {
           globalScaler[tInfo->pNumber] =
                   globalScaler[tInfo->qNumber] + 
                   globalScaler[tInfo->rNumber] +
                   scalerIncrement; //!!!
           assert(globalScaler[tInfo->pNumber] < INT_MAX);
       }
        */
           /* serial code:
            * if(tr->useFastScaling)
		{
		  tr->partitionData[model].globalScaler[tInfo->pNumber] = 
		    tr->partitionData[model].globalScaler[tInfo->qNumber] + 
		    tr->partitionData[model].globalScaler[tInfo->rNumber] +
		    (unsigned int)scalerIncrement;
		  assert(tr->partitionData[model].globalScaler[tInfo->pNumber] < INT_MAX);
		}
            */
	//}
#endif
   }
   
   
   /*
    * evaluateIterative
    */
   int pNumber,qNumber;
   double pL;
   unsigned char *tip = (unsigned char*)NULL;
   //double z; //qz
   //float *diagptable; //left
   
   pNumber = ti[0].pNumber;
   qNumber = ti[0].qNumber;
   qz = ti[0].qz[0];
   
   x1_start   = (float*)NULL;
   x2_start   = (float*)NULL;
   //*diagptable = (float*)NULL;
   
   //diagptable = left;
   
   
   if(pIsTip || qIsTip)
   {	        	    
      if(qIsTip)
	{	
	  x2_start = xVector[pNumber - tr->mxtips -1];
	  tip = yVector[qNumber];	 	      
	}           
      else
	{
	  x2_start = xVector[qNumber - tr->mxtips - 1];  
	  tip = yVector[pNumber];
	}
   }
  else
   {
	x1_start = xVector[pNumber - tr->mxtips - 1];
	x2_start = xVector[qNumber - tr->mxtips - 1];
   }
   
   
   switch(tr->rateHetModel)
   {
       case CAT:
        //   calcDiagptable_FLOAT(z, DNA_DATA, tr->NumberOfCategories, tr->cdta->patrat, tr->partitionData[model].EIGN, diagptable_FLOAT);
           calcDiagptable_FLOAT(qz, tr->NumberOfCategories, patrat, eign, left);
			  
	//   partitionLikelihood =  evaluateGTRCAT_FLOAT(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
	//							      x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT, 
	//							      tip, width, diagptable_FLOAT, tr->useFastScaling);
           pL = evaluateGTRCAT_FLOAT(rateCategory, wgt, x1_start, x2_start, tipVector, tip, left);
           
           break;
       case GAMMA:
           //calcDiagptable_FLOAT(z, DNA_DATA, 4, tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN, diagptable_FLOAT);
           calcDiagptable_FLOAT(qz, 4, gammaRates, eign, left);
        //   partitionLikelihood = evaluateGTRGAMMA_FLOAT(ex1, ex2, tr->partitionData[model].wgt,
	//							       x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
	//							       tip, width, diagptable_FLOAT, tr->useFastScaling);   
           pL = evaluateGTRGAMMA_FLOAT(wgt, x1_start, x2_start, tipVector, tip, left);
           break;
       /*case GAMMA_I:
           calcDiagptable(z, DNA_DATA, 4, tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN, diagptable);
           partitionLikelihood = evaluateGTRGAMMAINVAR(ex1, ex2, tr->partitionData[model].wgt, tr->partitionData[model].invariant,
								    x1_start, x2_start,
								    tr->partitionData[model].tipVector, tr->partitionData[model].frequencies, 
								    tr->partitionData[model].propInvariant,
								    tip, width, diagptable, tr->useFastScaling);
           break;*/
       default:
           assert(0);
   }
    
   
   
   

   //d_partitionLikelihood h_partitionLikelihood
   partitionLikelihood[thi] = pL + (scalerThread[thi*2*nofSpecies_d + pNumber] + scalerThread[thi*2*nofSpecies_d + qNumber]) *log(minlikelihood_FLOAT);

}
    
    
    
    
    
    
    
    
__device__ void parallelEvaluate()
{
    
    
    
        traversalInfo *tiL=ti;
        
        double *patratL = patrat;
        float **xVectorL = xVector;
        unsigned char **yVectorL = yVector;
        float *EVL = EV;
        int *rateCategoryL = rateCategory;
        float *tipVectorL = tipVector;
        int *wgtL = wgt;
        double *gammaRatesL = gammaRates;
        double *eignL = eign;
        double *eiL = ei;
        unsigned int *scalerThreadL = scalerThread;
    

        double p1, e1, ex1;
   int i;
   int j, k, l;
   __shared__ double d1_s[3], d2_s[3];
   __shared__ float left_s[400], right_s[400], EV_s[16], tipVector_s[64];
   __shared__ double ei_s[12], eign_s[3];

   float left[400];
   
   float
        *x1_start,
	*x2_start,
	*x3_start;  
   int scalerIncrement;
   unsigned char
	*tipX1,
	*tipX2;
  
   double qz, rz;
   traversalInfo *tInfo;
   
   int thi = blockDim.x*blockIdx.x + threadIdx.x;
   
   
   
            //opou 32 to dimBlock.x*.y*.z
      int tx = threadIdx.x, lri, pos;
      boolean step;
      float data_left, data_right;
      
      //ei_s[16], eign_s[3], patrat_s[25];
   
   
/*
   if (thi >= alignLength_d)
       return;
*/
   if (initScalerThread==0)
   {
     for(j=0; j<2*nofSpecies_d; j++)
     {
         scalerThreadL[thi*2*nofSpecies_d + j] =0;
     }
     if (thi==0)
         initScalerThread = 1;
   }
    
   

      
   
         tipVector_s[tx] = tipVectorL[tx];
      tipVector_s[32+tx] = tipVectorL[32+tx];
      
      
          if(tx<16){
              EV_s[tx] = EVL[tx];
              ei_s[tx] = eiL[tx];
              if(tx<3)
                  eign_s[tx] = eignL[tx];
          }                  
      __syncthreads();
   
   
   
   for(i = 1; i < tr->td[0].count; i++)
   {
       tInfo = &tiL[i];

        x1_start = (float*)NULL;
	x2_start = (float*)NULL;
	x3_start = (float*)NULL;  
    
	scalerIncrement = 0;

	tipX1 = (unsigned char *)NULL;
	tipX2 = (unsigned char *)NULL;
         
       switch(tInfo->tipCase)
       {
        case TIP_TIP:
           tipX1    = yVectorL[tInfo->qNumber];
           tipX2    = yVectorL[tInfo->rNumber];
           
           x3_start = xVectorL[tInfo->pNumber - tr->mxtips - 1];
           break;
        case TIP_INNER:
           tipX1    =  yVectorL[tInfo->qNumber];

           x2_start = xVectorL[tInfo->rNumber - tr->mxtips - 1];		 		    		 
	   x3_start = xVectorL[tInfo->pNumber - tr->mxtips - 1];
           break;
        case INNER_INNER:
	   x1_start = xVectorL[tInfo->qNumber - tr->mxtips - 1];		  
	   x2_start = xVectorL[tInfo->rNumber - tr->mxtips - 1];		 
	   x3_start = xVectorL[tInfo->pNumber - tr->mxtips - 1];		  
	   break;
	default:
           assert(0);
       }
       
       qz = tInfo->qz[0];
       rz = tInfo->rz[0];      

       


       
       
       switch(tr->rateHetModel)
       {
           case CAT:
              /*makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);*/
               
               //NumberOfCategories 1 2 25
               
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               //every thread compute its private left and right vectors


               
               for(l=0; l < tr->NumberOfCategories; l++){
                   if(tx<3){
                       d1_s[tx] = exp(patratL[l]*eign_s[tx]*qz);
                       d2_s[tx] = exp(patratL[l]*eign_s[tx]*rz);
                   }
                   __syncthreads();
                   step = (400%32!=0)?1:0;
                   for(lri=0; lri<400/32+step; lri++){
                       pos = lri*32 + tx;
                       if(pos < 400){
                           if(pos%4!=0){
                               data_left = (float) d1_s[pos%4-1]*ei_s[(pos - (pos/4+1))%12];
                               data_right = (float) d2_s[pos%4-1]*ei_s[(pos - (pos/4+1))%12];
                           }
                           else{
                               data_left = 1.0;
                               data_right = 1.0;
                           }
                           left_s[pos] = data_left;
                           right_s[pos] = data_right;
                       }
                   }
                   __syncthreads();
               }

               
               
               /*
               for(l = 0; l < tr->NumberOfCategories; l++)
                  {
                    for(j = 0; j < 3; j++)
                      {
                        p1 = patratL[l];
                        e1 = eignL[j];
                        ex1 = exp(p1 * e1 * qz);
                        d1[j] = ex1;
                        d2[j] = exp(patratL[l] * eignL[j] * rz);
                      }

                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * eiL[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * eiL[3 * j + k]));
                          }
                      }
                  }
                  */
               
              if (thi >= alignLength_d)
                return;
              newviewGTRCAT_FLOAT(tInfo->tipCase, EV_s, rateCategoryL,
					       x1_start, x2_start, x3_start, tipVector_s,
					       tipX1, tipX2,
					       left_s, right_s, wgtL ,&scalerIncrement
					       );

               
/*
              newviewGTRCAT_FLOAT(tInfo->tipCase, EVL, rateCategoryL,
					       x1_start, x2_start, x3_start, tipVectorL,
					       tipX1, tipX2,
					       left, right, wgtL ,&scalerIncrement
					       );
*/

              break;

           case GAMMA:
           case GAMMA_I:
/*
               makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, DNA_DATA);
*/
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               
               for(l=0; l < 4; l++){
                   if(tx<3){
                       d1_s[tx] = exp(patratL[l]*eign_s[tx]*qz);
                       d2_s[tx] = exp(patratL[l]*eign_s[tx]*rz);
                   }
                   __syncthreads();
                   step = (400%32!=0)?1:0;
                   for(lri=0; lri<400/32+step; lri++){
                       pos = lri*32 + tx;
                       if(pos < 400){
                           if(pos%4!=0){
                               data_left = (float) d1_s[pos%4-1]*ei_s[(pos - (pos/4+1))%12];
                               data_right = (float) d2_s[pos%4-1]*ei_s[(pos - (pos/4+1))%12];
                           }
                           else{
                               data_left = 1.0;
                               data_right = 1.0;
                           }
                           left_s[pos] = data_left;
                           right_s[pos] = data_right;
                       }
                   }
                   __syncthreads();
               }               
               
               
               /*
                  for(l = 0; l < 4; l++)
                  {
                    for(j = 0; j < 3; j++)
                      {
                        d1[j] = exp(gammaRatesL[l] * eignL[j] * qz);
                        d2[j] = exp(gammaRatesL[l] * eignL[j] * rz);
                      }
                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;
                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * eiL[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * eiL[3 * j + k]));
                          }
                      }
                  }     
                  */
               
               
               newviewGTRGAMMA_FLOAT_shared(tInfo->tipCase,
						x1_start, x2_start, x3_start, EV_s, tipVector_s,
						tipX1, tipX2,
						left_s, right_s, wgtL, &scalerIncrement);	
              if (thi >= alignLength_d)
                return;                  
/*
               newviewGTRGAMMA_FLOAT(tInfo->tipCase,
						x1_start, x2_start, x3_start, EVL, tipVectorL,
						tipX1, tipX2,
						left, right, wgtL, &scalerIncrement);			
*/
               break;

           default:
               assert(0);
       }


           scalerThreadL[thi*2*nofSpecies_d + tInfo->pNumber] =
                   scalerThreadL[thi*2*nofSpecies_d +tInfo->qNumber] + 
                   scalerThreadL[thi*2*nofSpecies_d +tInfo->rNumber] +
                   scalerIncrement; //!!!       
       
   }
   
   
   /*
    * evaluateIterative
    */
   int pNumber,qNumber;
   double pL;
   unsigned char *tip = (unsigned char*)NULL;

   
   pNumber = tiL[0].pNumber;
   qNumber = tiL[0].qNumber;
   qz = tiL[0].qz[0];
   
   x1_start   = (float*)NULL;
   x2_start   = (float*)NULL;
   
   
   if(isTip(pNumber, tr->mxtips) || isTip(qNumber, tr->mxtips))
   {	        	    
      if(isTip(qNumber, tr->mxtips))
	{	
	  x2_start = xVectorL[pNumber - tr->mxtips -1];
	  tip = yVectorL[qNumber];	 	      
	}           
      else
	{
	  x2_start = xVectorL[qNumber - tr->mxtips - 1];  
	  tip = yVectorL[pNumber];
	}
   }
  else
   {
	x1_start = xVectorL[pNumber - tr->mxtips - 1];
	x2_start = xVectorL[qNumber - tr->mxtips - 1];
   }
   
   
   switch(tr->rateHetModel)
   {
       case CAT:
        //   calcDiagptable_FLOAT(z, DNA_DATA, tr->NumberOfCategories, tr->cdta->patrat, tr->partitionData[model].EIGN, diagptable_FLOAT);
           calcDiagptable_FLOAT(qz, tr->NumberOfCategories, patratL, eign_s, left);
			  
	//   partitionLikelihood =  evaluateGTRCAT_FLOAT(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
	//							      x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT, 
	//							      tip, width, diagptable_FLOAT, tr->useFastScaling);
           pL = evaluateGTRCAT_FLOAT(rateCategoryL, wgtL, x1_start, x2_start, tipVector_s, tip, left);
           
           break;
       case GAMMA:
           //calcDiagptable_FLOAT(z, DNA_DATA, 4, tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN, diagptable_FLOAT);
           calcDiagptable_FLOAT(qz, 4, gammaRatesL, eign_s, left);
        //   partitionLikelihood = evaluateGTRGAMMA_FLOAT(ex1, ex2, tr->partitionData[model].wgt,
	//							       x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
	//							       tip, width, diagptable_FLOAT, tr->useFastScaling);   
           pL = evaluateGTRGAMMA_FLOAT(wgtL, x1_start, x2_start, tipVector_s, tip, left);
           break;

       default:
           assert(0);
   }

   //d_partitionLikelihood h_partitionLikelihood
   partitionLikelihood[thi] = pL + (scalerThreadL[thi*2*nofSpecies_d + pNumber] + scalerThreadL[thi*2*nofSpecies_d + qNumber]) *log(minlikelihood_FLOAT);

}    
    
    
    
    
    
    
    
    
    

__global__ void startKernelNewview(
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories)
{
    //if (threadIdx.x==1)
        //if (d_tree->td[0].count > 2)
            //printf("count = %d\n", d_tree->td[0].count);
    
    //traversalInfo *ti = d_ti;
   double p1, e1, ex1;
   int i;
   int j, k, l;
   double d1[3], d2[3];
   float left[400], right[400];
   //float *left, *right;
   //left = d_left + threadIdx.x*400;
   //right = d_right + threadIdx.x*400;

   float
        *x1_start,
	*x2_start,
	*x3_start;  
    
   //int
	//states = tr->partitionData[model].states,
	int scalerIncrement;
	//*wgt = (int*)NULL,	       
	//*ex3 = (int*)NULL;
    
   unsigned char
	*tipX1,
	*tipX2;
  
   double qz, rz;
   //int width =  tr->partitionData[model].width;    
    
   //traversalInfo *ti   = tr->td[0].ti; //ok
   traversalInfo *tInfo;
   //int tx = threadIdx.x;
   
   int thi = blockDim.x*blockIdx.x + threadIdx.x;
   if (thi >= alignLength_d)
       return;
   
   /*
   if (countr==0)
   {
   //if (tr->td[0].count>1){
     for(j=0; j<2*nofSpecies; j++)
     {
         scalerThread[thi*2*nofSpecies + j] =0;// globalScaler[j];
     }
   //}
   }
    */
   //tr_td0_count = tr->td[0].count; //tr changed!
   
   for(i = 1; i < tr_td0_count; i++)
   {
       tInfo = &ti[i];
       
       
        
        x1_start = (float*)NULL;
	x2_start = (float*)NULL;
	x3_start = (float*)NULL;  
    
	scalerIncrement = 0;

	tipX1 = (unsigned char *)NULL;
	tipX2 = (unsigned char *)NULL;
       
       
       
       
       switch(tInfo->tipCase)
       {
        case TIP_TIP:
           tipX1    = yVector[tInfo->qNumber];
           tipX2    = yVector[tInfo->rNumber];
           
           x3_start = xVector[tInfo->pNumber - tr_mxtips - 1];
           break;
        case TIP_INNER:
           tipX1    =  yVector[tInfo->qNumber];

           x2_start = xVector[tInfo->rNumber - tr_mxtips - 1];		 		    		 
	   x3_start = xVector[tInfo->pNumber - tr_mxtips - 1];
           break;
        case INNER_INNER:
	   x1_start = xVector[tInfo->qNumber - tr_mxtips - 1];		  
	   x2_start = xVector[tInfo->rNumber - tr_mxtips - 1];		 
	   x3_start = xVector[tInfo->pNumber - tr_mxtips - 1];		  
	   break;
	default:
           assert(0);
       }
       
       //left_FLOAT = tr->partitionData[model].left_FLOAT; //OK
       //right_FLOAT = tr->partitionData[model].right_FLOAT;  //OK
       // (*tInfo).qz[0];  (*tInfo).*(qz + 0);
       qz = tInfo->qz[0];
       rz = tInfo->rz[0];      

       //case DNA_DATA:
       
       switch(tr_rateHetModel)
       {
           case CAT:
              /*makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);*/
               
               //NumberOfCategories 1 2 25
               
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               //every thread compute its private left and right vectors
                  for(l = 0; l < tr_NumberOfCategories; l++)
                  {

                    for(j = 0; j < 3; j++)
                      {
                        p1 = patrat[l];
                        e1 = eign[j];
                        ex1 = exp(p1 * e1 * qz);
                        d1[j] = ex1;
                        d2[j] = exp(p1 * e1 * rz);
                      }


                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * ei[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * ei[3 * j + k]));
                            //left[l * 16 + j * 4 + k + 1]  = ((float)( ei[3 * j + k]));
                            //right[l * 16 + j * 4 + k + 1] = ((float)( ei[3 * j + k]));
                          }

                      }

                  }
                  
                

#ifndef notdef
              newviewGTRCAT_FLOAT(tInfo->tipCase, EV, rateCategory,
					       x1_start, x2_start, x3_start, tipVector,
					       tipX1, tipX2,
					       left, right, wgt ,&scalerIncrement
					       );
#endif
              break;

           case GAMMA:
           case GAMMA_I:
/*
               makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, DNA_DATA);
*/
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               
                  for(l = 0; l < 4; l++)
                  {

                    for(j = 0; j < 3; j++)
                      {
                        d1[j] = exp(gammaRates[l] * eign[j] * qz);
                        d2[j] = exp(gammaRates[l] * eign[j] * rz);
                      }


                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * ei[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * ei[3 * j + k]));
                            //left[l * 16 + j * 4 + k + 1]  = ((float)( ei[3 * j + k]));
                            //right[l * 16 + j * 4 + k + 1] = ((float)( ei[3 * j + k]));
                          }

                      }

                  }               

#ifndef notdef
                  
               newviewGTRGAMMA_FLOAT(tInfo->tipCase,
						x1_start, x2_start, x3_start, EV, tipVector,
						tipX1, tipX2,
						left, right, wgt, &scalerIncrement);			
#endif
               break;
           default:
               assert(0);
       }


#ifndef notdef
       //if(tr->useFastScaling)
        //{
       /*
       if(thi == 0)
       {
           globalScaler[tInfo->pNumber] =
                   globalScaler[tInfo->qNumber] + 
                   globalScaler[tInfo->rNumber] +
                   scalerIncrement[thi]; //!!!
           assert(globalScaler[tInfo->pNumber] < INT_MAX);
       }
       else
       {
           atomicAdd(&(globalScaler[tInfo->pNumber]), scalerIncrement[thi]);
       }
       */
       /*ver2 wrong*/

           scalerThread[thi*2*nofSpecies_d + tInfo->pNumber] =
                   scalerThread[thi*2*nofSpecies_d +tInfo->qNumber] + 
                   scalerThread[thi*2*nofSpecies_d +tInfo->rNumber] +
                   scalerIncrement; //!!!       
       
       /* ver1 wrong
       if(scaleNum == 2048)
       {
           globalScaler[tInfo->pNumber] =
                   globalScaler[tInfo->qNumber] + 
                   globalScaler[tInfo->rNumber] +
                   scalerIncrement; //!!!
           assert(globalScaler[tInfo->pNumber] < INT_MAX);
       }
        */
           /* serial code:
            * if(tr->useFastScaling)
		{
		  tr->partitionData[model].globalScaler[tInfo->pNumber] = 
		    tr->partitionData[model].globalScaler[tInfo->qNumber] + 
		    tr->partitionData[model].globalScaler[tInfo->rNumber] +
		    (unsigned int)scalerIncrement;
		  assert(tr->partitionData[model].globalScaler[tInfo->pNumber] < INT_MAX);
		}
            */
	//}
#endif
   }
   
}
        
    
    
    
#ifdef out
  
 /*getVects_FLOAT(tr, &tipX1, &tipX2, &x1_start, &x2_start, &tipCase);*/
static void getVects_FLOAT(tree *tr, unsigned char **tipX1, unsigned char **tipX2, float **x1_start, float **x2_start, int *tipCase)
{
  int pNumber = tr->td[0].ti[0].pNumber;
  int qNumber = tr->td[0].ti[0].qNumber;

  *x1_start = (float*)NULL,
  *x2_start = (float*)NULL;
  *tipX1 = (unsigned char*)NULL,
  *tipX2 = (unsigned char*)NULL;

  if(isTip(pNumber, tr->mxtips) || isTip(qNumber, tr->mxtips))
    {
      if(!( isTip(pNumber, tr->mxtips) && isTip(qNumber, tr->mxtips)) )
	{
	  *tipCase = TIP_INNER;
	  if(isTip(qNumber, tr->mxtips))
	    {
	      *tipX1 = tr->partitionData[model].yVector[qNumber];
	      *x2_start = tr->partitionData[model].xVector_FLOAT[pNumber - tr->mxtips - 1];

	    }
	  else
	    {
	      *tipX1 = tr->partitionData[model].yVector[pNumber];
	      *x2_start = tr->partitionData[model].xVector_FLOAT[qNumber - tr->mxtips - 1];
	    }
	}
      else
	{
	  *tipCase = TIP_TIP;
	  *tipX1 = tr->partitionData[model].yVector[pNumber];
	  *tipX2 = tr->partitionData[model].yVector[qNumber];
	}
    }
  else
    {
      *tipCase = INNER_INNER;

      *x1_start = tr->partitionData[model].xVector_FLOAT[pNumber - tr->mxtips - 1];
      *x2_start = tr->partitionData[model].xVector_FLOAT[qNumber - tr->mxtips - 1];
    }

}

#endif


/*sumCAT_FLOAT(tipCase, tr->partitionData[model].sumBuffer_FLOAT, x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
				 tipX1, tipX2, width);*/
/*sumCAT_FLOAT(tipCase, sumBuffer, x1_start, x2_start, tipVector,
				 tipX1, tipX2);*/
__device__ void sumCAT_FLOAT(int tipCase, float *sum, float *x1_start, float *x2_start, float *tipVector,
			 unsigned char *tipX1, unsigned char *tipX2)
{
  int i, j;
  float *x1, *x2;

  i = blockDim.x*blockIdx.x + threadIdx.x; 
  switch(tipCase)
    {
    case TIP_TIP:
      //for (i = 0; i < n; i++)
	//{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);

	  for(j = 0; j < 4; j++)
	    sum[i * 4 + j]     = x1[j] * x2[j];
	//}
      break;
    case TIP_INNER:
      //for (i = 0; i < n; i++)
	//{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &x2_start[4 * i];

	  for(j = 0; j < 4; j++)
	    sum[i * 4 + j]     = x1[j] * x2[j];
	//}
      break;
    case INNER_INNER:
      //for (i = 0; i < n; i++)
	//{
	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];

	  for(j = 0; j < 4; j++)
	    sum[i * 4 + j]     = x1[j] * x2[j];
	//}
      break;
    default:
      assert(0);
    }
}


/*sumGAMMA_FLOAT(tipCase, tr->partitionData[model].sumBuffer_FLOAT, x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
				    tipX1, tipX2, width);*/
/*sumGAMMA_FLOAT(tipCase, sumBuffer, x1_start, x2_start, tipVector,
				    tipX1, tipX2);*/
__device__ void sumGAMMA_FLOAT(int tipCase, float *sumtable, float *x1_start, float *x2_start, float *tipVector,
			   unsigned char *tipX1, unsigned char *tipX2)
{
  float *x1, *x2, *sum;
  int i, j, k;
  
  i = blockDim.x*blockIdx.x + threadIdx.x; 

  switch(tipCase)
    {
    case TIP_TIP:      
      //for (i = 0; i < n; i++)
	//{

	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);
	  sum = &sumtable[i * 16];

	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[k] * x2[k];

	//}
      break;
    case TIP_INNER:
      //for (i = 0; i < n; i++)
	//{

	  x1  = &(tipVector[4 * tipX1[i]]);
	  x2  = &x2_start[16 * i];
	  sum = &sumtable[16 * i];

	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[k] * x2[j * 4 + k];

	//}
      break;
    case INNER_INNER:
      //for (i = 0; i < n; i++)
	//{

	  x1  = &x1_start[16 * i];
	  x2  = &x2_start[16 * i];
	  sum = &sumtable[16 * i];

	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[j * 4 + k] * x2[j * 4 + k];

	//}
      break;
    default:
      assert(0);
    }
}






/*coreGTRCAT_FLOAT(width, tr->NumberOfCategories, sumBuffer_FLOAT,
				       &dlnLdlz, &d2lnLdlz2, tr->partitionData[model].wr_FLOAT, tr->partitionData[model].wr2_FLOAT,
				       tr->cdta->patrat, tr->partitionData[model].EIGN,  tr->partitionData[model].rateCategory, lz);*/

/*coreGTRCAT_FLOAT(tr->NumberOfCategories, sumBuffer,
				dlnLdlz, d2lnLdlz2, wr, wr2,
				patrat, eign, rateCategory, lz);*/
__device__ void coreGTRCAT_FLOAT(float *d_tmpSpace, int numberOfCategories, float *sum,
			     double *d1, double *d2, float *wrptr, float *wr2ptr,
			     double *rptr, double *EIGN, int *cptr, double lz)
//d_tmpSpace 25*4*sizeof(float)*nofThreads space for every thread to avoid malloc on GPU
// e.g: 100*4*4096 = 1.6MB
{
  int i;
  float
    *d, *d_start,
    tmp_0, tmp_1, tmp_2, inv_Li, dlnLidlz, d2lnLidlz2,
    dlnLdlz = 0.0,
    d2lnLdlz2 = 0.0,
    ef[6];
  double e[6];
  double dd1, dd2, dd3;
  int thi;
  
  e[0] = EIGN[0];
  e[1] = EIGN[0] * EIGN[0];
  e[2] = EIGN[1];
  e[3] = EIGN[1] * EIGN[1];
  e[4] = EIGN[2];
  e[5] = EIGN[2] * EIGN[2];

  for(i = 0; i < 6; i++)
    ef[i] = (float)e[i];

  //d = d_start = (float *)malloc(numberOfCategories * 4 * sizeof(float));
  thi = blockDim.x*blockIdx.x + threadIdx.x; 
  
  d_start = d_tmpSpace + thi*100; //every thread start point
  d = d_start;
  
  dd1 = e[0] * lz;
  dd2 = e[2] * lz;
  dd3 = e[4] * lz;

  for(i = 0; i < numberOfCategories; i++)
    {
      d[i * 4]     = ((float)EXP(dd1 * rptr[i]));
      d[i * 4 + 1] = ((float)EXP(dd2 * rptr[i]));
      d[i * 4 + 2] = ((float)EXP(dd3 * rptr[i]));
    }

  //for (i = 0; i < upper; i++)
    //{
      d = &d_start[4 * cptr[thi]];

      inv_Li = sum[4 * thi];
      inv_Li += (tmp_0 = d[0] * sum[4 * thi + 1]);
      inv_Li += (tmp_1 = d[1] * sum[4 * thi + 2]);
      inv_Li += (tmp_2 = d[2] * sum[4 * thi + 3]);

      inv_Li = 1.0/inv_Li;

      dlnLidlz   = tmp_0 * ef[0];
      d2lnLidlz2 = tmp_0 * ef[1];

      dlnLidlz   += tmp_1 * ef[2];
      d2lnLidlz2 += tmp_1 * ef[3];

      dlnLidlz   += tmp_2 * ef[4];
      d2lnLidlz2 += tmp_2 * ef[5];

      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;


      dlnLdlz   += wrptr[thi] * dlnLidlz;
      d2lnLdlz2 += wr2ptr[thi] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    //}

  d1[thi] = (double)dlnLdlz;
  d2[thi] = (double)d2lnLdlz2;

  //free(d_start);
}





/*coreGTRGAMMA_FLOAT(width, sumBuffer_FLOAT,
					 &dlnLdlz, &d2lnLdlz2, tr->partitionData[model].EIGN, tr->partitionData[model].gammaRates, lz,
					 tr->partitionData[model].wgt);*/
/*coreGTRGAMMA_FLOAT(sumBuffer,
                     dlnLdlz, d2lnLdlz2, eign, gammaRates, lz,
                     wgt);*/
__device__ void coreGTRGAMMA_FLOAT(float *d_tmpSpace, float *sumtable,
			       double *d1, double *d2, double *EIGN, double *gammaRates, double lz, 
                               int *wrptr)
//d_tmpspace 64*sizeof(float)*nofthreads
{
  int i, j, k;
  double ki, kisqr;

  float
    *diagptable, *diagptable_start, *sum,
    tmp_1, inv_Li, dlnLidlz, d2lnLidlz2,
    dlnLdlz = 0.0,
    d2lnLdlz2 = 0.0;

  int thi;
  //diagptable = diagptable_start = (float *)malloc(sizeof(float) * 64);
  
  thi = blockDim.x*blockIdx.x + threadIdx.x;
  
  diagptable_start = d_tmpSpace + 64*thi;
  diagptable = diagptable_start;
  
  for(i = 0; i < 4; i++)
    {
      ki = gammaRates[i];
      kisqr = ki * ki;

      diagptable[i * 16]     = ((float)EXP (EIGN[0] * ki * lz));
      diagptable[i * 16 + 1] = ((float)EXP (EIGN[1] * ki * lz));
      diagptable[i * 16 + 2] = ((float)EXP (EIGN[2] * ki * lz));

      diagptable[i * 16 + 3] = ((float)(EIGN[0] * ki));
      diagptable[i * 16 + 4] = ((float)(EIGN[0] * EIGN[0] * kisqr));

      diagptable[i * 16 + 5] = ((float)(EIGN[1] * ki));
      diagptable[i * 16 + 6] = ((float)(EIGN[1] * EIGN[1] * kisqr));

      diagptable[i * 16 + 7] = ((float)(EIGN[2] * ki));
      diagptable[i * 16 + 8] = ((float)(EIGN[2] * EIGN[2] * kisqr));
    }  

  //for (i = 0; i < upper; i++)
    //{
      diagptable = diagptable_start;
      sum = &(sumtable[thi * 16]);

      inv_Li      = 0.0;
      dlnLidlz    = 0.0;
      d2lnLidlz2  = 0.0;

      for(j = 0; j < 4; j++)
	{
	  inv_Li += sum[4 * j];

	  for(k = 0; k < 3; k++)
	    {
	      tmp_1      =  diagptable[16 * j + k] * sum[4 * j + k + 1];
	      inv_Li     += tmp_1;
	      dlnLidlz   += tmp_1 * diagptable[16 * j + k * 2 + 3];
	      d2lnLidlz2 += tmp_1 * diagptable[16 * j + k * 2 + 4];
	    }
	}

      inv_Li = 1.0 / inv_Li;

      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;



      dlnLdlz  += wrptr[thi] * dlnLidlz;
      d2lnLdlz2 += wrptr[thi] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    //}

  d1[thi] = (double)dlnLdlz;
  d2[thi] = (double)d2lnLdlz2;

  //free(diagptable_start);
}



__device__ void parallelNewView(
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories)
{
    
        traversalInfo *tiL=ti;
        
        double *patratL = patrat;
        float **xVectorL = xVector;
        unsigned char **yVectorL = yVector;
        float *EVL = EV;
        int *rateCategoryL = rateCategory;
        float *tipVectorL = tipVector;
        int *wgtL = wgt;
        double *gammaRatesL = gammaRates;
        double *eignL = eign;
        double *eiL = ei;
        unsigned int *scalerThreadL = scalerThread;
    
    
  float
        *x1_start,
	*x2_start,
	*x3_start;  
  unsigned char
	*tipX1,
	*tipX2;

  int thi = blockDim.x*blockIdx.x + threadIdx.x;
  
   if (thi >= alignLength_d)
      return;
  
//newview Start
   double p1, e1, ex1;
   int i;
   int j, k, l;
   double d1[3], d2[3];
   float left[400], right[400];
   //float *left, *right;
   //left = d_left + threadIdx.x*400;
   //right = d_right + threadIdx.x*400;

   int scalerIncrement;
  
   double qz, rz;

   traversalInfo *tInfo;
    
   for(i = 1; i < tr_td0_count; i++)
   {
       tInfo = &tiL[i];
              
        x1_start = (float*)NULL;
	x2_start = (float*)NULL;
	x3_start = (float*)NULL;  
    
	scalerIncrement = 0;

	tipX1 = (unsigned char *)NULL;
        tipX2 = (unsigned char *)NULL;
       
       switch(tInfo->tipCase)
       {
        case TIP_TIP:
           tipX1    = yVectorL[tInfo->qNumber];
           tipX2    = yVectorL[tInfo->rNumber];
           
           x3_start = xVectorL[tInfo->pNumber - tr_mxtips - 1];
           break;
        case TIP_INNER:
           tipX1    =  yVectorL[tInfo->qNumber];

           x2_start = xVectorL[tInfo->rNumber - tr_mxtips - 1];		 		    		 
	   x3_start = xVectorL[tInfo->pNumber - tr_mxtips - 1];
           break;
        case INNER_INNER:
	   x1_start = xVectorL[tInfo->qNumber - tr_mxtips - 1];		  
	   x2_start = xVectorL[tInfo->rNumber - tr_mxtips - 1];		 
	   x3_start = xVectorL[tInfo->pNumber - tr_mxtips - 1];		  
	   break;
	default:
           assert(0);
       }
       

       qz = tInfo->qz[0];
       rz = tInfo->rz[0];      


       
       switch(tr_rateHetModel)
       {
           case CAT:
              /*makeP_FLOAT(qz, rz, tr->cdta->patrat,   tr->partitionData[model].EI,
				       tr->partitionData[model].EIGN, tr->NumberOfCategories,
				       left_FLOAT, right_FLOAT, DNA_DATA);*/
               
               //NumberOfCategories 1 2 25
               
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               //every thread compute its private left and right vectors
              for(l = 0; l < tr_NumberOfCategories; l++)
              {

                    for(j = 0; j < 3; j++)
                      {
                        p1 = patratL[l];
                        e1 = eignL[j];
                        ex1 = exp(p1 * e1 * qz);
                        d1[j] = ex1;
                        d2[j] = exp(patratL[l] * eignL[j] * rz);
                      }


                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * eiL[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * eiL[3 * j + k]));
                            //left[l * 16 + j * 4 + k + 1]  = ((float)( ei[3 * j + k]));
                            //right[l * 16 + j * 4 + k + 1] = ((float)( ei[3 * j + k]));
                          }

                      }

              }     
              newviewGTRCAT_FLOAT(tInfo->tipCase, EVL, rateCategoryL,
					       x1_start, x2_start, x3_start, tipVectorL,
					       tipX1, tipX2,
					       left, right, wgtL ,&scalerIncrement
					       );
              break;

           case GAMMA:
           case GAMMA_I:
/*
               makeP_FLOAT(qz, rz, tr->partitionData[model].gammaRates,
				      tr->partitionData[model].EI, tr->partitionData[model].EIGN,
				      4, left_FLOAT, right_FLOAT, DNA_DATA);
*/
               //static void makeP_FLOAT(double z1, double z2, double *rptr, double *EI,  double *EIGN, int numberOfCategories, float *left, float *right, int data)
               
               for(l = 0; l < 4; l++)
               {

                    for(j = 0; j < 3; j++)
                      {
                        d1[j] = exp(gammaRatesL[l] * eignL[j] * qz);
                        d2[j] = exp(gammaRatesL[l] * eignL[j] * rz);
                      }


                    for(j = 0; j < 4; j++)
                      {
                        left[l * 16 + j * 4] = 1.0;
                        right[l * 16 + j * 4] = 1.0;

                        for(k = 0; k < 3; k++)
                          {
                            left[l * 16 + j * 4 + k + 1]  = ((float)(d1[k] * eiL[3 * j + k]));
                            right[l * 16 + j * 4 + k + 1] = ((float)(d2[k] * eiL[3 * j + k]));
                            //left[l * 16 + j * 4 + k + 1]  = ((float)( ei[3 * j + k]));
                            //right[l * 16 + j * 4 + k + 1] = ((float)( ei[3 * j + k]));
                          }

                      }

               }                  
               newviewGTRGAMMA_FLOAT(tInfo->tipCase,
						x1_start, x2_start, x3_start, EVL, tipVectorL,
						tipX1, tipX2,
						left, right, wgtL, &scalerIncrement);			
               break;
           default:
               assert(0);
       }

       scalerThreadL[thi*2*nofSpecies_d + tInfo->pNumber] =
                scalerThreadL[thi*2*nofSpecies_d +tInfo->qNumber] + 
                scalerThreadL[thi*2*nofSpecies_d +tInfo->rNumber] +
                scalerIncrement; //!!!       
       

   }
   
  //newview END
}





__device__ void parallelSum(
        int tr_mxtips,
        int tr_rateHetModel,

        int pIsTip,
        int qIsTip)
{
    
    
        traversalInfo *tiL = ti;
        float *tipVectorL = tipVector;

        float **xVectorL = xVector;
        unsigned char **yVectorL = yVector;
        float *sumBufferL = sumBuffer;
    
        
        
  float
        *x1_start,
	*x2_start;  
  unsigned char
	*tipX1,
	*tipX2;

  
  //sum START
  int tipCase;
  
  x1_start = (float*)NULL,
  x2_start = (float*)NULL;

  
  pIsTip = isTip(tiL[0].pNumber, tr_mxtips);
          
  qIsTip = isTip(tiL[0].qNumber, tr_mxtips);
  //getVects_FLOAT(tr, &tipX1, &tipX2, &x1_start_FLOAT, &x2_start_FLOAT, &tipCase, model); //cpu
  //getVects_FLOAT(tr, &tipX1, &tipX2, &x1_start, &x2_start, &tipCase); //gpu
  //telika ylopoihsh ths getVects ektos synarthshs:
  if(pIsTip || qIsTip)
    {
      if(!( pIsTip && qIsTip) )
	{
	  tipCase = TIP_INNER;
	  if(qIsTip)
	    {
	      tipX1 = yVectorL[tiL[0].qNumber];
	      x2_start = xVectorL[tiL[0].pNumber - tr_mxtips - 1];

	    }
	  else
	    {
	      tipX1 = yVectorL[tiL[0].pNumber];
	      x2_start = xVectorL[tiL[0].qNumber - tr_mxtips - 1];
	    }
	}
      else
	{
	  tipCase = TIP_TIP;
	  tipX1 = yVectorL[tiL[0].pNumber];
	  tipX2 = yVectorL[tiL[0].qNumber];
	}
    }
  else
    {
      tipCase = INNER_INNER;

      x1_start = xVectorL[tiL[0].pNumber - tr_mxtips - 1];
      x2_start = xVectorL[tiL[0].qNumber - tr_mxtips - 1];
    }
  

    switch(tr_rateHetModel)
    {
	case CAT:
                /*
                 sumCAT_FLOAT(tipCase, tr->partitionData[model].sumBuffer_FLOAT, x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
                                tipX1, tipX2, width);
                  */
                 sumCAT_FLOAT(tipCase, sumBufferL, x1_start, x2_start, tipVectorL,
				tipX1, tipX2);
	break;
	case GAMMA:
	case GAMMA_I:
		  /*sumGAMMA_FLOAT(tipCase, tr->partitionData[model].sumBuffer_FLOAT, x1_start_FLOAT, x2_start_FLOAT, tr->partitionData[model].tipVector_FLOAT,
			    tipX1, tipX2, width);*/
                 sumGAMMA_FLOAT(tipCase, sumBufferL, x1_start, x2_start, tipVectorL,
				 tipX1, tipX2);

	break;
	default:
                assert(0);
    }
}



__device__ void parallelExecCore(
        double *tr_coreLZ0,
        int tr_rateHetModel,
        int tr_NumberOfCategories)
{
  double lz;
  

  lz = *tr_coreLZ0;
  
  
  switch(tr_rateHetModel)
  {
	case CAT:
		/*coreGTRCAT_FLOAT(width, tr->NumberOfCategories, sumBuffer_FLOAT,
				&dlnLdlz, &d2lnLdlz2, tr->partitionData[model].wr_FLOAT, tr->partitionData[model].wr2_FLOAT,
				tr->cdta->patrat, tr->partitionData[model].EIGN,  tr->partitionData[model].rateCategory, lz);*/
            coreGTRCAT_FLOAT(tmpCatSpace, tr_NumberOfCategories, sumBuffer,
				dlnLdlz, d2lnLdlz2, wr, wr2,
				patrat, eign, rateCategory, lz);
            
	break;
        case GAMMA:
                /*coreGTRGAMMA_FLOAT(width, sumBuffer_FLOAT,
                                  &dlnLdlz, &d2lnLdlz2, tr->partitionData[model].EIGN, tr->partitionData[model].gammaRates, lz,
                                  tr->partitionData[model].wgt);*/
            coreGTRGAMMA_FLOAT(tmpDiagSpace, sumBuffer,
                               dlnLdlz, d2lnLdlz2, eign, gammaRates, lz,
                               wgt);
	break;
	default:
                assert(0);
  }    
}



    
__device__ void parallelNewzCore( //ousiastika einai o arxikos kernel pou eixa ylopoihsei san kernelNewzCore
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories,
        double *tr_coreLZ0,
        int pIsTip,
        int qIsTip,
        int firstIteration) 
{

 int thi = blockDim.x*blockIdx.x + threadIdx.x;
 if (thi >= alignLength_d)
      return;
  
 if (firstIteration)
 {
   //prepei na ektelestoun newviewIterative, sumCAT, sumGAMMA
   //me afthn thn seira
     
   parallelNewView(
           tr_td0_count,
           tr_mxtips,
           tr_rateHetModel,
           tr_NumberOfCategories);
   
   parallelSum(
            tr_mxtips,
            tr_rateHetModel,
           
            pIsTip,
            qIsTip
          );

 } 

  
 //ektelesh execCore PANTA
 //void execCore(tree *tr, volatile double *_dlnLdlz, volatile double *_d2lnLdlz2)

 parallelExecCore(
         tr_coreLZ0,
         tr_rateHetModel,
         tr_NumberOfCategories);
  

}
 
/*
   startKernelNewviewSumCore<<<dimGrid, dimBlock>>>(
             alignLength, nofSpecies, 
             d_scalerThread, d_EV, d_tipVector, d_gammaRates, 
             d_patrat, d_ei, d_eign, d_rateCategory, d_wgt, d_tree, d_ti, d_xVector, d_yVector);
 */
__global__ void startKernelNewzCore(
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories,
        double tr_coreLZ0,
        int pIsTip,
        int qIsTip,
        int firstIteration
        )
{

        //tr_td0_count = tr->td[0].count; //tr changed!
       parallelNewzCore(        
         tr_td0_count,
         tr_mxtips,
         tr_rateHetModel,
         tr_NumberOfCategories,
         &tr_coreLZ0,
         
         pIsTip,
         qIsTip,
         firstIteration
         );
       
 

}    


__device__ void parallelExecution(
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories,
        double *tr_coreLZ0,
        int pIsTip,
        int qIsTip,
        int firstIteration
)

{
    //mainly waits for master thread (not strictly)
    globalBarrier(); 
    
    if (chkLast)
        globalBarrier();     
    
    //check if device is done. Return to host
    if (!endGPUexecution){
        //choose kernel!
        switch (execKernel){            
            case NEWZCORE:
            {
                  parallelNewzCore(        
                         tr->td[0].count, //tr changed!
                         tr_mxtips,
                         tr_rateHetModel,
                         tr_NumberOfCategories,
                         tr_coreLZ0,
                         pIsTip,
                         qIsTip,
                         firstIteration_mw);
                break;
            }
            case NEWVIEW:
            {
                   parallelNewView(
                       tr->td[0].count,
                       tr_mxtips,
                       tr_rateHetModel,
                       tr_NumberOfCategories);
                break;
            }
            case EVALUATE:
            {
                parallelEvaluate();
                break;
            }
            default:
                assert(0);

        }
        globalBarrier(); //calculations done! for all device threads
    }
    
}



    
/*
 * barrier for master thread!
 * master thread doesn't need the __syncthreads()
 * 
 */
__device__ void cds()
{
 int nofBlocks = gridDim.x*gridDim.y*gridDim.z;
 int mySense = !b.sense;

  
 //if (threadIdx.x == 0){ 
  int old = atomicAdd((int *)&b.blockFinish, 1);

  if (old == nofBlocks-1){
	b.blockFinish = 0;
	b.sense = !b.sense;
  }
  else
	while (mySense != b.sense);
//}
 //__syncthreads();

}






/*
 * START path functions
 * 
 */

__device__ void endExecKernel()
{
  chkLast = 1;
  cds();
  endGPUexecution = TRUE;
  cds();
  b.blockFinish = 0;
}


__device__ void execWorkers(int kernel)
{
    execKernel = kernel;
          
    cds(); //start execution
          
    cds(); //end execution
    
}

__device__ void newviewGeneric (tree *tr, nodeptr p)
{  
  if(isTip(p->number, tr->mxtips))
    return;



      tr->td[0].count = 1;
      computeTraversalInfo(p, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
      
      if(tr->td[0].count > 1)
	{
	  //newviewIterative(tr);
          execWorkers(NEWVIEW);

	}
}


__device__ double evaluateGeneric (tree *tr, nodeptr p)
{
  double result;
  nodeptr q = p->back; 
  int i;
  

      ti[0].pNumber = p->number;
      ti[0].qNumber = q->number;          
  
      for(i = 0; i < tr->numBranches; i++)    
	ti[0].qz[i] =  q->z[i];
  
      tr->td[0].count = 1;
      if(!p->x)
	computeTraversalInfo(p, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
      if(!q->x)
	computeTraversalInfo(q, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);  
      

      
      //result = evaluateIterative(tr, FALSE);

      execWorkers(EVALUATE);
      result = 0.0;
      for(i=0; i<alignLength_d; i++)
          result += partitionLikelihood[i]; 

      tr->likelihood = result;    
      tr->perPartitionLH[0] = result;

  

  return result;
}








__device__ void topLevelMakenewz(double z0, int _maxiter, double *tr_coreLZ0)
{
    //topLevelMakenewz()  
  double   z[NUM_BRANCHES_GPU], zprev[NUM_BRANCHES_GPU], zstep[NUM_BRANCHES_GPU];
  double  dlnLdlz_local[NUM_BRANCHES_GPU], d2lnLdlz2_local[NUM_BRANCHES_GPU];
  int i, maxiter[NUM_BRANCHES_GPU], numBranches;
  
  boolean outerConverged[NUM_BRANCHES_GPU];
  boolean loopConverged;
  int tr_curvatOK, tr_executeModel;
   
  firstIteration_mw = TRUE;
  numBranches = 1;
  
 //tree *tr, double *z0=>z0, int _maxiter, double *result 

    for(i = 0; i < numBranches; i++)
    {
      z[i] = z0;
      maxiter[i] = _maxiter;
      outerConverged[i] = FALSE;
      tr_curvatOK     = TRUE;
    }

    
    do{  
      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr_curvatOK == TRUE)
	    {
	      tr_curvatOK = FALSE;

	      zprev[i] = z[i];

	      zstep[i] = (1.0 - zmax) * z[i] + zmin;
	    }
	}

      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr_curvatOK == FALSE)
	    {
	      double lz;

	      if (z[i] < zmin) z[i] = zmin;
	      else if (z[i] > zmax) z[i] = zmax;
	      lz    = log(z[i]);

	      *tr_coreLZ0 = lz;
	    }
	}

	  //for(model = 0; model < tr->NumberOfModels; model++)
	    //tr->executeModel[model] = !tr->curvatOK[0];
	tr_executeModel = !tr_curvatOK;

        

        if (tr_executeModel){
            execWorkers(NEWZCORE);
        }
        
        if (firstIteration_mw)              
            firstIteration_mw = FALSE;

          //reduction GPU results
          dlnLdlz_local[0] = 0.0;
          d2lnLdlz2_local[0] = 0.0;
          for(i = 0; i < alignLength_d; i++)
          {
            dlnLdlz_local[0]   += dlnLdlz[i];
            d2lnLdlz2_local[0] += d2lnLdlz2[i];
          }      
      
      //printf("dev reduction dlnLdlz: %f\n", dlnLdlz[0]);
      //printf("dev reduction d2lnLdlz2: %f\n", d2lnLdlz2[0]);
      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr_curvatOK == FALSE)
	    {
	      if ((d2lnLdlz2_local[i] >= 0.0) && (z[i] < zmax))
		zprev[i] = z[i] = 0.37 * z[i] + 0.63;  /*  Bad curvature, shorten branch */
	      else
		tr_curvatOK = TRUE;
	    }
	}

      for(i = 0; i < numBranches; i++)
	{
	  if(tr_curvatOK == TRUE && outerConverged[i] == FALSE)
	    {
	      if (d2lnLdlz2_local[i] < 0.0)
		{
		  double tantmp = -dlnLdlz_local[i] / d2lnLdlz2_local[i];
		  if (tantmp < 100)
		    {
		      z[i] *= EXP(tantmp);
		      if (z[i] < zmin)
			z[i] = zmin;

		      if (z[i] > 0.25 * zprev[i] + 0.75)
			z[i] = 0.25 * zprev[i] + 0.75;
		    }
		  else
		    z[i] = 0.25 * zprev[i] + 0.75;
		}
	      if (z[i] > zmax) z[i] = zmax;

	      maxiter[i] = maxiter[i] - 1;
	      if(maxiter[i] > 0 && (ABS(z[i] - zprev[i]) > zstep[i]))
		outerConverged[i] = FALSE;
	      else
		outerConverged[i] = TRUE;
	    }
	}

      loopConverged = TRUE;
      for(i = 0; i < numBranches; i++)
	loopConverged = loopConverged && outerConverged[i];
     
    }
  while (!loopConverged);
    
  
  /*chkLast = 1;
  cds();
  endGPUexecution = TRUE;
  cds();
  
 
  b.blockFinish = 0;*/
  tr_executeModel = TRUE;  

  dlnLdlz[0] = z[0]; //globalize result, send result to host

  //printf("thread0 res: %f\n", d_dlnLdlz[0]);
  //printf("thread 0 finished!\n");

}



__device__ void getxnode (nodeptr p)
{
  nodeptr  s;

  if ((s = p->next)->x || (s = s->next)->x)
    {
      p->x = s->x;
      s->x = 0;
    }

  assert(p->x);
}


__device__ void computeTraversalInfo(nodeptr p, traversalInfo *ti, int *counter, int maxTips, int numBranches)
{
  if(isTip(p->number, maxTips))
    return;

  
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

    

//makenewzGeneric(tr, p, q, z0, maxiter, 0, tr_coreLZ0, d_dlnLdlz, d_d2lnLdlz2, alignLength, ti));
__device__ void makenewzGeneric(tree *tr, nodeptr p, nodeptr q,
                                double z0, int maxiter, boolean mask, 
                                double *tr_coreLZ0, 
                                traversalInfo *ti)
{
    //NUM_BRANCHES_GPU == 1
  int i;
  //boolean originalExecute[NUM_BRANCHES_GPU];
  
//traversalInfo *ti   = tr->td[0].ti; //ok
  //execmodel --->> statiki desmefsh mnhmhs ston host
  //mask --->> mask==0
  
      ti[0].pNumber = p->number;
      ti[0].qNumber = q->number;
      for(i = 0; i < NUM_BRANCHES_GPU; i++)
	{
	 // originalExecute[i] =  tr->executeModel[i];
	  ti[0].qz[i] =  z0; //
	  /*if(mask)
	    {
	      if(tr->partitionConverged[i])
		tr->executeModel[i] = FALSE;
	      else
		tr->executeModel[i] = TRUE;
	    }*/
	}
      
      tr->td[0].count = 1;
      
      if(!p->x)
	//computeTraversalInfo(p, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
          computeTraversalInfo(p, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
      if(!q->x)
	//computeTraversalInfo(q, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
          computeTraversalInfo(q, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);

  
  //topLevelMakenewz(tr, z0, maxiter, result);
  topLevelMakenewz( z0, maxiter,  tr_coreLZ0); 

  /*for(i = 0; i < tr->numBranches; i++)
      tr->executeModel[i] = TRUE;*/
}




__device__ nodeptr  removeNodeBIG (tree *tr, nodeptr p, int numBranches)
{  
  double   zqr[NUM_BRANCHES], result[NUM_BRANCHES];
  nodeptr  q, r;
  int i;
        
  q = p->next->back;
  r = p->next->next->back;
  
  for(i = 0; i < numBranches; i++)
    zqr[i] = q->z[i] * r->z[i];  
  
  //void makenewzGeneric(tree *tr, nodeptr p, nodeptr q, double *z0, int maxiter, double *result, boolean mask)  
//__device__ void makenewzGeneric(tree *tr, nodeptr p, nodeptr q, double z0, int maxiter, boolean mask, double *tr_coreLZ0, traversalInfo *ti)
//makenewzGeneric(tr, p, q, z0, maxiter, 0, tr_coreLZ0, ti);
// original makenewzGeneric(tr, p, q, z0, newzpercycle, z, FALSE);

  //makenewzGeneric(tr, q, r, zqr, iterations, result, FALSE);   
  makenewzGeneric(tr, q, r, zqr[0], iterations, 0, (double *)&tr_coreLZ0mw, ti);
  result[0] = dlnLdlz[0];

  for(i = 0; i < numBranches; i++)        
    tr->zqr[i] = result[i];

  hookup(q, r, result, numBranches); 
      
  p->next->next->back = p->next->back = (node *) NULL;

  return  q; 
}




__device__ int update(tree *tr, nodeptr p)
{       
  nodeptr  q; 
  boolean smoothedPartitions[NUM_BRANCHES];
  int i;
  double   z[NUM_BRANCHES], z0[NUM_BRANCHES];
  double _deltaz;

  q = p->back;   

  for(i = 0; i < tr->numBranches; i++)
    z0[i] = q->z[i];    

//void makenewzGeneric(tree *tr, nodeptr p, nodeptr q, double *z0, int maxiter, double *result, boolean mask)  
//__device__ void makenewzGeneric(tree *tr, nodeptr p, nodeptr q, double z0, int maxiter, boolean mask, double *tr_coreLZ0, traversalInfo *ti)
//makenewzGeneric(tr, p, q, z0, maxiter, 0, tr_coreLZ0, ti);
// original makenewzGeneric(tr, p, q, z0, newzpercycle, z, FALSE);
   
    makenewzGeneric(tr, p, q, z0[0], newzpercycle, 0, (double *)&tr_coreLZ0mw, ti);
    z[0] = dlnLdlz[0]; //result
    
  for(i = 0; i < tr->numBranches; i++)    
    smoothedPartitions[i]  = tr->partitionSmoothed[i];
      
  for(i = 0; i < tr->numBranches; i++)
    {         
      if(!tr->partitionConverged[i])
	{

	  
	    _deltaz = 0.00002;
	 
	    
	  if(ABS(z[i] - z0[i]) > _deltaz)  
	    {	      
	      smoothedPartitions[i] = FALSE;       
	    }	             
	  p->z[i] = q->z[i] = z[i];	 
	}
    }
  
  for(i = 0; i < tr->numBranches; i++)    
    tr->partitionSmoothed[i]  = smoothedPartitions[i];
  
  return TRUE;
}



__device__ int localSmooth (tree *tr, nodeptr p, int maxtimes)
{ 
  nodeptr  q;
  int i;
  
  if (isTip(p->number, nofSpecies_d)) return FALSE;
  
   for(i = 0; i < tr->numBranches; i++)	
     tr->partitionConverged[i] = FALSE;	

  while (--maxtimes >= 0) 
    {     
      for(i = 0; i < tr->numBranches; i++)	
	tr->partitionSmoothed[i] = TRUE;
	 	
      q = p;
      do 
	{
	  if (! update(tr, q)) return FALSE;
	  q = q->next;
        } 
      while (q != p);
      
      if (allSmoothed(tr)) 
	break;
    }

  for(i = 0; i < tr->numBranches; i++)
    {
      tr->partitionSmoothed[i] = FALSE; 
      tr->partitionConverged[i] = FALSE;
    }

  return TRUE;
}




__device__ boolean insertBIG (tree *tr, nodeptr p, nodeptr q, int numBranches)
{
  nodeptr  r, s;
  int i;
  
  r = q->back;
  s = p->back;
      
  for(i = 0; i < numBranches; i++)
    tr->lzi[i] = q->z[i];
  
  if(Thorough_d)
    { 
      double  zqr[NUM_BRANCHES], zqs[NUM_BRANCHES], zrs[NUM_BRANCHES], lzqr, lzqs, lzrs, lzsum, lzq, lzr, lzs, lzmax;      
      double defaultArray[NUM_BRANCHES];	
      double e1[NUM_BRANCHES], e2[NUM_BRANCHES], e3[NUM_BRANCHES];
      double *qz;
      
      qz = q->z;
      
      for(i = 0; i < numBranches; i++)
	defaultArray[i] = defaultz;
      /*
        //makenewzGeneric(tr, q, r, zqr, iterations, result, FALSE);   
  makenewzGeneric(tr, q, r, zqr[0], iterations, 0, (double *)&tr_coreLZ0mw, ti);
  result[0] = dlnLdlz[0];
       */
      makenewzGeneric(tr, q, r, qz[0], iterations, 0, (double *)&tr_coreLZ0mw, ti);           
      zqr[0] = dlnLdlz[0];
      makenewzGeneric(tr, q, s, defaultArray[0], iterations, 0, (double *)&tr_coreLZ0mw, ti);                  
      zqs[0] = dlnLdlz[0];
      makenewzGeneric(tr, r, s, defaultArray[0], iterations, 0, (double *)&tr_coreLZ0mw, ti);
      zrs[0] = dlnLdlz[0];
      
      for(i = 0; i < numBranches; i++)
	{
	  lzqr = (zqr[i] > zmin) ? log(zqr[i]) : log(zmin); 
	  lzqs = (zqs[i] > zmin) ? log(zqs[i]) : log(zmin);
	  lzrs = (zrs[i] > zmin) ? log(zrs[i]) : log(zmin);
	  lzsum = 0.5 * (lzqr + lzqs + lzrs);
	  
	  lzq = lzsum - lzrs;
	  lzr = lzsum - lzqs;
	  lzs = lzsum - lzqr;
	  lzmax = log(zmax);
	  
	  if      (lzq > lzmax) {lzq = lzmax; lzr = lzqr; lzs = lzqs;} 
	  else if (lzr > lzmax) {lzr = lzmax; lzq = lzqr; lzs = lzrs;}
	  else if (lzs > lzmax) {lzs = lzmax; lzq = lzqs; lzr = lzrs;}          
	  
	  e1[i] = exp(lzq);
	  e2[i] = exp(lzr);
	  e3[i] = exp(lzs);
	}
      hookup(p->next,       q, e1, numBranches);
      hookup(p->next->next, r, e2, numBranches);
      hookup(p,             s, e3, numBranches);      		  
    }
  else
    {       
      double  z[NUM_BRANCHES]; 
      
      for(i = 0; i < numBranches; i++)
	{
	  z[i] = sqrt(q->z[i]);      
	  
	  if(z[i] < zmin) 
	    z[i] = zmin;
	  if(z[i] > zmax)
	    z[i] = zmax;
	}
      
      hookup(p->next,       q, z, tr->numBranches);
      hookup(p->next->next, r, z, tr->numBranches);	                         
    }
  
  newviewGeneric(tr, p);
  
  if(Thorough_d)
    {     
      localSmooth(tr, p, smoothings);   
      for(i = 0; i < numBranches; i++)
	{
	  tr->lzq[i] = p->next->z[i];
	  tr->lzr[i] = p->next->next->z[i];
	  tr->lzs[i] = p->z[i];            
	}
    }           
  
  return  TRUE;
}



__device__ int testInsertBIG (tree *tr, nodeptr p, nodeptr q)
{
  double  qz[NUM_BRANCHES], pz[NUM_BRANCHES];
  nodeptr  r;
  int doIt = TRUE;
  double startLH = tr->endLH;
  int i;
  
  r = q->back; 
  for(i = 0; i < tr->numBranches; i++)
    {
      qz[i] = q->z[i];
      pz[i] = p->z[i];
    }
  

  if(doIt) //TRUE
    {     
      if (! insertBIG(tr, p, q, tr->numBranches))       return FALSE;         
      
      evaluateGeneric(tr, p->next->next);   
       
      if(tr->likelihood > tr->bestOfNode)
	{
	  tr->bestOfNode = tr->likelihood;
	  tr->insertNode = q;
	  tr->removeNode = p;   
	  for(i = 0; i < tr->numBranches; i++)
	    {
	      tr->currentZQR[i] = tr->zqr[i];           
	      tr->currentLZR[i] = tr->lzr[i];
	      tr->currentLZQ[i] = tr->lzq[i];
	      tr->currentLZS[i] = tr->lzs[i];      
	    }
	}
      
      if(tr->likelihood > tr->endLH)
	{			  
	  tr->insertNode = q;
	  tr->removeNode = p;   
	  for(i = 0; i < tr->numBranches; i++)
	    tr->currentZQR[i] = tr->zqr[i];      
	  tr->endLH = tr->likelihood;                      
	}        
      
      hookup(q, r, qz, tr->numBranches);
      
      p->next->next->back = p->next->back = (nodeptr) NULL;
      
      if(Thorough_d)
	{
	  nodeptr s = p->back;
	  hookup(p, s, pz, tr->numBranches);      
	} 
      
      if((tr->doCutoff) && (tr->likelihood < startLH))
	{
	  tr->lhAVG += (startLH - tr->likelihood);
	  tr->lhDEC++;
	  if((startLH - tr->likelihood) >= tr->lhCutoff)
	    return FALSE;	    
	  else
	    return TRUE;
	}
      else
	return TRUE;
    }
  else
    return TRUE;  
}




__device__ void addTraverseBIG(tree *tr, nodeptr p, nodeptr q, int mintrav, int maxtrav)
{  
  if (--mintrav <= 0) 
    {              
      if (! testInsertBIG(tr, p, q))  return;

    }
  
  if ((!isTip(q->number, nofSpecies_d)) && (--maxtrav > 0)) 
    {    
      addTraverseBIG(tr, p, q->next->back, mintrav, maxtrav);
      addTraverseBIG(tr, p, q->next->next->back, mintrav, maxtrav);    
    }
}




__device__ int rearrangeBIG(tree *tr, nodeptr p, int mintrav, int maxtrav)   
{  
  double   p1z[NUM_BRANCHES], p2z[NUM_BRANCHES], q1z[NUM_BRANCHES], q2z[NUM_BRANCHES];
  nodeptr  p1, p2, q, q1, q2;
  int      mintrav2, i;  
  int doP = TRUE, doQ = TRUE;
  
  if (maxtrav < 1 || mintrav > maxtrav)  return 0;
  q = p->back;

  
  if (!isTip(p->number, nofSpecies_d) && doP) 
    {     
      p1 = p->next->back;
      p2 = p->next->next->back;
      
     
      if(!isTip(p1->number, nofSpecies_d) || !isTip(p2->number, nofSpecies_d))
	{
	  for(i = 0; i < tr->numBranches; i++)
	    {
	      p1z[i] = p1->z[i];
	      p2z[i] = p2->z[i];	   	   
	    }
	  
	  if (! removeNodeBIG(tr, p,  tr->numBranches)) return badRear;
	  
	  if (!isTip(p1->number, nofSpecies_d)) 
	    {
	      addTraverseBIG(tr, p, p1->next->back,
			     mintrav, maxtrav);         
	      addTraverseBIG(tr, p, p1->next->next->back,
			     mintrav, maxtrav);          
	    }
	  
	  if (!isTip(p2->number, nofSpecies_d)) 
	    {
	      addTraverseBIG(tr, p, p2->next->back,
			     mintrav, maxtrav);
	      addTraverseBIG(tr, p, p2->next->next->back,
			     mintrav, maxtrav);          
	    }
	  	  
	  hookup(p->next,       p1, p1z, tr->numBranches); 
	  hookup(p->next->next, p2, p2z, tr->numBranches);	   	    	    
	  newviewGeneric(tr, p);	   	    
	}
    }  
  
  if (!isTip(q->number, nofSpecies_d) && maxtrav > 0 && doQ) 
    {
      q1 = q->next->back;
      q2 = q->next->next->back;
      
      /*if (((!q1->tip) && (!q1->next->back->tip || !q1->next->next->back->tip)) ||
	((!q2->tip) && (!q2->next->back->tip || !q2->next->next->back->tip))) */
      if (
	  (
	   ! isTip(q1->number, nofSpecies_d) && 
	   (! isTip(q1->next->back->number, nofSpecies_d) || ! isTip(q1->next->next->back->number, nofSpecies_d))
	   )
	  ||
	  (
	   ! isTip(q2->number, nofSpecies_d) && 
	   (! isTip(q2->next->back->number, nofSpecies_d) || ! isTip(q2->next->next->back->number, nofSpecies_d))
	   )
	  )
	{
	  
	  for(i = 0; i < tr->numBranches; i++)
	    {
	      q1z[i] = q1->z[i];
	      q2z[i] = q2->z[i];
	    }
	  
	  if (! removeNodeBIG(tr, q, tr->numBranches)) return badRear;
	  
	  mintrav2 = mintrav > 2 ? mintrav : 2;
	  
	  if (/*! q1->tip*/ !isTip(q1->number, nofSpecies_d)) 
	    {
	      addTraverseBIG(tr, q, q1->next->back,
			     mintrav2 , maxtrav);
	      addTraverseBIG(tr, q, q1->next->next->back,
			     mintrav2 , maxtrav);         
	    }
	  
	  if (/*! q2->tip*/ ! isTip(q2->number, nofSpecies_d)) 
	    {
	      addTraverseBIG(tr, q, q2->next->back,
			     mintrav2 , maxtrav);
	      addTraverseBIG(tr, q, q2->next->next->back,
			     mintrav2 , maxtrav);          
	    }	   
	  
	  hookup(q->next,       q1, q1z, tr->numBranches); 
	  hookup(q->next->next, q2, q2z, tr->numBranches);
	  
	  newviewGeneric(tr, q); 	   
	}
    } 
  
  return  1;
}











/*
 * 
 * tree evaluate path
 * 
 */


__device__ int smooth (tree *tr, nodeptr p)
{
  nodeptr  q;
  
  if (! update(tr, p))               return FALSE; /*  Adjust branch */
  if (! isTip(p->number, nofSpecies_d)) 
    {                                  /*  Adjust descendants */
      q = p->next;
      while (q != p) 
	{
	  if (! smooth(tr, q->back))   return FALSE;
	  q = q->next;
	}	
	newviewGeneric(tr, p);     
    }
  
  return TRUE;
} 





__device__ int  smoothTree (tree *tr, int maxtimes)
{
  nodeptr  p, q;   
  int i, count = 0;
   
  p = tr->start;
  for(i = 0; i < tr->numBranches; i++)
    tr->partitionConverged[i] = FALSE;

  while (--maxtimes >= 0) 
    {    
      for(i = 0; i < tr->numBranches; i++)	
	tr->partitionSmoothed[i] = TRUE;		

      if (! smooth(tr, p->back))       return FALSE;
      if (!isTip(p->number, nofSpecies_d)) 
	{
	  q = p->next;
	  while (q != p) 
	    {
	      if (! smooth(tr, q->back))   return FALSE;
	      q = q->next;
	    }
	}
         
      count++;

      if (allSmoothed(tr)) 
	break;      
    }

  for(i = 0; i < tr->numBranches; i++)
    tr->partitionConverged[i] = FALSE;



  return TRUE;
} 


__device__ int treeEvaluate (tree *tr, double smoothFactor)       /* Evaluate a user tree */
{
  int result;
 

  
  result = smoothTree(tr, (int)((double)smoothings * smoothFactor));
  
  assert(result); 

  evaluateGeneric(tr, tr->start);   
    


  return TRUE;
}






















/*
 **************************************************************************** 
 *      
 **************************************************************************** 
 */














__device__ void topLevelMakenewzPath(double z0, int _maxiter, double *tr_coreLZ0)
{
    //topLevelMakenewz()  
  double   z[NUM_BRANCHES_GPU], zprev[NUM_BRANCHES_GPU], zstep[NUM_BRANCHES_GPU];
  double  dlnLdlz_local[NUM_BRANCHES_GPU], d2lnLdlz2_local[NUM_BRANCHES_GPU];
  int i, maxiter[NUM_BRANCHES_GPU], numBranches;
  
  boolean outerConverged[NUM_BRANCHES_GPU];
  boolean loopConverged;
  int tr_curvatOK, tr_executeModel;
   
  firstIteration_mw = TRUE;
  numBranches = 1;
  
 //tree *tr, double *z0=>z0, int _maxiter, double *result 

    for(i = 0; i < numBranches; i++)
    {
      z[i] = z0;
      maxiter[i] = _maxiter;
      outerConverged[i] = FALSE;
      tr_curvatOK     = TRUE;
    }

    
    do{  
      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr_curvatOK == TRUE)
	    {
	      tr_curvatOK = FALSE;

	      zprev[i] = z[i];

	      zstep[i] = (1.0 - zmax) * z[i] + zmin;
	    }
	}

      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr_curvatOK == FALSE)
	    {
	      double lz;

	      if (z[i] < zmin) z[i] = zmin;
	      else if (z[i] > zmax) z[i] = zmax;
	      lz    = log(z[i]);

	      *tr_coreLZ0 = lz;
	    }
	}

	  //for(model = 0; model < tr->NumberOfModels; model++)
	    //tr->executeModel[model] = !tr->curvatOK[0];
	tr_executeModel = !tr_curvatOK;

        

        if (tr_executeModel){
            execWorkers(NEWZCORE);
        }
        
        if (firstIteration_mw)              
            firstIteration_mw = FALSE;

          //reduction GPU results
          dlnLdlz_local[0] = 0.0;
          d2lnLdlz2_local[0] = 0.0;
          for(i = 0; i < alignLength_d; i++)
          {
            dlnLdlz_local[0]   += dlnLdlz[i];
            d2lnLdlz2_local[0] += d2lnLdlz2[i];
          }      
      
      //printf("dev reduction dlnLdlz: %f\n", dlnLdlz[0]);
      //printf("dev reduction d2lnLdlz2: %f\n", d2lnLdlz2[0]);
      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr_curvatOK == FALSE)
	    {
	      if ((d2lnLdlz2_local[i] >= 0.0) && (z[i] < zmax))
		zprev[i] = z[i] = 0.37 * z[i] + 0.63;  /*  Bad curvature, shorten branch */
	      else
		tr_curvatOK = TRUE;
	    }
	}

      for(i = 0; i < numBranches; i++)
	{
	  if(tr_curvatOK == TRUE && outerConverged[i] == FALSE)
	    {
	      if (d2lnLdlz2_local[i] < 0.0)
		{
		  double tantmp = -dlnLdlz_local[i] / d2lnLdlz2_local[i];
		  if (tantmp < 100)
		    {
		      z[i] *= EXP(tantmp);
		      if (z[i] < zmin)
			z[i] = zmin;

		      if (z[i] > 0.25 * zprev[i] + 0.75)
			z[i] = 0.25 * zprev[i] + 0.75;
		    }
		  else
		    z[i] = 0.25 * zprev[i] + 0.75;
		}
	      if (z[i] > zmax) z[i] = zmax;

	      maxiter[i] = maxiter[i] - 1;
	      if(maxiter[i] > 0 && (ABS(z[i] - zprev[i]) > zstep[i]))
		outerConverged[i] = FALSE;
	      else
		outerConverged[i] = TRUE;
	    }
	}

      loopConverged = TRUE;
      for(i = 0; i < numBranches; i++)
	loopConverged = loopConverged && outerConverged[i];
     
    }
  while (!loopConverged);
    
  
  chkLast = 1;
  cds();
  endGPUexecution = TRUE;
  cds();
  
 
  b.blockFinish = 0;
  tr_executeModel = TRUE;  

  dlnLdlz[0] = z[0]; //globalize result, send result to host

  //printf("thread0 res: %f\n", d_dlnLdlz[0]);
  //printf("thread 0 finished!\n");

}







__device__ void makenewzGenericPath(tree *tr, nodeptr p, nodeptr q,
                                double z0, int maxiter, boolean mask, 
                                double *tr_coreLZ0, 
                                traversalInfo *ti)
{
    //NUM_BRANCHES_GPU == 1
  int i;
  //boolean originalExecute[NUM_BRANCHES_GPU];
  
//traversalInfo *ti   = tr->td[0].ti; //ok
  //execmodel --->> statiki desmefsh mnhmhs ston host
  //mask --->> mask==0
  
      ti[0].pNumber = p->number;
      ti[0].qNumber = q->number;
      for(i = 0; i < NUM_BRANCHES_GPU; i++)
	{
	 // originalExecute[i] =  tr->executeModel[i];
	  ti[0].qz[i] =  z0; //
	  /*if(mask)
	    {
	      if(tr->partitionConverged[i])
		tr->executeModel[i] = FALSE;
	      else
		tr->executeModel[i] = TRUE;
	    }*/
	}
      
      tr->td[0].count = 1;
      
      if(!p->x)
	//computeTraversalInfo(p, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
          computeTraversalInfo(p, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
      if(!q->x)
	//computeTraversalInfo(q, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
          computeTraversalInfo(q, &(ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);

  
  //topLevelMakenewz(tr, z0, maxiter, result);
  topLevelMakenewzPath( z0, maxiter,  tr_coreLZ0); 

  /*for(i = 0; i < tr->numBranches; i++)
      tr->executeModel[i] = TRUE;*/
}











__device__ void execPath(tree *tr, nodeptr p, nodeptr q, 
                        double z0, int maxiter, double *tr_coreLZ0,
                        traversalInfo *ti)
{
    //makeNewzGeneric()
    //topLevelMakenewz(double z0, int _maxiter, double *tr_coreLZ0, double *d_dlnLdlz, double *d_d2lnLdlz2, int alignLength);    
    
    
    //__device__ void makenewzGeneric(tree *tr, nodeptr p, nodeptr q, double z0, int maxiter, boolean mask, double *tr_coreLZ0, double *d_dlnLdlz, double *d_d2lnLdlz2, int alignLength, traversalInfo *ti)
    
    
    makenewzGenericPath(tr, p, q, z0, maxiter, 0, tr_coreLZ0, ti);
    
    
    //update(tr, p);
    //localSmooth (tr,p, 3);
    
    //insertBIG (tr, p, q, numBranches);
    //!!!Thorough!!!
    //!!!newviewGeneric!!!
    //!!!chckLast!!!
    
    //testInsertBIG ( tr,  p,  q);
    //!!! evaluateGeneric !!!
    
    //addTraverseBIG(tr,  p,  q,  mintrav,  maxtrav);
    //!!!recursion!!!
    
    //rearrangeBIG(tr,  p,  mintrav,  maxtrav);
    //!!!newviewgeneric!!!
    
}    
























/*
 **************************************************************************** 
 *      LINK to HOST
 **************************************************************************** 
 */






__global__ void startKernelPathNewzCore(
        double z0,
        //int _maxiter, //for topLevelMakenewzVersion
        int maxiter,
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories,
        double tr_coreLZ0,

        int pIsTip,
        int qIsTip,

        nodeptr p, nodeptr q)
{
    
  //int thi = blockDim.x*blockIdx.x + threadIdx.x; 

    //if (threadIdx.x*blockIdx.x==1){
        endGPUexecution = FALSE;
        firstIteration_mw = TRUE;
        chkLast = FALSE;
   // }
  globalBarrier();  
    
  if (blockIdx.x == gridDim.x*gridDim.y*gridDim.z - 1)
  { //the last block
    if (threadIdx.x == 0)
      //execPath(z0, _maxiter, (double *)&tr_coreLZ0mw, d_dlnLdlz, d_d2lnLdlz2, alignLength); //version for topLevelMakenewz       
        execPath(tr, p, q, 
                 z0, maxiter, (double *)&tr_coreLZ0mw, ti); //version for makeNewzGeneric
    else
      return;
  }
  else{
     while(!endGPUexecution){
          parallelExecution(        
                         tr_td0_count,
                         tr_mxtips,
                         tr_rateHetModel,
                         tr_NumberOfCategories,
                         (double *)&tr_coreLZ0mw,
                         pIsTip,
                         qIsTip,
                         firstIteration_mw);
      }
  }   
}  



__device__ void setGlobalVars(int h_Thorough)
{
    
                             
    //tr_td0_count = ;
    /* 
       to tr->td[0].count arxikopoieite panta prin thn klhsh newz, newView, evaluate 
       to tr_td0_count to xrhsimopoiw mono otan stelnw thn timi apeksw 
     */
                         
    //tr_mxtips = ;
                         
   // tr_rateHetModel = tr->rateHetModel;
    
                         
    //tr_NumberOfCategories = tr->NumberOfCategories;
    Thorough_d = h_Thorough;


    
} 



__device__ void execBigPathTreeEval(double smoothFactor, node *tr_start, int h_Thorough)
{
    
    setGlobalVars(h_Thorough);
    tr->start = tr_start;
    assert(Thorough_d!=-1);
    

    treeEvaluate(tr, smoothFactor);

    endExecKernel();
    
}    




__device__ void execBigPathRearrange( nodeptr p, int mintrav, int maxtrav, int h_Thorough)
{
    
    setGlobalVars(h_Thorough);
    assert(Thorough_d!=-1);

    rearrangeBIG(tr, p, mintrav, maxtrav);  

    endExecKernel();
    
}    







__global__ void bigPathTreeEvalKernel(double smoothFactor, node *tr_start, int h_Thorough)
{
  //int thi = blockDim.x*blockIdx.x + threadIdx.x; 

    //if (threadIdx.x*blockIdx.x==1){
        endGPUexecution = FALSE;
        firstIteration_mw = TRUE;
        chkLast = FALSE;
   // }
  globalBarrier();  
    
  if (blockIdx.x == gridDim.x*gridDim.y*gridDim.z - 1)
  { //the last block
    if (threadIdx.x == 0)
        execBigPathTreeEval(smoothFactor, tr_start, h_Thorough);
    else
      return;
  }
  else{
     while(!endGPUexecution){
          parallelExecution(        
                         tr->td[0].count,
                         tr->mxtips,
                         tr->rateHetModel,
                         tr->NumberOfCategories,
                         (double *)&tr_coreLZ0mw,
                         1,
                         1,
                         firstIteration_mw);
      }
  }      
    
    
}






__global__ void bigPathRearrangeKernel(nodeptr p, int mintrav, int maxtrav, int h_Thorough)
{
  //int thi = blockDim.x*blockIdx.x + threadIdx.x; 

    //if (threadIdx.x*blockIdx.x==1){
        endGPUexecution = FALSE;
        firstIteration_mw = TRUE;
        chkLast = FALSE;
   // }
  globalBarrier();  
    
  if (blockIdx.x == gridDim.x*gridDim.y*gridDim.z - 1)
  { //the last block
    if (threadIdx.x == 0)
        execBigPathRearrange( p, mintrav, maxtrav, h_Thorough);
    else
      return;
  }
  else{
     while(!endGPUexecution){
          parallelExecution(        
                         tr->td[0].count,
                         tr->mxtips,
                         tr->rateHetModel,
                         tr->NumberOfCategories,
                         (double *)&tr_coreLZ0mw,
                         1,
                         1,
                         firstIteration_mw);
      }
  }      
    
    
}





extern "C" void bigPathTreeEval( double smoothFactor, node *tr_start)
{
    int almodthreads;
    dim3 dimGrid, dimBlock;    
    
    dimBlock.x = DIMBLOCKX;
    dimBlock.y = 1;
    dimBlock.z = 1;
    
    almodthreads = alignLength % dimBlock.x;

    if (almodthreads)
        dimGrid.x = alignLength/dimBlock.x +1;
    else
        dimGrid.x = alignLength/dimBlock.x;
    
    dimGrid.y = 1;
    dimGrid.z = 1;
    
    
    dimGrid.x++; //block(one thread of this block) running the path. Master thread
    assert(dimGrid.x <120); //avoid deadlock because of hardware
    assert(dimBlock.x <128);
    //assert(tr->mxtips==nofSpecies);
    assert(Thorough!=-1);
    
    bigPathTreeEvalKernel<<<dimGrid, dimBlock>>>(smoothFactor, tr_start, Thorough);
    
        
}



extern "C" void bigPathRearrange(nodeptr p, int mintrav, int maxtrav)
{
    
    
    int almodthreads;
    dim3 dimGrid, dimBlock;    
    
    dimBlock.x = DIMBLOCKX;
    dimBlock.y = 1;
    dimBlock.z = 1;
    
    almodthreads = alignLength % dimBlock.x;

    if (almodthreads)
        dimGrid.x = alignLength/dimBlock.x +1;
    else
        dimGrid.x = alignLength/dimBlock.x;
    
    dimGrid.y = 1;
    dimGrid.z = 1;
    
    
    dimGrid.x++; //block(one thread of this block) running the path. Master thread
    assert(dimGrid.x <120); //avoid deadlock because of hardware
    assert(dimBlock.x <128);
    //assert(tr->mxtips==nofSpecies);
    assert(Thorough!=-1);

    bigPathRearrangeKernel<<<dimGrid, dimBlock>>>(p, mintrav, maxtrav, Thorough);
        
} 













extern "C" void pathNewzcoreGPU( int pIsTip, int qIsTip, tree *tr, double z0, int maxiter, nodeptr p, nodeptr q)
{    
    int almodthreads;
    dim3 dimGrid, dimBlock;    
    
    dimBlock.x = DIMBLOCKX;
    dimBlock.y = 1;
    dimBlock.z = 1;
    
    almodthreads = alignLength % dimBlock.x;

    if (almodthreads)
        dimGrid.x = alignLength/dimBlock.x +1;
    else
        dimGrid.x = alignLength/dimBlock.x;
    
    dimGrid.y = 1;
    dimGrid.z = 1;
    
    
    dimGrid.x++; //block(one thread of this block) running the path. Master thread
    assert(dimGrid.x <120); //avoid deadlock because of hardware
    assert(dimBlock.x <128);
    assert(tr->mxtips==nofSpecies);
    
    /*__global__ void startKernelPathNewzCore(
        double z0,
        //int _maxiter, //for topLevelMakenewzVersion
        int maxiter,
        int tr_td0_count,
        int tr_mxtips,
        int tr_rateHetModel,
        int tr_NumberOfCategories,
        double tr_coreLZ0,
        float *d_tmpCatSpace,
        float *d_tmpDiagSpace,
        float *wr,
        float *wr2,
        double *d_dlnLdlz,
        double *d_d2lnLdlz2,
        float *sumBuffer,
        int pIsTip,
        int qIsTip,
        int alignLength,
        int nofSpecies,
        unsigned int *scalerThread, 
        float *EV,
        float *tipVector,
        double *gammaRates,
        double *patrat,
        double *ei,
        double *eign,
        int *rateCategory,
        int *wgt,
        tree *tr,
        traversalInfo *ti,
        float **xVector,
        unsigned char **yVector,
        nodeptr p, nodeptr q)*/
    
    
    startKernelPathNewzCore<<<dimGrid, dimBlock>>>(

            z0,
            maxiter,
            tr->td[0].count,
            tr->mxtips,
            tr->rateHetModel,
            tr->NumberOfCategories,
            tr->coreLZ[0],
            pIsTip, qIsTip,
            p, q);
    

}

    

extern "C" void kernelNewzCore(int firstIteration, int pIsTip, int qIsTip, tree *tr)
{
    //if (firstIteration)
       // makenewzIterative(); 
    //execCore(tr, dlnLdlz, d2lnLdlz2);
    
      
    int almodthreads;
    dim3 dimGrid, dimBlock;    
    
    dimBlock.x = DIMBLOCKX;
    dimBlock.y = 1;
    dimBlock.z = 1;
    
    almodthreads = alignLength % dimBlock.x;

    if (almodthreads)
        dimGrid.x = alignLength/dimBlock.x +1;
    else
        dimGrid.x = alignLength/dimBlock.x;
    
    dimGrid.y = 1;
    dimGrid.z = 1;
    
    assert(0);
    startKernelNewzCore<<<dimGrid, dimBlock>>>(
            tr->td[0].count,
            tr->mxtips,
            tr->rateHetModel,
            tr->NumberOfCategories,
            tr->coreLZ[0],
            pIsTip, qIsTip,
            firstIteration);
    //d_tree
    
    /*
     * tr->td[0].count
     * tr->mxtips
     * tr->rateHetModel
     * tr->NumberOfCategories
     * tr->coreLZ[0]
     * 
     */
}
    
    
   
    
    
    
    
    
extern "C" void kernelNewview(tree *tr)
{
    
    int almodthreads;
    dim3 dimGrid, dimBlock;    
    
    dimBlock.x = DIMBLOCKX;
    dimBlock.y = 1;
    dimBlock.z = 1;
    
    almodthreads = alignLength % dimBlock.x;

    if (almodthreads)
        dimGrid.x = alignLength/dimBlock.x +1;
    else
        dimGrid.x = alignLength/dimBlock.x;
    
    dimGrid.y = 1;
    dimGrid.z = 1;
    
    
    startKernelNewview<<<dimGrid, dimBlock>>>(
            tr->td[0].count,
            tr->mxtips,
            tr->rateHetModel,
            tr->NumberOfCategories
            );
   
    //d_tree
    
}
    
   







    
extern "C" void kernelnewViewEvaluate(int countr, int pIsTip, int qIsTip, int execModel)
{

    //testKernel<<<1,1024>>>(d_yVector, d_patrat, d_tipVector, d_xVector, d_wgt, d_tree, d_ti, d_left);
    

    int almodthreads;
    dim3 dimGrid, dimBlock;
    
#ifndef forEveryNofAlignment
    
    dimBlock.x = DIMBLOCKX;
    dimBlock.y = 1;
    dimBlock.z = 1;
    
    almodthreads = alignLength % dimBlock.x;

    if (almodthreads)
        dimGrid.x = alignLength/dimBlock.x +1;
    else
        dimGrid.x = alignLength/dimBlock.x;
    
    dimGrid.y = 1;
    dimGrid.z = 1;
    
    
    //printf("alignLength: %d, Grid.x: %d, Block.x: %d\n", alignLength, dimGrid.x, dimBlock.x);
    //assert(0);
#else
    //dimGrid.x = alignLength/32;
    //dimGrid.x = 1;
    dimGrid.x = alignLength;
    dimGrid.y = 1;
    dimGrid.z = 1;
    //dimBlock.x = alignLength;
    //dimBlock.x = 32;
    dimBlock.x = 1;
    dimBlock.y = 1;
    dimBlock.z = 1;
    assert(dimGrid.x*dimBlock.x == alignLength);
#endif
    
    startKernelnewViewEvaluate<<<dimGrid, dimBlock>>>(execModel, countr, pIsTip, qIsTip);


    
}




__global__ void startInitKernel(

 float *d1_umpX1,
 float *d1_umpX2,
 float *d1_left,
 float *d1_right,
            int alignLength1,
            int nofSpecies1,

            float *sumBuffer1,
              float *EV1,
              float *tipVector1,
              double *gammaRates1,

              double *patrat1,
              double *ei1,
              double *eign1,
              int *rateCategory1,
              int *wgt1,

              unsigned int *scalerThread1,
              double *partitionLikelihood1,

              double *dlnLdlz1,
              double *d2lnLdlz21,

              float *tmpCatSpace1,
              float *tmpDiagSpace1,

              float *wr21,
              float *wr1,


              tree *tr1,
              traversalInfo *ti1,
              float **xVector1,
              unsigned char **yVector1
            )
{
        d_umpX1 = d1_umpX1;
        d_umpX2 = d1_umpX2;
        d_left = d1_left;
        d_right = d1_right;
        
            alignLength_d = alignLength1;
            nofSpecies_d = nofSpecies1;

            sumBuffer = sumBuffer1;
              EV =EV1;
              tipVector =tipVector1;
              gammaRates =gammaRates1;

              patrat = patrat1;
              ei = ei1;
              eign = eign1;
              rateCategory = rateCategory1;
              wgt = wgt1;

              scalerThread = scalerThread1;
              partitionLikelihood = partitionLikelihood1;

              dlnLdlz = dlnLdlz1;
              d2lnLdlz2 = d2lnLdlz21;

              tmpCatSpace = tmpCatSpace1;
              tmpDiagSpace = tmpDiagSpace1;

              wr2 = wr21;
              wr = wr1;


              tr = tr1;
              //tr->td[0].ti= ti;
              ti = ti1;
              xVector = xVector1;
              yVector = yVector1;
    
}
    
extern "C" void initKernel()
{
    

    
    startInitKernel<<<1, 1>>>(
        d2_umpX1,
        d2_umpX2,
        d2_left,
        d2_right,
             alignLength,
             nofSpecies,

             d_sumBuffer,
            d_EV,
            d_tipVector,
            d_gammaRates,

            d_patrat,
            d_ei,
            d_eign,
            d_rateCategory,
            d_wgt,

            d_scalerThread,
            d_partitionLikelihood,

            d_dlnLdlz,
            d_d2lnLdlz2,

            d_tmpCatSpace,
            d_tmpDiagSpace,

            d_wr2,
            d_wr,


            d_tree,
            d_ti,
            d_xVector,
            d_yVector );


    
}


/* use of tree in gpu
 * cudamemcpy d_tr
 * 
 * tr->td[0].count
 * tr->mxtips
 * tr->rateHetModel
 * tr->NumberOfCategories
 * 
 * 
 */