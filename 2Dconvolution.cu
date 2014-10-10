/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_functions.h>
#include <helper_math.h>

// includes, project
#include "2Dconvolution.h"


////////////////////////////////////////////////////////////////////////////////
// declarations, forward

#define TILE_WIDTH BLOCK_SIZE
#define Mask_width KERNEL_SIZE
#define Mask_radius Mask_width/2
#define w (TILE_WIDTH + Mask_width - 1)

extern "C"


void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(int width, int height);
Matrix AllocateMatrix(int width, int height);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
void ConvolutionOnDeviceShared(const Matrix M, const Matrix N, Matrix P);
void ConvolutionOnDeviceStarter(const Matrix M, const Matrix N, Matrix P);
__device__ void batchLoading(int *dest, int *destX, int *destY, int *src, int *srcX, int *srcY,
                  int ti, int tj, int bi, int bj, int width, int pw);
void danemarcaTesting(int i, int j);


////////////////////////////////////////////////////////////////////////////////
// Înmulțirea fără memorie partajată
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P) {
    	//TODO: calculul rezultatului convoluției

 	int row = blockIdx.y * blockDim.y + threadIdx.y;
  	int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(col < P.width && row <P.height) {
		float C = 0.0;
		int i, j= 0;

		// Doing some loop unrolling for Xtra performance
		// Pentru ca tot a prezentat un coleg optimizari de cod ^^
		for(i=0; i < M.pitch; i++) {
			j=0;

			if( (col +j -2 >=0) && (col+j-2 < N.pitch) && (row+i-2 >= 0) && (row+i-2 < N.height))
                                        C = C +  N.elements[(row + i -2) * N.pitch + (col + j -2)] * M.elements[i*M.pitch + j ];
			j++; 
			if( (col +j -2 >=0) && (col+j-2 < N.pitch) && (row+i-2 >= 0) && (row+i-2 < N.height))
                                        C = C +  N.elements[(row + i -2) * N.pitch + (col + j -2)] * M.elements[i*M.pitch + j ];
                        j++;
			if( (col +j -2 >=0) && (col+j-2 < N.pitch) && (row+i-2 >= 0) && (row+i-2 < N.height))
                                        C = C +  N.elements[(row + i -2) * N.pitch + (col + j -2)] * M.elements[i*M.pitch + j ];
                        j++;
			if( (col +j -2 >=0) && (col+j-2 < N.pitch) && (row+i-2 >= 0) && (row+i-2 < N.height))
                                        C = C +  N.elements[(row + i -2) * N.pitch + (col + j -2)] * M.elements[i*M.pitch + j ];
                        j++;
			if( (col +j -2 >=0) && (col+j-2 < N.pitch) && (row+i-2 >= 0) && (row+i-2 < N.height))
                                        C = C +  N.elements[(row + i -2) * N.pitch + (col + j -2)] * M.elements[i*M.pitch + j ];
                        j++;


		}
		P.elements[row * P.pitch + col]= C;
	}
}




__device__ void batchLoading(int *dest, int *destX, int *destY, int *src, int *srcX, int *srcY, 
		   int ti, int tj, int bi, int bj, int width, int pw) {	
	*dest = ti * TILE_WIDTH + tj + TILE_WIDTH*TILE_WIDTH*pw;
	*destY= *dest/w;
        *destX= *dest%w;
	*srcY = bi * TILE_WIDTH + *destY - Mask_radius;
        *srcX = bj * TILE_WIDTH + *destX - Mask_radius;
        *src = (*srcY * width + *srcX);
}

////////////////////////////////////////////////////////////////////////////////
// Înmulițirea cu memorie partajată
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernelShared(Matrix M, Matrix N, Matrix P) {
    	//TODO: calculul rezultatului convoluției

	__shared__ float Bs[w][w];
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	int dest, destY, destX, src, srcY, srcX;
	// srcY, srcX = coordonatele din memoria globala
	// destX, destY = coordonate din memoria shared
	int ti= threadIdx.y;
	int tj= threadIdx.x;
	int bi= blockIdx.y;
	int bj= blockIdx.x;
	int width = N.width;
        int height= N.height;
	float value = 0.0;
	int y, x;

	
	batchLoading(&dest, &destX, &destY, &src, &srcX, &srcY, ti, tj, bi, bj, width, 0);

      	if (! (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width))
		Bs[destY][destX] = 0;
	else
		Bs[destY][destX] = N.elements[src];


	batchLoading(&dest, &destX, &destY, &src, &srcX, &srcY, ti, tj, bi, bj, width, 1);

	if(destY < w) {
		if (! (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width))
                	Bs[destY][destX] = 0;
        	else
                	Bs[destY][destX] = N.elements[src];
	}

	if(tj < P.width && ti <P.height)
		As[ti][tj] = M.elements [ti * M.pitch + tj];


        __syncthreads();

 
	for(int y = 0; y < Mask_width; y++) {
		x = 0;
		value = value + Bs[y+ti][x+tj] * As[y][x];
		//danemarcaTesting(M,N,As,Bs,y,x,ti,tj);
		++x;
		value = value + Bs[y+ti][x+tj] * As[y][x];
                ++x;
		value = value + Bs[y+ti][x+tj] * As[y][x];
                ++x;
		value = value + Bs[y+ti][x+tj] * As[y][x];
                ++x;
		value = value + Bs[y+ti][x+tj] * As[y][x];
                ++x;
        }
	
	y = bi * TILE_WIDTH + ti;
      	x = bj * TILE_WIDTH + tj;
      	if (y < P.height && x < P.width) {
		P.elements[y* P.pitch + x] = value;
		//printf("We fucking got here %f- %d\n", P.elements[(y * P.pitch + x)], (y * P.pitch + x));
	} 
      	__syncthreads(); 
}


////////////////////////////////////////////////////////////////////////////////
// Returnează 1 dacă matricele sunt ~ egale
////////////////////////////////////////////////////////////////////////////////
int CompareMatrices(Matrix A, Matrix B) {
    	int i;
    	if(A.width != B.width || A.height != B.height || A.pitch != B.pitch)
        	return 0;
    	int size = A.width * A.height;
    	for(i = 0; i < size; i++)
        	if(fabs(A.elements[i] - B.elements[i]) > MAX_ERR)
            		return 0;
    	return 1;
}


void GenerateRandomMatrix(Matrix m) {
    	int i;
    	int size = m.width * m.height;

    	srand(time(NULL));

    	for(i = 0; i < size; i++)
        	m.elements[i] = rand() / (float)RAND_MAX;
}


////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)  {
    	int width = 0, height = 0, i, j;
    	FILE *f, *out, *g, *h, *graphs;
    	if(argc < 2) {
        	printf("Argumente prea puține, trimiteți id-ul testului care trebuie rulat\n");
        	return 0;
    	}

	// Printing the test number in the file
        //graphs = fopen("graphs.out", "a");
        //fprintf(graphs, "%s ", argv[1]);

    	char name[100];
    	sprintf(name, "./tests/test_%s.txt", argv[1]);
    	f = fopen(name, "r");
	g= fopen("test.txt", "w");
	h= fopen("gold.txt", "w");
    	out = fopen("out.txt", "a");
    	
	fscanf(f, "%d%d", &width, &height);
	// Printing the matrix dimension in the file
        graphs = fopen("graphs.out", "a");
        fprintf(graphs, "%d ", width*height);


    	Matrix M;//kernel de pe host
    	Matrix N;//matrice inițială de pe host
    	Matrix P;//rezultat fără memorie partajată calculat pe GPU
    	Matrix PS;//rezultatul cu memorie partajată calculat pe GPU
    
    	M = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE);
    	N = AllocateMatrix(width, height);        
    	P = AllocateMatrix(width, height);
    	PS = AllocateMatrix(width, height);

    	GenerateRandomMatrix(M);
    	GenerateRandomMatrix(N);


	// calculează rezultatul pe CPU pentru comparație si masoara timpii
	StopWatchInterface *kernelTime = NULL;
        sdkCreateTimer(&kernelTime);
        sdkResetTimer(&kernelTime);
        sdkStartTimer(&kernelTime);
        Matrix reference = AllocateMatrix(P.width, P.height);
        computeGold(reference.elements, M.elements, N.elements, N.height, N.width);
        sdkStopTimer(&kernelTime);

        // Scrie rezultatele in fisiere
        fprintf(graphs, "%lf \n", sdkGetTimerValue(&kernelTime));
        for(i=0; i<P.width; i++) {
                for(j=0; j<P.height; j++)
                        fprintf(h, "%f ", reference.elements[i + j]);
                fprintf(h, "\n");
                }


	ConvolutionOnDeviceStarter(M, N, P);

    	// M * N pe device
    	ConvolutionOnDevice(M, N, P);

    	// M * N pe device cu memorie partajată
    	ConvolutionOnDeviceShared(M, N, PS);
	for(i=0; i<PS.width; i++) {
                for(j=0; j<PS.height; j++)
                        fprintf(g, "%f ", PS.elements[i + j]);
                fprintf(g,"\n");
        }


    	// verifică dacă rezultatul obținut pe device este cel așteptat
    	int res = CompareMatrices(reference, P);
    	printf("Test global %s\n", (1 == res) ? "PASSED" : "FAILED");
    	fprintf(out, "Test global %s %s\n", argv[1], (1 == res) ? "PASSED" : "FAILED");

    	// verifică dacă rezultatul obținut pe device cu memorie partajată este cel așteptat
  	//  int ress = CompareMatrices(reference, PS);
    	int ress = CompareMatrices(reference, PS);
    	printf("Test shared %s\n", (1 == ress) ? "PASSED" : "FAILED");
    	fprintf(out, "Test shared %s %s\n", argv[1], (1 == ress) ? "PASSED" : "FAILED");
   

    	// Free matrices
   	FreeMatrix(&M);
	FreeMatrix(&N);
    	FreeMatrix(&P);
    	FreeMatrix(&PS);

    	fclose(f);
    	fclose(out);
	fclose(g);
	fclose(h);
	fclose(graphs);
    	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P) {
    	Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device
	FILE *graphs;


    	//pentru măsurarea timpului de execuție în kernel
    	StopWatchInterface *kernelTime = NULL;
    	sdkCreateTimer(&kernelTime);
    	sdkResetTimer(&kernelTime);

    	//TODO: alocați matricele de pe device
	
	Md = AllocateDeviceMatrix(M.width, M.height);
	Nd = AllocateDeviceMatrix(N.width, N.height);
	Pd = AllocateDeviceMatrix(P.width, P.height);
	

	//TODO: copiați datele de pe host (M, N) pe device (MD, Nd)
	 cudaMemcpy(Md.elements, M.elements, M.width * M.height *sizeof(float), cudaMemcpyHostToDevice);
	 cudaMemcpy(Nd.elements, N.elements, N.width * N.height *sizeof(float), cudaMemcpyHostToDevice);


    	//TODO: setați configurația de rulare a kernelului
    	sdkStartTimer(&kernelTime); 

	//TODO: lansați în execuție kernelul  
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	
	float x, y;

	if( N.width%BLOCK_SIZE != 0)
		x= N.width/BLOCK_SIZE +1;
	else
		x= N.width/BLOCK_SIZE;

	if( N.height%BLOCK_SIZE != 0)
                y= N.height/BLOCK_SIZE +1;
        else
                y= N.height/BLOCK_SIZE;

	dim3 dimGrid((int) x, (int) y);
	ConvolutionKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
	  
    	cudaThreadSynchronize();
    	sdkStopTimer(&kernelTime);
    	printf ("Timp execuție kernel: %f ms\n", sdkGetTimerValue(&kernelTime));

	// Printing the result in the file
	graphs = fopen("graphs.out", "a");
	fprintf(graphs, "%lf ", sdkGetTimerValue(&kernelTime));
	fclose(graphs);

    	//TODO: copiaţi rezultatul pe host
	size_t size=  Pd.width * Pd.height * sizeof(float);
	cudaMemcpy(P.elements, Pd.elements, size ,cudaMemcpyDeviceToHost);	
    	//TODO: eliberați memoria matricelor de pe device
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}


void ConvolutionOnDeviceShared(const Matrix M, const Matrix N, Matrix PS) {
    	Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device
	FILE *graphs;

    	//pentru măsurarea timpului de execuție în kernel
    	StopWatchInterface *kernelTime = NULL;
    	sdkCreateTimer(&kernelTime);
    	sdkResetTimer(&kernelTime);
    	
	//TODO: alocați matricele de pe device
 	Md = AllocateDeviceMatrix(M.width, M.height);
        Nd = AllocateDeviceMatrix(N.width, N.height);
        Pd = AllocateDeviceMatrix(PS.width, PS.height);

    	//TODO: copiați datele de pe host (M, N) pe device (MD, Nd)
	cudaMemcpy(Md.elements, M.elements, M.width * M.height *sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Nd.elements, N.elements, N.width * N.height *sizeof(float), cudaMemcpyHostToDevice);

    	//TODO: setați configurația de rulare a kernelului
    	sdkStartTimer(&kernelTime);

	float x, y;

        if( N.width%BLOCK_SIZE != 0)
                x= N.width/BLOCK_SIZE +1;
        else
                x= N.width/BLOCK_SIZE;

        if( N.height%BLOCK_SIZE != 0)
                y= N.height/BLOCK_SIZE +1;
        else
                y= N.height/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);	
	dim3 dimGrid((int) x, (int) y);

    	//TODO: lansați în execuție kernelul  
	ConvolutionKernelShared<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
  
    	sdkStopTimer(&kernelTime);
    	printf ("Timp execuție kernel cu memorie partajată: %f ms\n", sdkGetTimerValue(&kernelTime));

	// Printing the result in the file
        graphs = fopen("graphs.out", "a");
        fprintf(graphs, "%lf ", sdkGetTimerValue(&kernelTime));
        fclose(graphs);


	size_t size=  Pd.width * Pd.height * sizeof(float);
    	
	//TODO: copiaţi rezultatul pe host
    	cudaError_t err= cudaMemcpy(PS.elements, Pd.elements, size ,cudaMemcpyDeviceToHost);
	// Debugging time. Daca apare o eroare la copiere printeaza
	if(strcmp(cudaGetErrorString(err), "no error" ) !=0 )
		printf ("----%s----\n", cudaGetErrorString(err));

	//TODO: eliberați memoria matricelor de pe device
	FreeDeviceMatrix(&Md);
        FreeDeviceMatrix(&Nd);
        FreeDeviceMatrix(&Pd);
}


// Ca sa rezolve problema cu timpul care inital nu scaleaza corect
void ConvolutionOnDeviceStarter(const Matrix M, const Matrix N, Matrix P) {
    	Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device

    	//TODO: alocați matricele de pe device
	Md = AllocateDeviceMatrix(M.width, M.height);
	Nd = AllocateDeviceMatrix(N.width, N.height);
	Pd = AllocateDeviceMatrix(P.width, P.height);

	//TODO: copiați datele de pe host (M, N) pe device (MD, Nd)
	 cudaMemcpy(Md.elements, M.elements, M.width * M.height *sizeof(float), cudaMemcpyHostToDevice);
	 cudaMemcpy(Nd.elements, N.elements, N.width * N.height *sizeof(float), cudaMemcpyHostToDevice);
    	
	//TODO: setați configurația de rulare a kernelului
	//TODO: lansați în execuție kernelul  
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	float x, y;
	if( N.width%BLOCK_SIZE != 0)
		x= N.width/BLOCK_SIZE +1;
	else
		x= N.width/BLOCK_SIZE;
	if( N.height%BLOCK_SIZE != 0)
                y= N.height/BLOCK_SIZE +1;
        else
                y= N.height/BLOCK_SIZE;

	dim3 dimGrid((int) x, (int) y);
	ConvolutionKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    	cudaThreadSynchronize();

    	//TODO: copiaţi rezultatul pe host
	size_t size=  Pd.width * Pd.height * sizeof(float);
	cudaMemcpy(P.elements, Pd.elements, size ,cudaMemcpyDeviceToHost);	
    	
	//TODO: eliberați memoria matricelor de pe device
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}



//just a random function used for testing ca sa vad daca am pus bine valorile in shares 
__device__ void danemarcaTesting(Matrix M, Matrix N, float **As, float **Bs,  int y, int x,
				int ti, int tj) {
	if(As[y][x]!= M.elements[y* M.pitch +x])
              	printf("Ceva e putred in DaneMarca\n");
       	if(Bs[y+ti][x+tj] != N.elements[(y+ti) *N.pitch + x+tj])
        	printf("Ceva e putred in DaNemarca\n");

}


// Alocă o matrice de dimensiune height*width pe device
Matrix AllocateDeviceMatrix(int width, int height) {
    	Matrix m;
    
	//TODO: alocați matricea și setați width, pitch și height
	m.width = m.pitch = width;
        m.height = height;
        size_t size= m.width* m.height* sizeof(float);
        cudaMalloc((void**) &m.elements, size);

    return m;
}

// Alocă matrice pe host de dimensiune height*width
Matrix AllocateMatrix(int width, int height) {
    	Matrix M;
    	M.width = M.pitch = width;
    	M.height = height;
    	int size = M.width * M.height;    
    	M.elements = (float*) malloc(size*sizeof(float));
    	return M;
}    

// Eliberează o matrice de pe device
void FreeDeviceMatrix(Matrix* M) {
    	cudaFree(M->elements);
    	M->elements = NULL;
}

// Eliberează o matrice de pe host
void FreeMatrix(Matrix* M) {
    	free(M->elements);
    	M->elements = NULL;
}
