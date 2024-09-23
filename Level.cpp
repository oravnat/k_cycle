#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#else
#include "cuda_runtime_dummy.h"
#endif
#include "Classes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <memory.h>
#include <iostream>
#include <iomanip>

void CheckCudaError(cudaError_t cudaStatus, const char *const file, int const line);
int CalcStride(int n);

#define CUDA_UM_ZERO_COPY

void *um_malloc(size_t size, int access_policy)
{
	//cout << "um_malloc(" << size << ", " << access_policy << ") call" << endl;
	void *ptr = nullptr;
	cudaError_t cudaStatus;
	switch (access_policy)
	{
#ifdef HAVE_CUDA
	case UM_ACCESS_GPU:
		cudaStatus = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
		CUDA_ERROR(cudaStatus);
		break;
	case UM_ACCESS_BOTH:
#ifdef CUDA_UM_ZERO_COPY
		// assumes that the direct access to sysmem is supported on this OS/GPU
		// The memory allocated by cudaMallocHost is in main-memory (CPU/host memory), but it is pinned, and in our systems the GPU have direct access to it
		cudaStatus = cudaMallocHost(&ptr, size);
		CUDA_ERROR(cudaStatus);
#else
		// default is the managed allocation with global attach
		cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
#endif
		break;
#endif
	case UM_ACCESS_CPU:
		return malloc(size);
		//cudaStatus = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
		//CUDA_ERROR(cudaStatus);
		break;
	default:
		cout << "Calling um_malloc with invalid access_policy" << endl;
	}
	return ptr;
}

void um_free(void *ptr, int access_policy)
{
	cudaError_t cudaStatus;
	switch (access_policy)
	{
#ifdef HAVE_CUDA
	case UM_ACCESS_GPU:
		cudaFree(ptr);
			break;
	case UM_ACCESS_BOTH:
#ifdef CUDA_UM_ZERO_COPY
		cudaStatus = cudaFreeHost(ptr);
		CUDA_ERROR(cudaStatus);
#else
		cudaFree(ptr);
#endif
			break;
#endif
	case UM_ACCESS_CPU:
		free(ptr);
		//cudaFree(ptr);
		break;
	}
}

Level::Level() : courserLevel(nullptr), residualStencil(nullptr), jacobiStencil(nullptr), stencil(nullptr), w(1.0), bLastLevel(false)
{
}


Level::~Level()
{
	delete courserLevel;
	ReleaeMem();
}

#ifdef HAVE_CUDA
void LogLevelSizes(size_t width, size_t height)
{
	cudaError_t cudaStatus;
	void* devPtr;
	size_t pitch;
	cudaStatus = cudaMallocPitch(&devPtr, &pitch, width*sizeof(tfloat), height);
	CUDA_ERROR(cudaStatus);
	cudaFree(devPtr);
	int stride = CalcStride(width);
	stride = stride * sizeof(tfloat);
	// the output sizes are in bytes 
	cout << "LogLevelSizes(" << width << "," << height << ") - stride=" << stride << ", pitch=" << pitch << " bytes" << endl;
}
#endif

//bool Level::BuildLevel(int n, tfloat h, bool bGpu, const tfloat* base_stencil, tfloat w, tfloat ep, int iLevelIndex, int p_access_policy)
bool Level::BuildLevel(int width, int height, bool bGpu, const tfloat stencil[3][3], tfloat w, int iLevelIndex, int p_access_policy)
{
	const int STENCIL_SIZE = 9;
		
	this->bGpu = bGpu;
	this->iLevelIndex = iLevelIndex;
	this->w = w;

	access_policy = p_access_policy;

	//LogLevelSizes(width, height);
	int stride = CalcStride(width);

	//cout << "Allocating 3 matrices, each of size " << stride*height*sizeof(tfloat) << " bytes" << endl;

	if (!u.AllocateMemory(width, height, stride, bGpu, access_policy))
		return false;

	if (!f.AllocateMemory(width, height, stride, bGpu, access_policy))
		return false;

	//cout << "Allocating temp memory for level " << n << endl;

	if (!temp.AllocateMemory(width, height, stride, bGpu, access_policy))
		return false;

	this->stencil = new tfloat[STENCIL_SIZE];
	residualStencil = (tfloat*)um_malloc(sizeof(tfloat)*STENCIL_SIZE, access_policy);
	jacobiStencil = (tfloat*)um_malloc(sizeof(tfloat)*STENCIL_SIZE, access_policy);

	for (int i = 0; i < STENCIL_SIZE; i++)
	{
		tfloat v = stencil[i / 3][i % 3];
		residualStencil[i] = v;
		this->stencil[i] = v;
		jacobiStencil[i] = w * v / stencil[1][1];
	}
	d = residualStencil[4];
	jacobiStencil[4] -= 1.0; // jacobiStencil[4] = w-1

	return true;
}

void Level::ReleaeMem()
{
	um_free(residualStencil, access_policy);
	um_free(jacobiStencil, access_policy);
	delete[] this->stencil;
}
