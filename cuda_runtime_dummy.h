// dummy file
#include <stdio.h>
#include <stdlib.h>

/* */
enum cudaError_t
{
	cudaSuccess = 0
};

#define cudaMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define cudaMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define cudaMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

/**
* CUDA memory copy types
*/
enum cudaMemcpyKind
{
	cudaMemcpyHostToHost = 0,      /**< Host   -> Host */
	cudaMemcpyHostToDevice = 1,      /**< Host   -> Device */
	cudaMemcpyDeviceToHost = 2,      /**< Device -> Host */
	cudaMemcpyDeviceToDevice = 3,      /**< Device -> Device */
	cudaMemcpyDefault = 4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

#define __restrict__ \
        __restrict


static cudaError_t cudaDeviceSynchronize(void)
{
	return cudaSuccess;
}

template<class T>
static cudaError_t cudaMallocManaged(
	T            **devPtr,
	size_t         size,
	unsigned int   flags = cudaMemAttachGlobal
)
{
	printf("Error calling cudaMallocManaged() from dummy file!\n");
	exit(EXIT_FAILURE);
	return cudaSuccess;
}

static cudaError_t cudaFree(void *devPtr)
{
	return cudaSuccess;
}

static cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	return cudaSuccess;
}

static cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	return cudaSuccess;
}




//cublas

/* CUBLAS status type returns */
typedef enum {
	CUBLAS_STATUS_SUCCESS = 0,
	CUBLAS_STATUS_NOT_INITIALIZED = 1,
	CUBLAS_STATUS_ALLOC_FAILED = 3,
	CUBLAS_STATUS_INVALID_VALUE = 7,
	CUBLAS_STATUS_ARCH_MISMATCH = 8,
	CUBLAS_STATUS_MAPPING_ERROR = 11,
	CUBLAS_STATUS_EXECUTION_FAILED = 13,
	CUBLAS_STATUS_INTERNAL_ERROR = 14,
	CUBLAS_STATUS_NOT_SUPPORTED = 15,
	CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;


struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

static cublasStatus_t cublasCreate(cublasHandle_t *handle)
{
	return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t cublasDestroy(cublasHandle_t handle)
{
	return CUBLAS_STATUS_SUCCESS;

}

static cublasStatus_t cublasDnrm2(cublasHandle_t handle,
	int n,
	const double *x,
	int incx,
	double *result)  /* host or device pointer */
{
	return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t cublasDdot(cublasHandle_t handle,
	int n,
	const double *x,
	int incx,
	const double *y,
	int incy,
	double *result)  /* host or device pointer */
{
	return CUBLAS_STATUS_SUCCESS;
}
