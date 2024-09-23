#pragma once

#define HAVE_OPEN_MP

#include <stdlib.h>
#ifdef HAVE_OPEN_MP
#include <omp.h>
#else
double omp_get_wtime();
#endif

#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#endif

using namespace std;

#define USE_DOUBLE

#ifdef USE_DOUBLE

typedef double tfloat;
#define cusparse_gtsv2_nopivot cusparseDgtsv2_nopivot

#else

typedef float tfloat;
#define cusparse_gtsv2_nopivot cusparseSgtsv2_nopivot

#endif

// access policies for UM
#define UM_ACCESS_CPU  0
#define UM_ACCESS_GPU  1
#define UM_ACCESS_BOTH 2

const int MAX_LEVELS = 14;

// custom memory management routines
void* um_malloc(size_t size, int access_policy);
void  um_free(void *ptr, int access_policy);

#ifdef HAVE_CUDA
void CheckCudaError(cudaError_t cudaStatus, const char *const file, int const line);
#define CUDA_ERROR(cudaStatus) CheckCudaError(cudaStatus, __FILE__, __LINE__)
#else
#define CUDA_ERROR(cudaStatus) if (cudaStatus != cudaSuccess) throw "Cuda Error!"
#endif

class Timer
{
	double start, finish, duration;
public:
	Timer() : start(0), finish(0), duration(0) {}
	void StartMeasure()
	{
		start = omp_get_wtime();
	}
	void StopMeasure()
	{
		finish = omp_get_wtime();
		duration += finish - start;
	}
	// GetDuration should be used only after stopping the measurment
	// Return value is in seconds
	double GetDuration()
	{
		return duration;
	}
	// Return value is in ms
	double GetRuntime()
	{
		return duration*1000.0;
	}
	double GetTimeFromLastStart()
	{
		return omp_get_wtime() - start;
	}
};

#ifdef HAVE_CUDA
class CudaTimer
{
	double runtime; // in ms
	cudaEvent_t startEvent, stopEvent;
public:
	CudaTimer() : runtime(0)
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaEventCreate(&startEvent);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaEventCreate(&stopEvent);
		CUDA_ERROR(cudaStatus);
	}
	~CudaTimer()
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaEventDestroy(startEvent);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaEventDestroy(stopEvent);
		CUDA_ERROR(cudaStatus);
	}
	void StartMeasure()
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaEventRecord(startEvent);
		CUDA_ERROR(cudaStatus);
	}
	// Also blocks the CPU for the event
	void StopMeasure()
	{
		cudaError_t cudaStatus;
		float elapsedTime; // in ms
		cudaStatus = cudaEventRecord(stopEvent);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaEventSynchronize(stopEvent);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		CUDA_ERROR(cudaStatus);
		runtime += elapsedTime;
	}
	// GetDuration should be used only after stopping the measurment
	// Return value is in seconds
	double GetDuration()
	{
		return runtime / 1000.0;
	}
	// Return value is in  ms
	double GetRuntime()
	{
		return runtime;
	}
	// Return value is in seconds
	double GetTimeFromLastStart()
	{
		cudaError_t cudaStatus;
		float elapsedTime; // in ms
		cudaStatus = cudaEventRecord(stopEvent);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		CUDA_ERROR(cudaStatus);
		return 1000.0*elapsedTime;
	}
};
#endif

enum RelaxMethod
{
	OptimalPointJacobi, PointRB, OptimalXLineJacobi, OptimalYLineJacobi, XZebra, YZebra, XYZebra, YXZebra, NUMBER_OF_RELAX_METHODS
};

class Matrix2D
{
public:
	int nx, ny;
	int count;
	int access_policy;
	int stride;
	bool bGpu;

	Matrix2D() : p(nullptr), nx(0), ny(0), count(0), stride(0) {}
	tfloat* p;
	bool AllocateMemory(int nx, int ny, int stride, bool bGpu, int access_policy = -1);
	void MakeZero();
	// may need to call ZeroBorder() after MakeRandom()
	void MakeRandom();
	void Add(const Matrix2D& R);
	void Add(const Matrix2D& R, tfloat scale);
	void Multiply(tfloat scale);
	tfloat Norm() const;
	tfloat DotProduct(const Matrix2D& R) const;
	void ZeroBorder();
	void CoptyTo(Matrix2D& to) const; // "to" must be a matrix of the same size as "this"
	void CoptyBorderTo(Matrix2D& to) const; // "to" must be a matrix of the same size as "this"
	void Swap(Matrix2D& other);
	void Transpose(Matrix2D& dest) const;
	~Matrix2D();
private:
	Matrix2D(Matrix2D&) = delete;
	void operator=(Matrix2D&) = delete;
};

class Level
{
public:
	Matrix2D u, f, temp;
	tfloat* stencil; // residual stencil, CPU memory
	tfloat* residualStencil; // residual stencil, CPU/GPU memory
	tfloat* jacobiStencil; // Jacobi stencil, CPU/GPU memory
	tfloat w; //Jacobi damping factor
	Level* courserLevel;
	bool bGpu;
	bool bLastLevel;
	int access_policy;
	int iLevelIndex; // 0 is the finest level
	tfloat d; // diagonal coefficient

	Level();
	~Level();
	bool BuildLevel(int width, int height, bool bGpu, const tfloat stencil[3][3], tfloat w, int iLevelIndex, int p_access_policy);
	void ReleaeMem();
};

class MG2D
{
public:
	int nu1, nu2;
	int nCalls;
	tfloat ep;
	tfloat ang;
	tfloat w; // Jacobi damping factor for finest level
	RelaxMethod relaxMethod;
	Level* finestLevel;
	Matrix2D solution;
	Matrix2D tmp1, tmp2;

	bool bSmoothIterationsOnGpu;

	double durationCounter;
	int countDurationForLevel;
	double msExecutionTime; // MS for last cycle/solve/average time (depends on function used)
	double msNormCalcTime;
	int NUM_SOLVES;

public:
	MG2D(RelaxMethod relaxMethod, int nu1, int nu2, tfloat ep, tfloat ang) :
		relaxMethod(relaxMethod), nu1(nu1), nu2(nu2), ep(ep), ang(ang),
		nCalls(0), w(1.0), finestLevel(nullptr), bSmoothIterationsOnGpu(false), durationCounter(0.0), countDurationForLevel(0)
	{
		//bSmoothIterationsOnGpu = true;
	}

	~MG2D() { Destroy(); }

	void CalcStencil(tfloat result[3][3], tfloat finerStencil[3][3], tfloat hx, tfloat hy);
	bool Build(int n, tfloat h);
	bool Destroy();
	void KCycle(int k, tfloat h);
	void KCycle(int k, tfloat h, Level* pLevel, int iLevel = 1);
	void Relax(Level* pLevel, int nu);
	void Relax(Level* pLevel);
	void Residual(Level* pLevel, const Matrix2D& u, const Matrix2D& f, Matrix2D& r);
	void Restrict(Level* pLevel, const Matrix2D& src, Matrix2D& dest);
	void Prolong(Level* pLevel, const Matrix2D& src, Matrix2D& dest);
	void Solve(Level* pLevel, Matrix2D& u, const Matrix2D& f);
	void Jacobi(Level* pLevel, Matrix2D& u, const Matrix2D& f); // u is both input and output
	void RedBlackGS(Level* pLevel, Matrix2D& u, const Matrix2D& f); // u is both input and output
	void XZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f); // u is both input and output
	void YZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f); // u is both input and output
	void XYZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f); // u is both input and output
	void YXZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f); // u is both input and output

	void RandomU();
	double UNorm();
	double FNorm();
	double ResidualNorm();
	double ErrorNorm();
	void PrintResidualNorm(Level* pLevel);
};

class ConjugateGradient
{
public:
	tfloat ep;
	tfloat ang;
	MG2D mg;
	Matrix2D x;
	int k;
	int nIteraions; // number of iterations in the last run
	double errorNorms[1001];
public:
	ConjugateGradient(int nu1, int nu2, tfloat ep, tfloat ang) : ep(ep), ang(ang), k(1), mg(PointRB, nu1, nu2, ep, ang), nIteraions(0)
	{

	}
	bool Build(int n, tfloat h, tfloat w);
	void Run(tfloat h);
	void RunWithPreconditioner(tfloat h, double ratioWanted, bool bStopByResidualRatio);
	void ApplyPreconditioner(Matrix2D& dest, Matrix2D& src, tfloat h);
};

struct CyclicReductionValues
{
	tfloat a, b, c;
	tfloat ra, fl, fu;
};
