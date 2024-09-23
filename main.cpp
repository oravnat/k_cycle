#ifdef HAVE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cusparse.h>

#else

#include "cuda_runtime_dummy.h"
#define CudaTimer Timer

#endif

#include <stdio.h>
#include <stdlib.h>
#include "Classes.hpp"
#include <math.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <assert.h>

#ifndef HAVE_OPEN_MP
double omp_get_wtime()
{
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    auto nowNs = now.tv_sec*1000000000ull + now.tv_nsec;
    return (nowNs / 1000000000.0);
}
#endif

// linux only:
#ifdef LINUX
#include <unistd.h>
#include <execinfo.h>
#include <sys/resource.h>
#endif

#define M_PI       3.14159265358979323846   // pi

void test_gmres(int nLevels); //temp
void run_gmres(MG2D& mg, int max_iterations = 10, double threshold = 0.001);
void CyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n);
void IterativeCyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n);
void IterativeCyclicReductionInPlace(tfloat* __restrict__ B, tfloat a, tfloat b, tfloat c, int n);


void CheckCudaError(cudaError_t cudaStatus, const char *const file, int const line)
{
	//size_t size;
	//void *array[10];
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaStatus: " << cudaStatus << " at " << file << ":" << line << endl;
#ifdef HAVE_CUDA
		if (cudaStatus == cudaErrorMemoryAllocation) //2
			cout << "cudaStatus == cudaErrorMemoryAllocation" << endl;
#endif

		/*
		// linux only:
	  	// get void*'s for all entries on the stack
  		size = backtrace(array, 10);
		// print out all the frames to stderr
  		backtrace_symbols_fd(array, size, STDERR_FILENO);
		*/

		//assert(cudaStatus == cudaSuccess);
		throw "Cuda Error!";
	}
}

//const int MAX_CYCLES = 20000;
const int MAX_CYCLES = 8000;
const int FIRST_LEVEL = 4;
int nLevels = 4;
const int NUM_K = 5;
//const int NUM_K = 1;
const int FIRST_K = 0;
int last_k = 4;
const char* methodNames[] = {
	"(v-cycle)",
	"(f-cycle)",
	"(k3)     ",
	"(k4)     ",
	"(w-cycle)" };

int kk[] = { 1, 2, 3, 4, 99 };
//int kk[] = { 0 };
const double epsilon = 1e-5;
double angleDeg = 45;
double angle = angleDeg * (M_PI / 180);

//single float variables have 7.22 decimal digits
//double float variables have 15.95 decimal digits
const double ratioWanted = 1e8; // reidual/error norm reduce factor wanted
const bool bStopByResidualRatio = false; // true for residual norm, false for error norm
// important!: we may have problems when bStopByResidualRatio=false and problemIndex > 0, because of a mix of a continuous solution and a discrete solution!
//const double rtol = 1e-10; // relative tolerance, the algorithm is considered converged if ||b-Ax|| / ||b|| < rtol, if x0 is zero and bStopByResidualRatio is true, this is equivalent to ratioWanted
const double rtol = 0; // relative tolerance, the algorithm is considered converged if ||b-Ax|| / ||b|| < rtol, if x0 is zero and bStopByResidualRatio is true, this is equivalent to ratioWanted

bool bUseGpu = false;
int problemIndex = 0; // problemIndex>0 is usually used with bStopByResidualRatio=true
RelaxMethod relaxMethod = OptimalPointJacobi;
//RelaxMethod relaxMethod = XZebra;
//RelaxMethod relaxMethod = YZebra;
//RelaxMethod relaxMethod = XYZebra;
//RelaxMethod relaxMethod = YXZebra;
////RelaxMethod relaxMethod = PointRB;
const double epsilons[] = { 1, 1e-1, 1e-2, 1e-3, 1e-4 };
const int COUNT_DURATION_FOR_LEVEL = 7;
const char* testName = "";
bool bAlwaysCalcUNrom = false;
bool bAlwaysCalcErrorNrom = false;
bool bAlwaysCalcResidualNrom = false;
bool bGalerkin = false;
bool bSemiCoarsening = false;
bool bSPAI0 = false;
bool bHybrid = false;
bool bRelaxSolve = false;


#ifdef HAVE_CUDA
bool CudaInc(tfloat* __restrict__ p, int nBlocks, int nThreads, cudaStream_t stream);
bool CudaNone(tfloat* __restrict__ p, int nBlocks, int nThreads, cudaStream_t stream);
#endif
void deviceQuery();

/*
memory requirements:
matrix size at finest level: (2^MAX_LEVELS+1)^2
default tfloat size (default is double): 8 bytes
number of matrices per level: 3
=> memory required for level l: ((2^l+1)^2) * 8 * 3 == 24 * 4^l bytes
l = 12 => memory > 24 * 4^12 = 402,653,184 bytes
l = 13 => memory > 24 * 4^13 = 1.6106e+09 bytes
if m bytes are required for the largest level, about m*(4/3) total bytes are required for all the levels, because:
m + m/4 + m/16 + ... = m * (4/3)
*/

struct Stat
{
	double duration;
	double durationInLevel;
	double msRuntime;
	int nLevels;
	int nCycles;
	double startNorm; // L2 error norm
	double finishNorm; // L2 error norm
	double startErrorNorm; // L2 error norm
	double finishErrorNorm; // L2 error norm
	int k;
	double norms[MAX_CYCLES + 1];
	double errorNorms[MAX_CYCLES + 1];
	double residualNorms[MAX_CYCLES + 1];
	double normCalcDuration;
	double startResidualNorm; // L2 residual norm
	double finishResidualNorm; // L2 residual norm
	bool bTimeOut;
};

int CalcStride(int n)
{
	//const int ALIGMENT = 32; // in tfloat
	//const int ALIGMENT = 64; // in tfloat, when sizeof(t_float) = sizeof(double) = 8, ALIGMENT=64 means 8*64=512 bytes, which seems to be good for current architectures
	//const int ALIGMENT = 128; // in tfloat, when sizeof(t_float) = sizeof(float) = 8, ALIGMENT=128 means 4*128=512 bytes, which seems to be good for current architectures
	const int ALIGMENT_BYTES = 512; // in bytes

	int aligmnet = ALIGMENT_BYTES / sizeof(tfloat);

	int result = n;
	while (result % aligmnet)
		result++;
	return result;
}

void SaveToFile(const char* filename, MG2D& cyc)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();

	ofstream file(filename);
	Matrix2D& mat = cyc.finestLevel->u;
	int count = mat.count;
	for (int i = 0; i < count; i++)
		file << mat.p[i] << " ";
}

// Print double array to standard output, for ploting in matlab, including line end:
void PrintArray(double a[], int n)
{
	cout << "[";
	for (int i = 0; i < n; i++)
	{
		cout << a[i];
		if (i < n - 1)
			cout << ", ";
	}
	cout << "];" << endl;
}

void PrintRelaxMethod()
{
	if (relaxMethod == PointRB)
		cout << "relaxMethod = PointRB;" << endl;
	else if (relaxMethod == OptimalPointJacobi)
		cout << "relaxMethod = OptimalPointJacobi;" << endl;
	else if (relaxMethod == XZebra)
		cout << "relaxMethod = XZebra;" << endl;
	else if (relaxMethod == YZebra)
  		cout << "relaxMethod = YZebra;" << endl;
	else if (relaxMethod == XYZebra)
		cout << "relaxMethod = XYZebra;" << endl;
	else if (relaxMethod == YXZebra)
		cout << "relaxMethod = YXZebra;" << endl;
	else
		cout << "unknown relaxMethod!" << endl;
	if (bSemiCoarsening)
		cout << "Semi-Coarsening" << endl;
	else
		cout << "Standard-Coarsening" << endl;
	if (bRelaxSolve)
		cout << "Relax Solve" << endl;
}

tfloat Ax(int idx, int stride, tfloat* stencil, Matrix2D& x)
{
	tfloat result =
		(stencil[0] * x.p[idx - stride - 1] + stencil[1] * x.p[idx - stride] + stencil[2] * x.p[idx - stride + 1]
		+ stencil[3] * x.p[idx - 1] + stencil[4] * x.p[idx] + stencil[5] * x.p[idx + 1]
		+ stencil[6] * x.p[idx + stride - 1] + stencil[7] * x.p[idx + stride] + stencil[8] * x.p[idx + stride + 1]);
	return result;
}

// important note!: the soulution of problemIndex 1 and 2 by this method is the solution of the continuous problem,
// and not of the discrete problem. This means that the error norm may start to be increased after some iterations.
bool PrepareProblem(MG2D& mg, int n, double epsilon, double angle)
{
	int stride = CalcStride(n + 1);
	const tfloat C = cos(angle);
	const tfloat S = sin(angle);
	const tfloat C2 = C*C;
	const tfloat S2 = S*S;
	// problem 0: zero right side (zero solution). Random (but consistent) init guess.
	if (problemIndex == 0)
	{
		srand(1);
		mg.RandomU();
	}
	else if (problemIndex == 1)
	{
		// u = sin(pi*x)+sin(pi*y)
		// zero initial guess
		// x = (0,1), y=(0,1)
		// Uxx = - pi^2 * sin(pi*x)
		// Uyy = - pi^2 * sin(pi*y)
		// Uxy = 0
		// Lu = (C^2+ep*S^2)Uxx + 2(1-ep)CSUxy + (ep*C^2+S^2)Uyy =
		//    = -(C^2+ep*S^2) * pi^2 * sin(pi*x)
		//    + -(ep*C^2+S^2) * pi^2 * sin(pi*y)

		if (!mg.solution.AllocateMemory(n + 1, n + 1, stride, mg.finestLevel->bGpu))
			return false;
		tfloat* p = mg.finestLevel->f.p;
		tfloat* up = mg.finestLevel->u.p;
		tfloat* sp = mg.solution.p;

		//srand(1);
		//mg.RandomU();
		mg.finestLevel->u.MakeZero();

		if (mg.finestLevel->bGpu)
		{
			cudaError_t cudaStatus;
			cudaStatus = cudaDeviceSynchronize();
		}

		// prepare right side:
		// ix = 0 = > x = 0.0; ix = n => x = 1.0
		for (int iy = 0; iy < n + 1; iy++)
		{
			for (int ix = 0, idx = iy*stride + 0; ix < n + 1; ix++, idx++)
			{
				double x = (double)ix / n;
				double y = (double)iy / n;

				double u = sin(M_PI*x) + sin(M_PI * y);
				double f = -(C2 + epsilon*S2) * M_PI * M_PI * sin(M_PI * x);
				f += -(epsilon*C2 + S2)* M_PI * M_PI * sin(M_PI * y);

				sp[idx] = u;
				//up[idx] = u; // for tests: already start with the solution
				if (ix > 0 && iy > 0 && ix < n && iy < n) // the boundaries of f must be 0
					p[idx] = f;

				/*if (ix == 0)
				{
					cout << x << " ";
				}*/
			}
		}
		// prepare boundary conditions:
		for (int i = 0; i < n + 1; i++)
		{
			up[i] = sp[i]; //sp[0][i]
			up[i + stride*n] = sp[i + stride*n]; //sp[n][i]
			up[i*stride] = sp[i*stride]; //sp[i][0]
			up[n + i*stride] = sp[n + i*stride]; //sp[i][n]

			//cout << up[i] << " ";
		}
		//if (mg.finestLevel->u2.n > 0)
			//mg.finestLevel->u.CoptyTo(mg.finestLevel->u2);

	}
	else if (problemIndex == 2)
	{
		// u = 2sin(pi*x)+sin(2*pi*y)
		// zero initial guess
		// x = (0,1), y=(0,1)
		// Uxx = - 2 * pi^2 * sin(pi*x)
		// Uyy = - 4 * pi^2 * sin(2*pi*y)
		// Uxy = 0
		// Lu = (C^2+ep*S^2)Uxx + 2(1-ep)CSUxy + (ep*C^2+S^2)Uyy =
		//    = -2*(C^2+ep*S^2) * pi^2 * sin(pi*x)
		//    + -4*(ep*C^2+S^2) * pi^2 * sin(2*pi*y)

		if (!mg.solution.AllocateMemory(n + 1, n + 1, stride, mg.finestLevel->bGpu))
			return false;
		tfloat* p = mg.finestLevel->f.p;
		tfloat* up = mg.finestLevel->u.p;
		tfloat* sp = mg.solution.p;

		//srand(1);
		//mg.RandomU();
		mg.finestLevel->u.MakeZero();

		if (mg.finestLevel->bGpu)
		{
			cudaError_t cudaStatus;
			cudaStatus = cudaDeviceSynchronize();
		}

		// prepare right side:
		// ix = 0 = > x = 0.0; ix = n => x = 1.0
		for (int iy = 0; iy < n + 1; iy++)
		{
			for (int ix = 0, idx = iy * stride + 0; ix < n + 1; ix++, idx++)
			{
				double x = (double)ix / n;
				double y = (double)iy / n;

				double u = 2 * sin(M_PI*x) + sin(2 * M_PI * y);
				double f = -2 * (C2 + epsilon*S2) * M_PI * M_PI * sin(M_PI * x);
				f += -4 * (epsilon*C2 + S2)* M_PI * M_PI * sin(2 * M_PI * y);

				sp[idx] = u;
				if (ix > 0 && iy > 0 && ix < n && iy < n) // the boundaries of f must be 0
					p[idx] = f;

			}
		}
		// prepare boundary conditions:
		for (int i = 0; i < n + 1; i++)
		{
			up[i] = sp[i]; //sp[0][i]
			up[i + stride * n] = sp[i + stride * n]; //sp[n][i]
			up[i * stride] = sp[i * stride]; //sp[i][0]
			up[n + i * stride] = sp[n + i * stride]; //sp[i][n]
		}

		//if (mg.finestLevel->u2.n > 0)
			//mg.finestLevel->u.CoptyTo(mg.finestLevel->u2);
	}

	if (problemIndex > 0)
	{
		tfloat* p = mg.finestLevel->f.p;

		// here, the value 0.0 remains on the boundary of u in the entire cycle
		// f is modified in order to include the boundary conditions
		for (int i = 1; i < n; i++)
		{
			int idx;
			idx = stride + i;
			p[idx] -= Ax(idx, stride, mg.finestLevel->residualStencil, mg.finestLevel->u);
			idx = stride * (n - 1) + i;
			p[idx] -= Ax(idx, stride, mg.finestLevel->residualStencil, mg.finestLevel->u);
			if (i > 1 && i < n - 1)
			{
				idx = i * stride + 1;
				p[idx] -= Ax(idx, stride, mg.finestLevel->residualStencil, mg.finestLevel->u);
				idx = i * stride + n - 1;
				p[idx] -= Ax(idx, stride, mg.finestLevel->residualStencil, mg.finestLevel->u);
			}
		}
		mg.solution.ZeroBorder(); // this is important, because most functions assume zero border
		mg.finestLevel->u.MakeZero();
	}

	// for tests:
	//mg.finestLevel->u.Transpose(mg.finestLevel->temp);
	//mg.finestLevel->u.Swap(mg.finestLevel->temp);

	return true;
}

double TestKCycleTime(MG2D& cyc, int n_cycles, tfloat h, int k)
{
	double result = 0;
	const int NUM_SOLVES = 1;
	int countDurationForLevel = cyc.countDurationForLevel;
	Timer timer;
	CudaTimer cudatimer;

	cyc.NUM_SOLVES = NUM_SOLVES;

	// first pass is a warm up, in the second pass we measure times
	for (int doTiming = 0; doTiming <= 1; doTiming++) // todo: move out of the function
	{
		for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)  // todo: move out of the function
		{

			if (doTiming)
			{
				timer.StartMeasure();
				if (bUseGpu)
					cudatimer.StartMeasure();
				cyc.countDurationForLevel = countDurationForLevel;
			}
			else
				cyc.countDurationForLevel = 0;
			for (int iter = 1; iter <= n_cycles; iter++)
			{
				cyc.KCycle(k, h);
			}

			if (doTiming)
			{
				timer.StopMeasure();
				if (bUseGpu)
					cudatimer.StopMeasure();  // a synchronized call
			}
		}
	}

	if (bUseGpu)
		cyc.msExecutionTime = cudatimer.GetRuntime() / NUM_SOLVES;
	else
		cyc.msExecutionTime = timer.GetRuntime() / NUM_SOLVES;
	return result;
}


void TestKCycleTime(int nLevels, Stat& stat, int k)
{
	cudaError_t cudaStatus;
	MG2D cyc(relaxMethod, 2, 2, 1.0, 0.0); // 6, 6 pre/post relaxations is the default for Jacobi in Multigrid2DCuda

	int n_cycles = 200;
	int n = 1 << nLevels;
	tfloat h = 1.0 / n;

	stat.nCycles = n_cycles;

	//cout << "Building cycle (calling cyc.Build(n, h))" << endl;

	cyc.w = 2.0 / 3.0;
	bool bOK = cyc.Build(n, h);

	if (!bOK)
		cout << "cyc.Build(n, h) failed!" << endl;

	//cout << "Cycle has been built" << endl;

	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);
	cyc.RandomU();
	stat.startNorm = cyc.UNorm();
	stat.startResidualNorm = cyc.ResidualNorm();
	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);

	cyc.countDurationForLevel = COUNT_DURATION_FOR_LEVEL; // finest level is 1

	//stat.duration = TestKCycleTime(cyc, n_cycles, h, k) / n_cycles;
	//stat.durationInLevel = cyc.durationCounter / n_cycles / NUM_SOLVES;
	// it is a mistake to divide by n_cycles here!!!
	// there is a division in the calling method.
	// this error began on 14/07/2019 and fixed on 28/07/2019
	TestKCycleTime(cyc, n_cycles, h, k);
	stat.duration = -1;
	stat.durationInLevel = cyc.durationCounter / cyc.NUM_SOLVES;
	stat.msRuntime = cyc.msExecutionTime;

	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);
	stat.finishNorm = cyc.UNorm();
	stat.finishResidualNorm = cyc.ResidualNorm();
}

void TestKCycleTime(const int FIRST_LEVEL, const int MAX_LEVELS)
{
	Stat stat;
	Stat topLevelStat[NUM_K];

	double totalTime[NUM_K][30];
	double averageCycleTime[NUM_K][30] = { 0.0 }; // starting from level 1, in unused levels there will be a value of 0.0
	int nCycles[NUM_K][30];
	for (int k = FIRST_K; k <= last_k; k++)
	{
		for (int nLevels = MAX_LEVELS; nLevels >= FIRST_LEVEL; nLevels--)
		{
			cout << "Testing " << methodNames[k] << " with " << nLevels << " levels " << endl;
			TestKCycleTime(nLevels, stat, kk[k]);
			totalTime[k][nLevels] = stat.msRuntime / 1000.0;
			averageCycleTime[k][nLevels] = stat.msRuntime / stat.nCycles / 1000.0;
			nCycles[k][nLevels] = stat.nCycles;

			if (nLevels == MAX_LEVELS)
			{
				cout << "Initial norm: "  << stat.startNorm <<  endl;
				cout << "Final norm: " << stat.finishNorm << endl;
				cout << "Average time for one cycle by cuda events: " << stat.msRuntime / stat.nCycles / 1000.0 << endl;
				// The next 2 lines are valid only when using the CPU for the calculations:
				cout << "Average time for level " << COUNT_DURATION_FOR_LEVEL << " and coarser levels: " << stat.durationInLevel / stat.nCycles << endl;;
				cout << "Fraction spent in coarser levels: " << setprecision(1) << (stat.durationInLevel / (stat.msRuntime / 1000.0)) * 100.0 << "%" << endl;
				cout << setprecision(6);
				topLevelStat[k] = stat;
			}
		}
		cout << endl;
	}

	cout << "\n\n\n";
	cout << "=== k-cycle results ===" << endl;
	cout << fixed;
	for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
	{
		int dim = (1 << nLevels) - 1;
		cout << "levels: " << nLevels << ", size: " << dim << " ^ 2 = " << dim * dim << endl;
		cout << "k             | average time(ms)" << endl;
		cout << "-----------------------------" << endl;
		for (int k = FIRST_K; k <= last_k; k++)
		{
			//printf("%d %s   | %12.6f\n", kk[k], methodNames[k], averageCycleTime[k][nLevels] * 1000.0);
			cout << kk[k] << " " << methodNames[k] << "   | " << setw(12) << averageCycleTime[k][nLevels] * 1000.0 << endl;
		}
		cout << endl;
	}
	cout << "\n\n\n";
	for (int k = FIRST_K; k <= last_k; k++)
	{
		cout << "k" << kk[k] << " = [";
		for (int nLevels = 1; nLevels <= MAX_LEVELS; nLevels++)
		{
			//printf("%f", averageCycleTime[k][nLevels] * 1000.0);
			cout << averageCycleTime[k][nLevels] * 1000.0;
			if (nLevels < MAX_LEVELS)
				cout << ",";
		}
		cout << "];" << endl;
	}
	cout << "\n\n\n" << endl;

	if (!bUseGpu)
	{
		cout << "Durations spent in coarsest grids:" << endl;
		cout << "k              | average duration(ms)   | levels 1" << "-" << COUNT_DURATION_FOR_LEVEL << " | fraction " << endl;
		cout << "-----------------------------------------------------------------" << endl;
		for (int k = FIRST_K; k <= last_k; k++)
		{
			cout << setprecision(4);
			cout << fixed;
			cout << setw(2) << kk[k] << "             | " << setw(22) << topLevelStat[k].msRuntime / topLevelStat[k].nCycles << " | " << setw(10) << topLevelStat[k].durationInLevel * 1000.0 / topLevelStat[k].nCycles << " | " << setprecision(1) << (topLevelStat[k].durationInLevel / (topLevelStat[k].msRuntime / 1000.0)) * 100 << "%" << endl;
			cout << setprecision(6);
		}
		cout << "\n\n\n" << endl;
	}
}

// The function calculates the optimal smoothing factor for relaxMethod
// currently assuming full coarsening for all methods (including line relaxations) - except for xzebra_semi_y_coarsening
// see "Multigrid smoothing for symmetric nine-point stencils" by Irad Yavneh and Elena Olvovsky: https://doi.org/10.1016/S0096-3003(97)10035-2
double calcSmoothing(RelaxMethod relaxMethod, double ep, double ang, bool printInfo, bool bPrintSelectInfo)
{
	double result = 1.0;
	double mu = 0.0;

	// damped Jacobi iteration, A >= C
	double sn = sin(ang);
	double csn = cos(ang);
	double S2 = sn*sn;
	double C2 = csn*csn;
	double D = 0;
	//A = csn ^ 2 + ep*sn ^ 2;
	//C = ep*csn ^ 2 + sn ^ 2;
	double A = ep*C2 + S2;
	double C = C2 + ep*S2;
	double B = (1 - ep) * sn * csn;
	double sm = A + C + D;
	A = A / sm;
	B = B / sm;
	C = C / sm;
	D = D / sm;
	double maxAC = max(A, C);
	double AC2 = maxAC*maxAC;

	// for B = 0:
	//smin = -A - abs(D - C);
	//smax = max(A, abs(D - C) - A);

	// for D = 0:
	// When using Galerkin, D will not be 0 on coarser grids
	// but I don't know how to calculate optimally damped point Jacobi when both
	// B and D are not 0
	double smin = -1;
	double smax = sqrt(maxAC *maxAC + B * B); // for point Jacobi
	double pjw = 2 / (2 - (smax + smin));
	double pjmu = (smax - smin) / (2 - (smax + smin));
	smax = max(sqrt(C*C + B*B), abs(B) / sqrt(1 - A*A)); // for x-line Jacobi - full coarsening
	double ljw = 2 / (2 - (smax + smin));
	double ljmu = (smax - smin) / (2 - (smax + smin));
	smax = max(sqrt(A*A + B*B), abs(B) / sqrt(1 - C*C)); // for y-line Jacobi - full coarsening
	double yljw = 2 / (2 - (smax + smin));
	double yljmu = (smax - smin) / (2 - (smax + smin));

	double RBmu = max(0.5 * (AC2 + sqrt(AC2*AC2 + 4 * B*B)), D); // for point RB, assuming A >= C

	double f = abs(B)*(abs(B) + sqrt(1 - A*A)) / (2 * (1 - A*A)); // for x-line RB
	double xlineRB = max(1.0 / 8.0, C*C + B*B); // for x-line RB(x - zebra)
	xlineRB = max(xlineRB, f); // x-zebra (full coarsening)
	double xzebra_semi_y_coarsening = max(1.0/8.0, f); // x-zebra (semi coarsening)

	f = abs(B)*(abs(B) + sqrt(1 - C*C)) / (2 * (1 - C*C)); // for y-line RB
	double ylineRB = max(1.0 / 8.0, A*A + B*B); // for y-line RB(y - zebra)
	ylineRB = max(ylineRB, f); // y-zebra (full coarsening)

	if (printInfo)
	{
		cout << "optimal w: " << pjw << ", optimal mu for point Jacobi: " << pjmu << endl;
		cout << "optimal w: " << ljw << ", optimal mu for x-line Jacobi: " << ljmu << endl;
		cout << "optimal w: " << yljw << ", optimal mu for y-line Jacobi: " << yljmu << endl;
		cout << "optimal mu for point RB: " << RBmu << endl;
		cout << "optimal mu for x-line RB: " << xlineRB << endl;
		cout << "optimal mu for y-line RB: " << ylineRB << endl;
		cout << "optimal mu for x-line RB and semi-coarsening: " << xzebra_semi_y_coarsening << endl;
	}

	if (relaxMethod == OptimalPointJacobi) // point jacobi
	{
		result = pjw;
		mu = pjmu;
	}
	else if (relaxMethod == OptimalXLineJacobi) // line jacobi on x coordinate
	{
		result = ljw;
		mu = ljmu;
	}
	else if (relaxMethod == OptimalYLineJacobi) // line jacobi on y coordinate
	{
		result = yljw;
		mu = yljmu;
	}
	else if (relaxMethod == PointRB)
		mu = RBmu;
	else if (relaxMethod == XZebra)
		mu = xlineRB;
	else if (relaxMethod == YZebra)
		mu = ylineRB;

	if (bPrintSelectInfo)
	{
		cout << "Expected smoothing factor: " << mu << endl;
	}

	return result;
}

double SolveByFixedNumberOfCycles(Stat& stat, int n_cycles, int nLevels, int k, int nu1, int nu2, double epsilon)
{
	double result = 0;
	double sum = 0;
	double norm = 0.0;
	double errorNorm = 0.0;
	double residualNorm = 0.0;

	cudaError_t cudaStatus;
	Timer normCalcTime;
	Timer solveTimer;
	MG2D cyc(relaxMethod, nu1, nu2, epsilon, angle);

	int n = 1 << nLevels;
	tfloat h = 1.0 / n;

	double w = calcSmoothing(cyc.relaxMethod, epsilon, angle, false, false);
	cyc.w = w;
	bool bOK = cyc.Build(n, h);

	if (cyc.finestLevel->bGpu)
	{
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
	}

	PrepareProblem(cyc, n, epsilon, angle);

	stat.startNorm = cyc.UNorm();
	stat.startResidualNorm = cyc.ResidualNorm();
	stat.startErrorNorm = cyc.ErrorNorm();
	stat.norms[0] = stat.startNorm;

	stat.nCycles = n_cycles;
	solveTimer.StartMeasure();
	for (int iter = 1; iter <= n_cycles; iter++)
	{
		cyc.KCycle(k, h);

		//normCalcTime.StartMeasure();
		//errorNorm = cyc.ErrorNorm();
		//normCalcTime.StopMeasure();

		stat.norms[iter] = norm;
		stat.errorNorms[iter] = errorNorm;
		stat.residualNorms[iter] = residualNorm;
	}

	if (cyc.finestLevel->bGpu)
	{
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
	}

	solveTimer.StopMeasure();

	//stat.normCalcDuration = normCalcTime.GetDuration();
	stat.duration = solveTimer.GetDuration();

	stat.finishNorm = cyc.UNorm();
	stat.finishResidualNorm = cyc.ResidualNorm();
	stat.finishErrorNorm = cyc.ErrorNorm();

	return result;
}

void TestFixedNumberOfIterations(int nLevels)
{
	const int N_PRE_RELAXATIONS = 2;
	const int N_POST_RELAXATIONS = 2;
	//const int nEps = sizeof(epsilons) / sizeof(epsilons[0]);
	const int nEps = 1;
	Stat stats[NUM_K][nEps];
	int n_cycles[] = { 3906 , 928, 494 , 403, 391 }; // for 11 levels and epsilon = 0.0001
	//int n_cycles[] = { 6909 , 1403, 651 , 495, 470 }; // for 12 levels and epsilon = 0.0001

	cout << "nLevels: " << nLevels << endl;
	cout << "N_PRE_RELAXATIONS: " << N_PRE_RELAXATIONS << endl;
	cout << "N_POST_RELAXATIONS: " << N_POST_RELAXATIONS << endl;

	for (int iEps = 0; iEps < nEps; iEps++)
	{
		double epsilon = epsilons[iEps];
		epsilon = 0.0001;
		cout << "epsilon=" << epsilon << endl;
		for (int k = FIRST_K; k <= last_k; k++)
		{
			Stat& stat = stats[k][iEps];
			cout << "Testing " << methodNames[k] << " with " << nLevels << " levels" << endl;
			SolveByFixedNumberOfCycles(stat, n_cycles[k], nLevels, kk[k], N_PRE_RELAXATIONS, N_POST_RELAXATIONS, epsilon);

			cout << "finishNorm: " << stat.finishNorm << endl;
			cout << "finishResidualNorm: " << stat.finishResidualNorm << endl;
			cout << "finishErrorNorm: " << stat.finishErrorNorm << endl;
		}
	}

	cout << endl << endl << endl;
	/*cout << "eps = [";
	for (int iEps = 0; iEps < nEps; iEps++)
	{
		cout << epsilons[iEps];
		if (iEps < nEps - 1)
			cout << ", ";
	}
	cout << "];" << endl;*/
	for (int k = FIRST_K; k <= last_k; k++)
	{
		cout << "average_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			double averageCycleTime = stat.duration / stat.nCycles;
			cout << averageCycleTime * 1000.0;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		cout << "total_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			cout << stat.duration * 1000.0;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		cout << "nCycles_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			cout << stat.nCycles;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		/*cout << "totalNormTime_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			double time = stat.normCalcDuration;
			cout << time * 1000.0;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;
		cout << "fraction of norm calc time" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			double frac = stat.normCalcDuration / stat.duration;
			cout << frac;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;*/

	}
	cout << endl << endl << endl;
}

double Solve(MG2D& cyc, Stat& stat, double stopThreshold, tfloat h, int k, int max_cycles = MAX_CYCLES)
{
	double result = 0;
	double sum = 0;
	double unorm = 0.0;
	double norm = 0.0;
	double first_norm = 0.0;
	double prev_norm = 0.0;
	double errorNorm = 0.0;
	double residualNorm = 0.0;
	bool bStopByTimeLimit = true;
	double timeLimit = 1800; // seconds
	//double timeLimit = 60; // seconds
	double norm_of_F = 1.0;
	double rel_norm = 0.0;

	cudaError_t cudaStatus;
	Timer solveTimer;
	CudaTimer normCalcTime;
	CudaTimer cudaTimer;

	norm_of_F = cyc.FNorm(); // 0 for problem index 0
	stat.nCycles = max_cycles;
	stat.bTimeOut = false;
	solveTimer.StartMeasure();
	cudaTimer.StartMeasure();

	//run_gmres(cyc);

	for (int iter = 1; iter <= max_cycles; iter++)
	{
 		cyc.KCycle(k, h);
		//cyc.Relax(cyc.finestLevel);

		//normCalcTime.StartMeasure();
		// The norm calculation calls block the CPU (synchronized calls)
		if (bAlwaysCalcUNrom)
			unorm = cyc.UNorm();
		if (bAlwaysCalcErrorNrom || !bStopByResidualRatio)
			errorNorm = cyc.ErrorNorm();
		if (bAlwaysCalcResidualNrom || bStopByResidualRatio)
			residualNorm = cyc.ResidualNorm();
		//normCalcTime.StopMeasure(); // also a synchronized call

		stat.norms[iter] = unorm;
		stat.errorNorms[iter] = errorNorm;
		stat.residualNorms[iter] = residualNorm;

		if (bStopByResidualRatio)
			norm = residualNorm;
		else
			norm = errorNorm;

		bool bQuit = false;

		if (iter == 1)
			first_norm = norm;
		else 
		{
			if (norm > first_norm)
			{
				cout << "Error: the norm is increasing!" << endl;
				cout << "first_norm: " << first_norm << ", norm: " << norm << endl;
				bQuit = true;
			}
			if (norm > prev_norm)
			{
				cout << "Error: the norm is increasing!" << endl;
				cout << "prev_norm: " << prev_norm << ", norm: " << norm << endl;
				bQuit = true;
			}
		}
		prev_norm = norm;

		//cout << iter << ") norm: " << norm << endl;

		if (norm <= stopThreshold)
			bQuit = true;
		if (isnan(norm))
			bQuit = true;
		if (bStopByTimeLimit)
		{
			if (solveTimer.GetTimeFromLastStart() >= timeLimit)
			{
				cout << "Timeout!" << endl;
				stat.bTimeOut = true;
				bQuit = true;
			}
		}
		if (problemIndex > 0)
		{
			double norm_of_residual = residualNorm;
			rel_norm = norm_of_residual / norm_of_F;
			if (rtol > 0 && rel_norm < rtol)
				bQuit = true;

			//cout << "norm_of_residual: " << norm_of_residual << endl;
			//cout << "rel_norm: " << rel_norm << endl;
		}

		if (bQuit)
		{
			stat.nCycles = iter;
			break;
		}
	}

	cudaTimer.StopMeasure(); // synchronized call

	solveTimer.StopMeasure();

	//cout << "solveTimer: " << solveTimer.GetDuration() << endl;

	result = solveTimer.GetDuration();
	cyc.msExecutionTime = cudaTimer.GetRuntime();
	cyc.msNormCalcTime = normCalcTime.GetRuntime();
	stat.normCalcDuration = normCalcTime.GetDuration();

	//cout << "normCalcTime.GetDuration(): " << normCalcTime.GetDuration() << endl;

	//cout << "rel_norm: " << rel_norm << endl;

	return result;
}

void SolveTime(int nLevels, Stat& stat, int k, int nu1, int nu2, double epsilon, double ratioWanted, int max_cycles = MAX_CYCLES)
{
	cudaError_t cudaStatus;
	MG2D cyc(relaxMethod, nu1, nu2, epsilon, angle);

	int n = 1 << nLevels;
	tfloat h = 1.0 / n;

	double w = calcSmoothing(cyc.relaxMethod, epsilon, angle, false, true);
	//cyc.w = 2.0 / 3.0;
	cyc.w = w;
	bool bOK = cyc.Build(n, h);

	if (cyc.finestLevel->bGpu)
	{
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
	}

	PrepareProblem(cyc, n, epsilon, angle);
	//cyc.finestLevel->u.CoptyTo(cyc.finestLevel->u2);

	stat.startNorm = cyc.UNorm();
	stat.norms[0] = stat.startNorm;
	stat.startResidualNorm = cyc.ResidualNorm();
	stat.residualNorms[0] = stat.startResidualNorm;
	stat.startErrorNorm = cyc.ErrorNorm();
	stat.errorNorms[0] = stat.startErrorNorm;

	if (problemIndex > 0)
	{
		tfloat norm_of_F = cyc.FNorm(); // 0 for problem index 0
		double norm_of_residual = cyc.ResidualNorm();
		double rel_norm = norm_of_residual / norm_of_F;
		cout << "Initial relative norm: " << rel_norm << endl; // should be 1 when x0 is zero
	}

	double threshold = stat.startErrorNorm / ratioWanted;
	if (bStopByResidualRatio)
		threshold = stat.startResidualNorm / ratioWanted;
	stat.duration = Solve(cyc, stat, threshold, h, k, max_cycles);
	stat.msRuntime = cyc.msExecutionTime;

	if (cyc.finestLevel->bGpu)
	{
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
	}
	stat.finishNorm = cyc.UNorm();
	stat.finishResidualNorm = cyc.ResidualNorm();
	stat.finishErrorNorm = cyc.ErrorNorm();

	/*
	// outputs for verifying number of function calls:
	cout << "Number of iterations: " << stat.nCycles << endl;
	cout << "Number of levels: " << nLevels << endl;
	cout << "kappa: " << k << endl;
	cout << "Number of function calls: " << cyc.nCalls << endl;
	*/
	//SaveToFile("ErrorAfterSolve.txt", cyc);
}

// old function, should not be used
void TestKCycleSolveTime()
{
	const int N_PRE_RELAXATIONS = 2;
	const int N_POST_RELAXATIONS = 2;
	Stat stats[NUM_K][MAX_LEVELS + 1];

	double totalTime[NUM_K][30];
	double averageCycleTime[NUM_K][30];
	int nCycles[NUM_K][30];
	double normTime[NUM_K][30];
	double lastRatio[NUM_K][30];
	for (int k = FIRST_K; k <= last_k; k++)
	{
		for (int nLevels = MAX_LEVELS; nLevels >= FIRST_LEVEL; nLevels--)
		{
			Stat& stat = stats[k][nLevels];
			cout << "Testing " << methodNames[k] << " with " << nLevels << " levels" << endl;
			SolveTime(nLevels, stat, kk[k], N_PRE_RELAXATIONS, N_POST_RELAXATIONS, epsilon, ratioWanted);
			//SolveTime(nLevels, stat, kk[k], N_PRE_RELAXATIONS, N_POST_RELAXATIONS, epsilon, ratioWanted, 300);
			totalTime[k][nLevels] = stat.duration;
			averageCycleTime[k][nLevels] = stat.duration / stat.nCycles;
			normTime[k][nLevels] = stat.normCalcDuration;

			nCycles[k][nLevels] = stat.nCycles;
			int nCyc = stat.nCycles;
			lastRatio[k][nLevels] = stat.norms[nCyc] / stat.norms[nCyc - 1];

			if (nLevels == MAX_LEVELS)
			{
				cout << "Initial norm: " << stat.startNorm << endl;
				cout << "Final norm: " << stat.finishNorm << endl;
				cout << "Number of cycles: " << stat.nCycles << endl;
				cout << "Timeout: " << stat.bTimeOut << endl;
			}

			//break;
		}
	}

	printf("\n\n\n");
	printf("=== k-cycle results ===\n");
	for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
	{
		//nLevels = MAX_LEVELS;

		//int dim = dims[0][i];
		int dim = (1 << nLevels) - 1;
		printf("levels: %d, size: %d ^ 2 = %d\n", nLevels, dim, dim * dim);
		//printf("k              | average time(ms) | total time(ms)   | #cycles | last ratio       | last error norm\n");
		printf("k              | average time(ms) | total time(ms)   | #cycles | last ratio       | error norm reduction | residual norm reduction\n");
		printf("-----------------------------------------------------------\n");
		for (int k = FIRST_K; k <= last_k; k++)
		{
			Stat& stat = stats[k][nLevels];
			int nCyc = nCycles[k][nLevels];
			double averageCycleTime = stat.duration / stat.nCycles;

			//printf("%2d %s   | %16.6f | %16.6f | %7d | %16.6f | %16.6f\n", kk[k], methodNames[k], averageCycleTime[k][nLevels] * 1000.0, totalTime[k][nLevels] * 1000.0, nCyc, lastRatio[k][nLevels], stat.finishNorm);
			printf("%2d %s   | %16.6f | %16.6f | %7d | %16.6f | %16.6E | %16.6E\n", kk[k], methodNames[k], averageCycleTime * 1000.0, totalTime[k][nLevels] * 1000.0, nCyc, lastRatio[k][nLevels], stat.finishNorm / stat.startNorm, stat.finishResidualNorm / stat.startResidualNorm);
			//printf("%2d %s   | %16.6f | %16.6f | %d | %16.6f\n", kk[k], methodNames[k], averageCycleTime[k][nLevels] * 1000.0, totalTime[k][nLevels] * 1000.0, nCycles[k][nLevels], normTime[k][nLevels] * 1000.0);
			//printf("%2d %s   | %16.6f | %16.6f | %7d | %16.6f | %16.6f\n", kk[k], methodNames[k], averageCycleTime[k][nLevels] * 1000.0, totalTime[k][nLevels] * 1000.0, nCyc, stat.startNorm, stat.finishNorm);

		}
		printf("\n");

		printf("last error norm ratios:\n");
		printf("k              | last error ratio | prev error ratio | 3rd ratio\n");
		printf("-----------------------------------------------------------\n");
		for (int k = FIRST_K; k <= last_k; k++)
		{
			Stat& stat = stats[k][nLevels];
			int nCyc = nCycles[k][nLevels];

			printf("%2d %s   | %16.6f | %16.6f | %16.6f\n", kk[k], methodNames[k], stat.norms[nCyc] / stat.norms[nCyc - 1], stat.norms[nCyc - 1] / stat.norms[nCyc - 2], stat.norms[nCyc - 2] / stat.norms[nCyc - 3]);
		}
		printf("last residual norm ratios:\n");
		printf("k              | last residual ratio | prev residual ratio | 3rd ratio\n");
		printf("-----------------------------------------------------------\n");
		for (int k = FIRST_K; k <= last_k; k++)
		{
			Stat& stat = stats[k][nLevels];
			int nCyc = nCycles[k][nLevels];

			printf("%2d %s   | %16.6f | %16.6f | %16.6f\n", kk[k], methodNames[k], stat.residualNorms[nCyc] / stat.residualNorms[nCyc - 1], stat.residualNorms[nCyc - 1] / stat.residualNorms[nCyc - 2], stat.residualNorms[nCyc - 2] / stat.residualNorms[nCyc - 3]);
		}


		printf("\n");

	}
	printf("\n\n\n");
	for (int k = FIRST_K; k <= last_k; k++)
	{
		printf("average_k%d = [", kk[k]);
		for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
		{
			//nLevels = MAX_LEVELS;

			printf("%f", averageCycleTime[k][nLevels] * 1000.0);
			if (nLevels < MAX_LEVELS)
				printf(", ");
		}
		printf("];\n");
		printf("total_k%d = [", kk[k]);
		for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
		{
			printf("%f", totalTime[k][nLevels] * 1000.0);
			if (nLevels < MAX_LEVELS)
				printf(", ");
		}
		printf("];\n");
		printf("nCycles_k%d = [", kk[k]);
		for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
		{
			printf("%d", nCycles[k][nLevels]);
			if (nLevels < MAX_LEVELS)
				printf(", ");
		}
		printf("];\n");
	}
	printf("\n\n\n");


	// Print norms of first k and max levels (for ploting in matlab):
	Stat& stat = stats[FIRST_K][MAX_LEVELS];
	int nCyc = stat.nCycles;
	cout << "err_norms = ";
	PrintArray(stat.errorNorms, nCyc);
	cout << "res_norms = ";
	PrintArray(stat.residualNorms, nCyc);
	/* */
}

void TestKCycleSolveTimeAsFunctionOfEps(int nLevels)
{
	const int N_PRE_RELAXATIONS = 2;
	const int N_POST_RELAXATIONS = 2;
	const int nEps = sizeof(epsilons) / sizeof(epsilons[0]);
	Stat stats[NUM_K][nEps];
	const bool bPrintNorms = false;

	cout << "nLevels: " << nLevels << endl;
	cout << "N_PRE_RELAXATIONS: " << N_PRE_RELAXATIONS << endl;
	cout << "N_POST_RELAXATIONS: " << N_POST_RELAXATIONS << endl;
	PrintRelaxMethod();

	for (int iEps = 0; iEps < nEps; iEps++)
	{
		double epsilon = epsilons[iEps];
		cout << "epsilon=" << epsilon << endl;
		for (int k = FIRST_K; k <= last_k; k++)
		{
			Stat& stat = stats[k][iEps];
			cout << "Testing " << methodNames[k] << " with " << nLevels << " levels" << endl;
			SolveTime(nLevels, stat, kk[k], N_PRE_RELAXATIONS, N_POST_RELAXATIONS, epsilon, ratioWanted);
			if (stat.bTimeOut)
				cout << "timeout!" << endl;

			if (bPrintNorms)
			{
				int nCyc = stat.nCycles;
				cout << "err_norms = ";
				PrintArray(stat.errorNorms, nCyc);
				cout << "res_norms = ";
				PrintArray(stat.residualNorms, nCyc);
			}
		}
	}

	cout << "\n\n" << endl;
	cout << "eps = [";
	for (int iEps = 0; iEps < nEps; iEps++)
	{
		cout << epsilons[iEps];
		if (iEps < nEps - 1)
			cout << ", ";
	}
	cout << "];" << endl;
	for (int k = FIRST_K; k <= last_k; k++)
	{
		cout << "average_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			double averageCycleTime = stat.duration / stat.nCycles;
			cout << averageCycleTime * 1000.0;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		cout << "total_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			cout << stat.duration * 1000.0;
			if (stat.bTimeOut)
				cout << "!timeout!";
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		cout << "cu_total_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			if (stat.bTimeOut)
				cout << "!timeout!";
			else
				cout << stat.msRuntime;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		/*cout << "total_ex_nrm__k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			if (stat.bTimeOut)
				cout << "!timeout!";
			else
				cout << stat.msRuntime - stat.normCalcDuration * 1000.0;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;*/

		cout << "nCycles_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			cout << stat.nCycles;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		/*cout << "totalNormTime_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			double time = stat.normCalcDuration;
			cout << time * 1000.0;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;
		cout << "fraction of norm calc time" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			double frac = stat.normCalcDuration / stat.duration;
			cout << frac;
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;

		cout << "last_error_norm_ratios_k" << kk[k] << " = [";
		for (int iEps = 0; iEps < nEps; iEps++)
		{
			Stat& stat = stats[k][iEps];
			int n = stat.nCycles;
			cout << stat.errorNorms[n - 1] / stat.errorNorms[n - 2];
			if (iEps < nEps - 1)
				cout << ", ";
		}
		cout << "];" << endl;
		*/

	}
	cout << "\n\n" << endl;
}


// should be called with bAlwaysCalcUNrom=true and bAlwaysCalcResidualNrom=true
void CalcKCycleNorms()
{
	const int N_LEVELS = 7;
	Stat stats[NUM_K];
	const int v1 = 2, v2 = 2;

	for (int k = FIRST_K; k <= last_k; k++)
	{
		Stat& stat = stats[k];
		cout << "Testing " << methodNames[k] << " with " << N_LEVELS << " levels" << endl;
		cout << "K(" << v1 << "," << v2 << ")" << endl;
		SolveTime(N_LEVELS, stat, kk[k], v1, v2, epsilon, ratioWanted);

		cout << "initial norm: " << stat.startNorm << endl;
		cout << "final norm: " << stat.finishNorm << endl;
		cout << "number of cycles: " << stat.nCycles << endl;

		int nCyc = stat.nCycles;
		cout << "lastNorms_k" << kk[k] << " = [";
		for (int i = nCyc - 20; i <= nCyc; i++)
		{
			cout << scientific << stat.norms[i];
			if (i < nCyc)
				cout << ", ";
		}
		cout.unsetf(ios_base::floatfield);
		cout << "];" << endl;
		cout << "last norm ratio: " << stat.norms[nCyc] / stat.norms[nCyc - 1] << endl;
		cout << "lastResidualNorms_k" << kk[k] << " = [";
		for (int i = nCyc - 20; i <= nCyc; i++)
		{
			cout << scientific << stat.residualNorms[i];
			if (i < nCyc)
				cout << ", ";
		}
		cout.unsetf(ios_base::floatfield);
		cout << "];" << endl;
		cout << "\n" << endl;
	}
}

void CalcCGNorms(int nLevels)
{
	const int v1 = 4, v2 = 4;
	for (int k = 0; k < NUM_K; k++)
	{
		int n = 1 << nLevels;
		tfloat h = 1.0 / n;
		double w = calcSmoothing(relaxMethod, epsilon, angle, false, false);
		ConjugateGradient cg(v1, v2, epsilon, angle);
		cg.mg.relaxMethod = relaxMethod;
		cg.Build(n, h, w);
		cg.k = kk[k];

		PrepareProblem(cg.mg, n, epsilon, angle);
		tfloat startNorm = cg.mg.UNorm();

		cout << "Testing PCG" << methodNames[k] << " with " << nLevels << " levels" << endl;
		cout << "K(" << v1 << "," << v2 << ")" << endl;

		Timer t;

		t.StartMeasure();

		cg.RunWithPreconditioner(h, ratioWanted, false);

		t.StopMeasure();

		tfloat finishNorm = cg.mg.UNorm();

		int nIter = cg.nIteraions;
		cout << "completed with " << nIter << " iterations" << endl;
		cout << "lastCGErrorNorms_k" << kk[k] << " = [";
		for (int i = nIter - 21; i < nIter; i++)
		{
			cout << scientific << cg.errorNorms[i];
			if (i < nIter - 1)
				cout << ", ";
		}
		cout << "];" << endl;
		/*printf("lastCGResidualNorms_k%d = [", kk[k]);
		for (int i = nCyc - 20; i <= nCyc; i++)
		{
			printf("%e", stat.residualNorms[i]);
			if (i < nCyc)
				printf(", ");
		}
		printf("];\n");*/
		cout << "\n" << endl;
	}

}

void TestConjugateGradient(int nLevels, int k, int nu1, int nu2, Stat& stat, double ratioWanted)
{
	int n = 1 << nLevels;
	tfloat h = 1.0 / n;
	double w = calcSmoothing(relaxMethod, epsilon, angle, false, false);
	ConjugateGradient cg(nu1, nu2, epsilon, angle);
	cg.mg.relaxMethod = relaxMethod;
	cg.Build(n, h, w);
	cg.k = k;
	cudaError_t cudaStatus;

	if (cg.mg.finestLevel->bGpu)
	{
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
	}

	PrepareProblem(cg.mg, n, epsilon, angle);
	tfloat startNorm = cg.mg.UNorm();
	if (problemIndex > 0)
	{
		tfloat norm_of_F = cg.mg.FNorm(); // 0 for problem index 0
		double norm_of_residual = cg.mg.ResidualNorm();
		double rel_norm = norm_of_residual / norm_of_F;
		cout << "Initial relative norm: " << rel_norm << endl; // should be 1 when x0 is zero
	}

	Timer t;

	t.StartMeasure();

	//cg.mg.KCycle(1, h);
	//cg.Run(h);
	cg.RunWithPreconditioner(h, ratioWanted, bStopByResidualRatio);

	t.StopMeasure();

	tfloat finishNorm = cg.mg.UNorm();

	stat.duration = t.GetDuration();
	stat.startNorm = startNorm;
	stat.finishNorm = finishNorm;
	stat.nCycles = cg.nIteraions;
	for (int i = 0; i < cg.nIteraions; i++)
		stat.errorNorms[i] = cg.errorNorms[i];
}

void PrintStatLine(int k, const char* name, double duration, int nCycles, const char* epiloge = "")
{
	cout << setprecision(2);
	cout << fixed;
	cout << setw(2) << k << " " << name << "   | " << setw(16) << duration * 1000.0 << " | " << setw(7) << nCycles << epiloge << endl;
	cout << setprecision(6);
}

void PrintStatLine(int k, const char* name, const Stat& stat)
{
	//printf("%2d %s   | %16.6f | %16.6f | %7d | %16.6f | %16.6E | %16.6E\n", kk[k], methodNames[k], averageCycleTime * 1000.0, totalTime[k][nLevels] * 1000.0, nCyc, lastRatio[k][nLevels], stat.finishNorm / stat.startNorm, stat.finishResidualNorm / stat.startResidualNorm);

	cout << setprecision(2);
	cout << fixed;
	cout << setw(2) << k << " " << name << "   | " << setw(16) << stat.duration * 1000.0 << " | " << setw(7) << stat.nCycles << endl;
	cout << setprecision(6);
}

// Compare ConjugateGradient to standard k-cycle for various k's.
// Only the max level is compared
void CompareToConjugateGradient(int nu1, int nu2)
{
	//const int NUM_SOLVES = 4;
	const int NUM_SOLVES = 1;
	const bool skipCG = false;
	const bool skipStandAlone = false;

	Stat cgStats[NUM_SOLVES][NUM_K];
	Stat mgStats[NUM_SOLVES][NUM_K];
	int dim = (1 << nLevels) - 1;

	cout << "CompareToConjugateGradient: nu1=" << nu1 << ",nu2=" << nu2 << endl;
	cout << "#levels: " << nLevels << " , size: " << dim << " ^ 2 = " << dim *dim << endl;

	for (int k = FIRST_K; k <= last_k; k++)
	{
		for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)
		{
			Stat& cgStat = cgStats[iSolve][k];
			Stat& mgStat = mgStats[iSolve][k];
			cout << endl;
			if (!skipCG)
			{
				cout << "Testing conjugate gradient " << methodNames[k] << " with " << nLevels << " levels" << endl;
				TestConjugateGradient(nLevels, kk[k], nu1, nu2, cgStat, ratioWanted);
				cout << endl;
			}
			if (!skipStandAlone)
			{
				cout << "Testing multigird " << methodNames[k] << " with " << nLevels << " levels" << endl;
				SolveTime(nLevels, mgStat, kk[k], nu1, nu2, epsilon, ratioWanted);

				cout << "Initial error norm: " << mgStat.startErrorNorm << endl;
				cout << "Final error norm: " << mgStat.finishErrorNorm << endl;
			}
		}
	}

	//cout << "k              | total time(ms)   | #cycles | last ratio       | error norm reduction | residual norm reduction" << endl;
	cout << endl;
	if (!skipStandAlone)
	{
		cout << endl;
		cout << "Stand-alone multigrid results:" << endl;
		cout << "k              | total time(ms)   | #cycles " << endl;
		cout << "-----------------------------------------------------------" << endl;

		for (int k = FIRST_K; k <= last_k; k++)
		{
			double duration = 0.0;
			int nCycles = 0;
			bool bTimeOut = false;
			for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)
			{
				Stat& mgStat = mgStats[iSolve][k];
				duration += mgStat.duration;
				nCycles = mgStat.nCycles;
				if (mgStat.bTimeOut)
					bTimeOut = true;
			}
			duration /= NUM_SOLVES;
			PrintStatLine(kk[k], methodNames[k], duration, nCycles, bTimeOut ? " - timeout!" : "");
		}
	}

	if (!skipCG)
	{
		cout << endl;
		cout << "Conjugate gradient (MGCG) results:" << endl;
		cout << "k              | total time(ms)   | #cycles " << endl;
		cout << "-----------------------------------------------------------" << endl;

		for (int k = FIRST_K; k <= last_k; k++)
		{
			double duration = 0.0;
			int nCycles = 0;
			for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)
			{
				Stat& cgStat = cgStats[iSolve][k];
				duration += cgStat.duration;
				nCycles = cgStat.nCycles;
			}
			duration /= NUM_SOLVES;
			PrintStatLine(kk[k], "(cg)     ", duration, nCycles);
		}
	}

	if (!skipStandAlone)
	{
		cout << endl;
		cout << "Last ratios MG:" << endl;
		for (int k = FIRST_K; k <= last_k; k++)
		{
			int nCycles = 0;
			Stat& mgStat = mgStats[0][k];
			nCycles = mgStat.nCycles;
			double ratio = mgStat.errorNorms[nCycles - 1] / mgStat.errorNorms[nCycles - 2];
			if (bStopByResidualRatio)
			{
				cout << "Residual norm" << endl;
				ratio = mgStat.residualNorms[nCycles - 1] / mgStat.residualNorms[nCycles - 2];
			}
			cout << setw(2) << kk[k] << " " << "(MG)   " << "   | " << setw(16) << ratio << " | " << setw(7) << nCycles << endl;
		}
	}

	if (!skipCG)
	{
		cout << endl;
		cout << "Last ratios MGCG:" << endl;
		for (int k = FIRST_K; k <= last_k; k++)
		{
			int nCycles = 0;
			Stat& cgStat = cgStats[0][k];
			nCycles = cgStat.nCycles;
			double ratio = cgStat.errorNorms[nCycles - 1] / cgStat.errorNorms[nCycles - 2];
			cout << setw(2) << kk[k] << " " << "(CG)   " << "   | " << setw(16) << ratio << " | " << setw(7) << nCycles << endl;
		}
	}

	/*
	//if (bStopByResidualRatio)
	{
		cout << "mg_res_norms = ";
		PrintArray(mgStats[0][FIRST_K].residualNorms, mgStats[0][FIRST_K].nCycles);
	}
	//else
	{
		cout << setprecision(8);
		cout << "mg_err_norms = ";
		PrintArray(mgStats[0][FIRST_K].errorNorms, mgStats[0][FIRST_K].nCycles);
		cout << setprecision(6);
	}/* */
	//cout << "cg_err_norms = ";
	//PrintArray(cgStats[0][FIRST_K].errorNorms, cgStats[0][FIRST_K].nCycles);*/
}

void CompareAngle(int nu1, int nu2, bool bCG, double runtimes[], int a_nCycles[])
{
	//const int NUM_SOLVES = 4;
	const int NUM_SOLVES = 1;

	Stat mgStats[NUM_SOLVES][NUM_K];
	Stat cgStats[NUM_SOLVES][NUM_K];
	int dim = (1 << nLevels) - 1;

	cout << "CompareAngle: nu1=" << nu1 << ",nu2=" << nu2 << endl;
	cout << "nLevels=" << nLevels << endl;
	cout << "levels: " << nLevels << " , size: " << dim << " ^ 2 = " << dim *dim << endl;
	cout << "angleDeg: " << angleDeg << endl;

	for (int k = FIRST_K; k <= last_k; k++)
	{
		for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)
		{
			Stat& cgStat = cgStats[iSolve][k];
			Stat& mgStat = mgStats[iSolve][k];
			cout << endl;
			if (bCG)
			{
				cout << "Testing conjugate gradient " << methodNames[k] << " with " << nLevels << " levels" << endl;
				TestConjugateGradient(nLevels, kk[k], nu1, nu2, cgStat, ratioWanted);
				cout << endl;
			}
			else
			{
				cout << "Testing multigird " << methodNames[k] << " with " << nLevels << " levels" << endl;
				SolveTime(nLevels, mgStat, kk[k], nu1, nu2, epsilon, ratioWanted);
				cout << "Initial error norm: " << mgStat.startErrorNorm << endl;
				cout << "Final error norm: " << mgStat.finishErrorNorm << endl;
			}
		}
	}

	//cout << "k              | total time(ms)   | #cycles | last ratio       | error norm reduction | residual norm reduction" << endl;
	cout << endl;
	cout << endl;

	cout << "angleDeg: " << angleDeg << endl;

	if (!bCG)
	{
  		cout << "Stand-alone multigrid results:" << endl;
  		cout << "k              | total time(ms)   | #cycles " << endl;
  		cout << "-----------------------------------------------------------" << endl;

  		for (int k = FIRST_K; k <= last_k; k++)
  		{
  			double duration = 0.0;
  			int nCycles = 0;
  			bool bTimeOut = false;
  			for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)
  			{
  				Stat& mgStat = mgStats[iSolve][k];
  				duration += mgStat.duration;
  				nCycles = mgStat.nCycles;
  				if (mgStat.bTimeOut)
  					bTimeOut = true;
  			}
  			duration /= NUM_SOLVES;
  			PrintStatLine(kk[k], methodNames[k], duration, nCycles, bTimeOut ? " - timeout!" : "");
			runtimes[k] = duration;
			a_nCycles[k] = nCycles;
			if (bTimeOut)
				runtimes[k] = 0;
  		}
	}

	if (bCG)
	{
		cout << endl;
		cout << "Conjugate gradient (MGCG) results:" << endl;
		cout << "k              | total time(ms)   | #cycles " << endl;
		cout << "-----------------------------------------------------------" << endl;

		for (int k = FIRST_K; k <= last_k; k++)
		{
			double duration = 0.0;
			int nCycles = 0;
			for (int iSolve = 0; iSolve < NUM_SOLVES; iSolve++)
			{
				Stat& cgStat = cgStats[iSolve][k];
				duration += cgStat.duration;
				nCycles = cgStat.nCycles;
			}
			duration /= NUM_SOLVES;
			PrintStatLine(kk[k], "(cg)     ", duration, nCycles);
			runtimes[k] = duration;
			a_nCycles[k] = nCycles;
		}
	}

	if (!bCG)
	{
  		cout << endl;
  		cout << "Last ratios MG:" << endl;
  		for (int k = FIRST_K; k <= last_k; k++)
  		{
  			int nCycles = 0;
  			Stat& mgStat = mgStats[0][k];
  			nCycles = mgStat.nCycles;
  			double ratio = mgStat.errorNorms[nCycles - 1] / mgStat.errorNorms[nCycles - 2];
  			if (bStopByResidualRatio)
  			{
  				cout << "Residual norm" << endl;
  				ratio = mgStat.residualNorms[nCycles - 1] / mgStat.residualNorms[nCycles - 2];
  			}
  			cout << setw(2) << kk[k] << " " << "(MG)   " << "   | " << setw(16) << ratio << " | " << setw(7) << nCycles << endl;
  		}
	}


  /*
	//if (bStopByResidualRatio)
	{
		cout << "mg_res_norms = ";
		PrintArray(mgStats[0][FIRST_K].residualNorms, mgStats[0][FIRST_K].nCycles);
	}
	//else
	{
		cout << setprecision(8);
		cout << "mg_err_norms = ";
		PrintArray(mgStats[0][FIRST_K].errorNorms, mgStats[0][FIRST_K].nCycles);
		cout << setprecision(6);
	}/* */
	//cout << "cg_err_norms = ";
	//PrintArray(cgStats[0][FIRST_K].errorNorms, cgStats[0][FIRST_K].nCycles);*/
}

void CompareAngles(bool bCG = true)
{
  const double angles[] = {10, 30, 45, 60, 80};
  //const double angles[] = {80};
  const int nAng = sizeof(angles) / sizeof(angles[0]);
  //bCG = false;
  const bool bStandAlone = true;
  double runtimes[nAng][NUM_K];
  int nCycles[nAng][NUM_K];
  double runtimes_cg[nAng][NUM_K];
  int nCycles_cg[nAng][NUM_K];

  int nu1 = 2, nu2 = 2;

  cout << "CompareAngles: nu1=" << nu1 << ",nu2=" << nu2 << endl;

  for (int iAng = 0; iAng < nAng; iAng++)
  {
    angleDeg = angles[iAng];
    angle = angleDeg * (M_PI / 180);
    if (bCG)
		CompareAngle(nu1, nu2, bCG, runtimes_cg[iAng], nCycles_cg[iAng]);
	if (bStandAlone)
		CompareAngle(nu1, nu2, false, runtimes[iAng], nCycles[iAng]);
  }

  if (bStandAlone)
  {
	  cout << "Stand-alone multigrid results:" << endl;
	  cout << "k \\ angle     ";
	  for (int iAng = 0; iAng < nAng; iAng++)
		  cout << " | " << angles[iAng];
	  cout << endl;

	  for (int k = FIRST_K; k <= last_k; k++)
	  {
		  cout << kk[k] << " " << methodNames[k];
		  for (int iAng = 0; iAng < nAng; iAng++)
			  cout << " | " << runtimes[iAng][k] * 1000.0 << "(" << nCycles[iAng][k] << ")";
		  cout << endl;
	  }
  }

  if (bCG)
  {
    cout << "Conjugate gradient (MGCG) results:" << endl;
    cout << "k \\ angle     ";
    for (int iAng = 0; iAng < nAng; iAng++)
      cout << " | " << angles[iAng];
    cout << endl;
    for (int k = FIRST_K; k <= last_k; k++)
    {
      cout << kk[k] << " " << methodNames[k];
      for (int iAng = 0; iAng < nAng; iAng++)
        cout << " | " << runtimes_cg[iAng][k] * 1000.0 << "(" << nCycles_cg[iAng][k] << ")";
      cout << endl;
    }
  }

  if (bStandAlone)
  {
	  // matlab vars:
	  cout << "times = [";
	  for (int k = FIRST_K; k <= last_k; k++)
	  {
		  for (int iAng = 0; iAng < nAng; iAng++)
		  {
			  cout << runtimes[iAng][k] * 1000.0;
			  if (iAng < nAng - 1)
				  cout << ", ";
			  else if (k < last_k)
				  cout << ";" << endl;
			  else
				  cout << "];" << endl;
		  }
	  }
	  cout << endl;
	  cout << "cycles = [";
	  for (int k = FIRST_K; k <= last_k; k++)
	  {
		  for (int iAng = 0; iAng < nAng; iAng++)
		  {
			  cout << nCycles[iAng][k];
			  if (iAng < nAng - 1)
				  cout << ", ";
			  else if (k < last_k)
				  cout << ";" << endl;
			  else
				  cout << "];" << endl;
		  }
	  }
	  cout << endl;
  }
  if (bCG)
  {
    cout << "cg_times = [";
    for (int k = FIRST_K; k <= last_k; k++)
    {
      for (int iAng = 0; iAng < nAng; iAng++)
      {
        cout << runtimes_cg[iAng][k] * 1000.0;
        if (iAng < nAng - 1)
          cout << ", ";
        else if (k < last_k)
          cout << ";" << endl;
        else
          cout << "];" << endl;
      }
    }
    cout << endl;
    cout << "cg_cycles = [";
    for (int k = FIRST_K; k <= last_k; k++)
    {
      for (int iAng = 0; iAng < nAng; iAng++)
      {
        cout << nCycles_cg[iAng][k];
        if (iAng < nAng - 1)
          cout << ", ";
        else if (k < last_k)
          cout << ";" << endl;
        else
          cout << "];" << endl;
      }
    }
    cout << endl;
  }
}

#ifdef HAVE_CUDA

void CheckKernelLaunchTime(int n_blocks)
{
	const int N_THREADS = 1024;
	//const int N_THREADS = 32;
	//const int N_BLOCKS = 64; // 16 Multiprocessors on CGGC computer, 10 on home computer, max 2048 threads per Multiprocessor
	const int CALLS = 1000000;
	cudaStream_t stream = nullptr;

	int size = n_blocks * N_THREADS;

	CUDA_ERROR(cudaStreamCreate(&stream));
	int access_policy = UM_ACCESS_GPU;
	tfloat* p = (tfloat*)um_malloc(sizeof(tfloat)*size, access_policy);
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);
	for (int i = 0; i < size; i++)
	{
		p[i] = i;
	}
	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);
	Timer timer;
	timer.StartMeasure();
	for (int i = 0; i < CALLS; i++)
	{
		CudaInc(p, n_blocks, N_THREADS, nullptr);
		//CudaInc(p, n_blocks, N_THREADS, stream);
		CudaNone(p, n_blocks, N_THREADS, nullptr);
	}

	cudaStatus = cudaDeviceSynchronize();
	CUDA_ERROR(cudaStatus);

	timer.StopMeasure();

	//cout << "threads: " << size << endl;
	//cout << "Duration: " << timer.GetDuration() * 1000.0 / CALLS << " ms" << endl;
	// Duration: 0.00437379 ms - 1 block - CGGC computer
	// Duration: 0.00203135 ms - 1 block - home computer

	cout << timer.GetDuration() * 1000.0 / CALLS / 2 << ", ";

	um_free(p, access_policy);
	CUDA_ERROR(cudaStreamDestroy(stream));

}

void CheckKernelLaunchTime()
{
	cout << "nThreads = [";
	for (int nBlocks = 1; nBlocks <= 32; nBlocks *= 2)
	//for (int nBlocks = 32; nBlocks < 1000; nBlocks += 32)
	//for (int nBlocks = 1024; nBlocks < 4000; nBlocks += 512)
	{
		int size = nBlocks * 1024;
		cout << size << ", ";
	}
	cout << "];" << endl;
	cout << "durations = [";
	for (int nBlocks = 1; nBlocks <= 32; nBlocks *= 2)
		CheckKernelLaunchTime(nBlocks);
	//for (int nBlocks = 32; nBlocks < 1000; nBlocks += 32)
	//for (int nBlocks = 1024; nBlocks < 4000; nBlocks += 512)
		//CheckKernelLaunchTime(nBlocks);
	cout << "];" << endl;
	//CheckKernelLaunchTime(1);
}
#endif

void PrintParamters()
{
	cout << "bUseGpu: " << bUseGpu << endl;
	cout << "epsilon: " << epsilon << endl;
	cout << "angle: " << angle << endl;
	cout << "angleDeg: " << angleDeg << "(degrees)" << endl;
	cout << "ratioWanted: " << ratioWanted << endl;
	cout << "rtol: " << rtol << endl;
	cout << "bStopByResidualRatio: " << bStopByResidualRatio << endl;
	cout << "problemIndex: " << problemIndex << endl;
	cout << "MAX_LEVELS: " << MAX_LEVELS << endl;
	cout << "nLevels: " << nLevels << endl;
	cout << "bGalerkin: " << bGalerkin << endl;

	size_t free, total;
	//cudaMemGetInfo(&free, &total);
	//cout << "Memory: " << free << " free, " << total << " total" << endl;
}

// This function is used to measure the time for matrix operations in order to optimize these operations based on the memory bandwidth needed for them
void TestRelaxationDurations()
{
	double averageDurations[30];
	double bandwidths[30];
	double rates[30];
	//for (int nLevels = MAX_LEVELS; nLevels >= FIRST_LEVEL; nLevels--)
	{
		int n = 1 << nLevels;
		cout << "Testing relaxation with " << n << "x" << n << " unknowns (=\"" << nLevels << " levels\")" << endl;

		cudaError_t cudaStatus;
		MG2D cyc(relaxMethod, 1, 1, 1.0, 0.0);

		int n_iterations = 10000;
		tfloat h = 1.0 / n;

		cout << "Building levels" << endl;
		cyc.w = 2.0 / 3.0;
		bool bOK = cyc.Build(n, h);

		if (!bOK)
		{
			cout << "MG2D::Build failed" << endl;
			return;
		}

		cout << "Making a random initial guess" << endl;
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
		//cyc.RandomU();
		//cudaStatus = cudaDeviceSynchronize();
		//CUDA_ERROR(cudaStatus);

		PrepareProblem(cyc, n, 1.0, 0.0);

		double norm_of_F = cyc.FNorm(); // 0 for problem index 0
		double norm_of_residual = cyc.ResidualNorm();;
		double rel_norm = norm_of_residual / norm_of_F;

		cout << "Initial u norm: " << cyc.UNorm() << endl;
		cout << "Initial error norm: " << cyc.ErrorNorm() << endl;
		cout << "norm_of_F: " << norm_of_F << endl;
		cout << "norm_of_residual: " << norm_of_residual << endl;
		cout << "rel_norm: " << rel_norm << endl;

		//Timer timer;
		Timer timerFull;
		Timer synchTimer;
		CudaTimer timer;
		CudaTimer internalTimer;

		// replace managed memory by GPU only memory:
		size_t count = cyc.finestLevel->u.count;

		/*
		um_free(cyc.finestLevel->f.p, UM_ACCESS_GPU);
		um_free(cyc.finestLevel->u.p, UM_ACCESS_GPU);
		um_free(cyc.finestLevel->temp.p, UM_ACCESS_GPU);
		cudaStatus = cudaMalloc(&cyc.finestLevel->f.p, sizeof(tfloat)*count);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaMalloc(&cyc.finestLevel->u.p, sizeof(tfloat)*count);
		CUDA_ERROR(cudaStatus);
		cudaStatus = cudaMalloc(&cyc.finestLevel->temp.p, sizeof(tfloat)*count);
		CUDA_ERROR(cudaStatus);
		cudaMemset(cyc.finestLevel->f.p, 0, sizeof(tfloat)*count);
		cudaMemset(cyc.finestLevel->u.p, 0, sizeof(tfloat)*count);
		cudaMemset(cyc.finestLevel->temp.p, 0, sizeof(tfloat)*count);
		/* */

		cout << "Starting full timer" << endl;
		timerFull.StartMeasure();
		// first pass is a warm up, in the second pass we measure times
		for (int doTiming = 0; doTiming <= 1; doTiming++)
		{
			if (doTiming)
				timer.StartMeasure();
			//#pragma omp parallel // bad performance if here
			for (int iter = 1; iter <= n_iterations; iter++)
			{
				//if (doTiming)
					//internalTimer.StartMeasure();
				cyc.Relax(cyc.finestLevel);
				//double norm = cyc.UNorm();
				//if (doTiming)
					//internalTimer.StopMeasure();

				//cyc.Residual(cyc.finestLevel, cyc.finestLevel->u, cyc.finestLevel->f, cyc.finestLevel->temp);
				//cyc.Restrict(cyc.finestLevel, cyc.finestLevel->temp, cyc.finestLevel->courserLevel->f);
				//cyc.Prolong(cyc.finestLevel, cyc.finestLevel->courserLevel->u, cyc.finestLevel->temp);
				//cyc.finestLevel->u.Add(cyc.finestLevel->temp); // bandwidth of 154 with 10 levels (80%)

				//cyc.finestLevel->u.MakeZero();
				//cyc.finestLevel->courserLevel->u.MakeZero();
				//cudaMemset(cyc.finestLevel->u.p, 0, cyc.finestLevel->u.count * sizeof(tfloat));
				//cudaMemset(cyc.finestLevel->u.p, 0, 1 * sizeof(tfloat) / 8);

				// norms:
				//double norm = cyc.UNorm();
				//double errorNorm = cyc.ErrorNorm();
				//double residualNorm = cyc.ResidualNorm();
				//cout << residualNorm << endl;
				if (bRelaxSolve)
					break; // for single block relax tests
			}

			/*synchTimer.StartMeasure();
			cudaStatus = cudaDeviceSynchronize();
			CUDA_ERROR(cudaStatus);
			synchTimer.StopMeasure();*/

			if (doTiming)
				timer.StopMeasure();
		}

		/*synchTimer.StartMeasure();
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
		synchTimer.StopMeasure();*/

		timerFull.StopMeasure();

		averageDurations[nLevels] = timer.GetDuration() / n_iterations;

		cout << "Finished test with " << n << "x" << n << " unknonws. " << timer.GetDuration() << " seconds measured time, " << timerFull.GetDuration() << " seconds bruto" << endl;
		//cout << internalTimer.GetDuration() << " seconds internal timer" << endl;
		//cout << "synch time: " << synchTimer.GetDuration() << endl;
		norm_of_F = cyc.FNorm(); // 0 for problem index 0
		norm_of_residual = cyc.ResidualNorm();;
		rel_norm = norm_of_residual / norm_of_F;
		cout << "Final u norm: " << cyc.UNorm() << endl;
		cout << "Final error norm: " << cyc.ErrorNorm() << endl;
		cout << "norm_of_F: " << norm_of_F << endl;
		cout << "norm_of_residual: " << norm_of_residual << endl;
		cout << "rel_norm: " << rel_norm << endl;

		double size = cyc.finestLevel->u.count * sizeof(tfloat) / 1000000.0; // in MB
		double duration = averageDurations[nLevels] * 1000.0; // in ms
		double bandwidth = size * 3 / duration; // in GB /s - for Jacobi
		//double bandwidth = size * 4 / duration; // in GB /s - for current (3/4/2021) xzebra
		//double restrictBandwidth = size * 1.25 / duration; // in GB /s - for restrict and prolong
		//double addBandwidth = size * 3 / duration; // in GB /s - for add (for one addition we  need 1 read from dest + 1 read from source + 1 write to dest = 3 memory accesses
		//double zero1Bandwidth = size / duration; // in GB /s - for MakeZero - on the current level
		//double zeroBandwidth = size * 0.25 / duration; // in GB /s - for MakeZero - on a coarser level
		double normBandwidth = size / duration; // in GB /s - for UNorm - on the current level

		// must be changed according to the function we measure!:
		bandwidths[nLevels] = bandwidth;
		//rates[nLevels] = cyc.finestLevel->u.count / (averageDurations[nLevels] * 1000000000.0);
		//rates[nLevels] = (31*31) / (averageDurations[nLevels] * 1000000000.0);
		rates[nLevels] = (15 * 15) / (averageDurations[nLevels] * 1000000000.0); // for matrix of 15x15. need to multiply by about 10 (operation per element in relaxation). only one core of 10 (in my computer) is used.
		// approximated max possible rate expecteed: 68.4/10/10 = 0.68

		//cout << "Relax actual bandwidth: " << size << "*3/" << duration << " = " << bandwidth << " GB / s" << endl;

		//cout << "Residual actual bandwidth: " << size << "*3/" << duration << " = " << bandwidth << " GB / s" << endl;
		//cout << "Restrict actual bandwidth: " << size << "*1.25/" << duration << " = " << restrictBandwidth << " GB / s" << endl;
		//cout << "Prolong actual bandwidth: " << size << "*1.25/" << duration << " = " << restrictBandwidth << " GB / s" << endl;
		//cout << "Add actual bandwidth: " << size << "*3/" << duration << " = " << addBandwidth << " GB / s" << endl;
		//cout << "Make zero actual bandwidth: " << size << "/" << duration << " = " << zero1Bandwidth << " GB / s" << endl;
		//cout << "Make zero actual bandwidth: " << size << "*0.25/" << duration << " = " << zeroBandwidth << " GB / s" << endl;
	}

	cout << endl << endl << endl;
	cout << "=== relaxation results ===" << endl;
	cout << "levels | average duration(ms) | bandwidth (GB/s) | rate (count/runtime, G/s) " << endl;
	cout << "-----------------------------" << endl;
	cout << left;
	//for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
	{
		cout << setw(6) << nLevels << " | " << setw(20) << averageDurations[nLevels] * 1000.0 << " | " << setw(16) << bandwidths[nLevels] << " | " << rates[nLevels] << endl;
	}
	cout << endl << endl << endl;
	cout << "durations = [";
	//for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
	{
		cout << averageDurations[nLevels] * 1000.0;
		if (nLevels < MAX_LEVELS)
			cout << ",";
	}
	cout << "];" << endl;

	cout << "bandwidths = [";
	//for (int nLevels = FIRST_LEVEL; nLevels <= MAX_LEVELS; nLevels++)
	{
		cout << bandwidths[nLevels];
		if (nLevels < MAX_LEVELS)
			cout << ",";
	}
	cout << "];" << endl;
}

int MathTests()
{
	cout << (13 / 3) << endl; //4
	cout << ((-13) / 3) << endl; //-4
	cout << (int)(-13.0 / 3.0) << endl; //-4
	cout << floor(-13.0 / 3.0) << endl; //-5

	return 0;
}

// len is in ints, so amount of memory is 4*len bytes
double CacheTests(unsigned int len)
{
	int* arr = new int[len];
	double duration = 0.0;

	memset(arr, 0, sizeof(int)*len);

	Timer timer;
	timer.StartMeasure();

	// Loop 1 - 23.3039 ms on CGGGC-EE2015-10
	//for (int i = 0; i < len; i++) arr[i] *= 3;

	// Loop 2 - 24.9007 ms on CGGGC-EE2015-10
	//for (int i = 0; i < len; i += 16) arr[i] *= 3;


	int steps = 64 * 1024 * 1024; // Arbitrary number of steps
	int lengthMod = len - 1;
	for (int i = 0; i < steps; i++)
	{
		arr[(i * 16) & lengthMod]++; // (x & lengthMod) is equal to (x % arr.Length)
	}

	timer.StopMeasure();

	delete[] arr;

	duration = timer.GetDuration();

	return duration;
}

//http://igoro.com/archive/gallery-of-processor-cache-effects/
void CacheTests()
{
	//const int len = 64 * 1024 * 1024;
	//double duration = CacheTests(len);
	//cout << "The loop took " << duration * 1000.0 << " ms" << endl;

	unsigned int maxLen = 128 * 1024 * 1024;
	cout << "durations = [";
	// len is in ints, so amount of memory is 4*len bytes
	for (unsigned int len = 1024; len <= maxLen; len *= 2)
	{
		double duration = CacheTests(len);
		cout << duration;
		if (len < maxLen)
			cout << ", ";
	}
	cout << "];" << endl;

	//durations = [0.0373936, 0.0366916, 0.0361253, 0.0365357, 0.11901, 0.119842, 0.119668, 0.163188, 0.16501, 0.165394, 0.175581, 0.193966, 0.36384, 0.39506, 0.395278, 0.393031, 0.389482, 0.389406, 0.388632, 0.389192, 0.391612];
}


double GPUCacheTests(unsigned int height, unsigned int width)
{
	double duration = 0.0;
	Matrix2D m;

	int stride = width;
	if (!m.AllocateMemory(width, height, stride, true))
	{
		cout << "Error allocating matrix memory!" << endl;
		return 0.0;
	}

	Timer timer;
	timer.StartMeasure();

	int steps = 64 * 1024; // Arbitrary number of steps
	for (int i = 0; i < steps; i++)
		m.Multiply(1.0);

	timer.StopMeasure();


	duration = timer.GetDuration();

	return duration;
}

void GPUCacheTests()
{
	const int width = 512;
	const int rowBytes = width * sizeof(tfloat);


	//for (int height = 32; height <= 1024; height *= 2)
	for (int height = 32; height <= 1024; height *= 2)
	{
		int matrixBytes = height * rowBytes;
		//cout << "width: " << width << " tfloats" << endl;
		//cout << "height: " << height << endl;
		cout << "matrix size: " << matrixBytes << " bytes" << endl;
		double duration = GPUCacheTests(height, width);
		cout << "The loop took " << duration * 1000.0 << " ms" << endl;
	}


	/*unsigned int maxLen = 1024 * 1024 * 1024;
	cout << "durations = [";
	for (unsigned int len = 1024; len <= maxLen; len *= 2)
	{
		double duration = CacheTests(len);
		cout << duration;
		if (len < maxLen)
			cout << ", ";
	}
	cout << "];" << endl;*/
}

#ifdef HAVE_CUDA
void MyQuery()
{
	int dev = 0;

	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	cout << "\nDevice " << dev << " \"" << deviceProp.name << "\"" << endl;

	cout << "concurrentManagedAccess: " << deviceProp.concurrentManagedAccess << endl;
	cout << "concurrentKernels: " << deviceProp.concurrentKernels << endl;

	int value;
	cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem, dev);
	cout << "cudaDevAttrCanUseHostPointerForRegisteredMem: " << value << endl; // 0 on my home computer ???
	cudaDeviceGetAttribute(&value, cudaDevAttrCanMapHostMemory, dev);
	cout << "cudaDevAttrCanMapHostMemory: " << value << endl; // 1 on my home computer
	cout << "unifiedAddressing " << deviceProp.unifiedAddressing << endl; // 1 on my home computer
	cout << "canMapHostMemory " << deviceProp.canMapHostMemory << endl; // 1 on my home computer
	cout << "canUseHostPointerForRegisteredMem " << deviceProp.canUseHostPointerForRegisteredMem << endl; // 0 on my home computer ???

}
#endif

// linux only:
#ifdef LINUX
void change_stack_size()
{
	// change stack size on linux:
	const rlim_t kStackSize = 16 * 1024 * 1024;   // min stack size = 16 MB
	struct rlimit rl;
	int result;

	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize)
		{
			rl.rlim_cur = kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
				}
			}
	}
}
#endif

#ifdef HAVE_CUDA
void TestCyclicReduction()
{
	tfloat a = 1.0;
	tfloat b = 1.0;
	tfloat c = 1.0;
	tfloat f[] = { 3, 6, 9, 12, 15, 18, 13 };
	tfloat u[9] = { 0, 3, 6, 9, 12, 15, 18, 13, 0 };

	/*
	tfloat f[] = { 3, 6, 5 };
	tfloat u[5] = { 0, 3, 6, 5, 0 };

	CyclicReduction(a, b, c, f, u, 3);
	IterativeCyclicReduction(a, b, c, f, u, 3);

	cout << "u = [";
	for (int i = 0; i < 5; i++)
		cout << u[i] << " "; // should be "0 1 2 3 0"
	cout << "]" << endl;
	*/

	//CyclicReduction(a, b, c, f, u, 7);
	IterativeCyclicReductionInPlace(u, a, b, c, 7);

	cout << "u = [";
	for (int i = 0; i < 9; i++)
		cout << u[i] << " "; // should be "0 1 2 3 4 5 6 7 0"
	cout << "]" << endl;

}

void TestCyclicReduction2()
{
	tfloat a = -4.0;
	tfloat b = 1.0;
	tfloat c = 1.0;
	tfloat f[511] = { 0 };
	tfloat u[513] = { 0 };
	CyclicReduction(a, b, c, f, u, 511);
	cout << "u = [";
	for (int i = 0; i < 5; i++)
		cout << u[i] << " ";
	cout << "]" << endl;

}
#endif

void SmallChecks()
{
	int prec = cout.precision(); // default: 6
	cout << prec << endl;
}

void PrintUsage(char * progamName)
{
	/*
	cout << "Syntax:" << endl;
	//cout << progamName << " [test_name] args" << endl;
	cout << "k_cycle args" << endl;
	//cout << "test_name"
	cout << "-max_levels/-n_levels - number of multigrid levels to run";
	*/

	cout << "usgae: k_cycle [-max_levels n] [-n_levels n] [-last_k k] [-semi-coarsening 0/1]\n";
	cout << "               [-test test_name][-relax-method relax_name][-problem-index 0/1/2][-gpu][-Galerkin]\n\n";
	cout << "options (the first one is the default):\n";
	cout << "-test test_name                    test_name = usage/TestKCycleTime/TestKCycleSolveTimeAsFunctionOfEps/CompareToConjugateGradient/CompareAnglesNoCG\n";
	cout << "-relax-method relax_name           relax_name = OptimalPointJacobi/XZebra\n";
}

void ParseArgs(int argc, char *argv[])
{
	//if (argc == 1)
		//PrintUsage(argv[0]);
	for (int i = 1; i < argc; i++)
	{
		if (i + 1 < argc)
		{
			if (strcmp(argv[i], "-max_levels") == 0)
				nLevels = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-n_levels") == 0)
				nLevels = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-last_k") == 0)
				last_k = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-semi-coarsening") == 0)
				bSemiCoarsening = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-test") == 0)
				testName = argv[i + 1];
			if (strcmp(argv[i], "-relax-method") == 0)
			{
				if (strcmp(argv[i + 1], "XZebra") == 0)
					relaxMethod = XZebra;
			}
			if (strcmp(argv[i], "-problem-index") == 0)
				problemIndex = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], "-gpu") == 0)
		{
#ifdef HAVE_CUDA
			bUseGpu = true;
#else
			cout << "-gpu option have been given, but the program was not compiled with cuda support" << endl;
			exit(1);
#endif // HAVE_CUDA
		}
		if (strcmp(argv[i], "-Galerkin") == 0)
			bGalerkin = true;
	}
}

cublasHandle_t cublasHandle = 0;
//cusparseHandle_t cusparseHandle = 0;
void* pBuffer;

#ifdef HAVE_CUDA
void AllocateBuffer()
{
	int nLevels = 14;
	int n = 1 << nLevels;
	size_t bufferSizeInBytes;
	cusparseStatus_t status;
	status = cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle, n, n / 2, NULL, NULL, NULL, NULL, n, &bufferSizeInBytes);
	if (status == CUSPARSE_STATUS_SUCCESS)
	{
		cout << "Buffer size: " << bufferSizeInBytes << endl;
		//2684846080 for nLevels = 14, with n right-hand sides
		//1342668800 for nLevels = 14, with n/2 right-hand sides

		// 34619904 for nLevels=11 with my in-place code
	}
	else
		cout << "Error in CusparseTests, status=" << status << endl;

	//Buffer size: 2,684,846,080
	pBuffer = um_malloc(bufferSizeInBytes, UM_ACCESS_GPU);

}

/*void CusparseTests()
{
	cusparseStatus_t status;
	tfloat a = 1.0;
	tfloat b = 1.0;
	tfloat c = 1.0;
	//tfloat f[] = { 3, 6, 5 };
	//tfloat u[5];
	//tfloat d[3] = {a, a, a};
	//tfloat dl[3] = {0, b, b};
	//tfloat du[3] = {c, c, 0};
	tfloat *u, *d, *dl, *du;
	u = (tfloat *)um_malloc(3*sizeof(tfloat), UM_ACCESS_GPU);
	d = (tfloat*)um_malloc(3 * sizeof(tfloat), UM_ACCESS_GPU);
	dl = (tfloat*)um_malloc(3 * sizeof(tfloat), UM_ACCESS_GPU);
	du = (tfloat*)um_malloc(3 * sizeof(tfloat), UM_ACCESS_GPU);

	AllocateBuffer();
	u[0] = 3;
	u[1] = 6;
	u[2] = 5;
	d[0] = a;
	d[1] = a;
	d[2] = a;
	dl[0] = 0;
	dl[1] = b;
	dl[2] = b;
	du[0] = c;
	du[1] = c;
	du[2] = 0;

	status = cusparseDgtsv2_nopivot(cusparseHandle, 3, 1, dl, d, du, u, 3, pBuffer);
	if (CUSPARSE_STATUS_SUCCESS == status)
		cout << "cusparseDgtsv2_nopivot success" << endl;
	else
		cout << "cusparseDgtsv2_nopivot error: " << status << endl;

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();

	cout << "u = [";
	for (int i = 0; i < 3; i++)
		cout << u[i] << " "; // should be "1 2 3"
	cout << "]" << endl;

	um_free(u, UM_ACCESS_GPU);
	um_free(d, UM_ACCESS_GPU);
	um_free(dl, UM_ACCESS_GPU);
	um_free(du, UM_ACCESS_GPU);
	//um_free(pBuffer, UM_ACCESS_GPU);
}*/
#endif

int main(int argc, char *argv[])
{
	// for windows: stack reverse size: 100000000 in Linker-System (project properties)
#ifdef LINUX
	change_stack_size(); // for linux
#endif

	ParseArgs(argc, argv);

	Timer overAllTimer;
	overAllTimer.StartMeasure();

	//cudaDeviceReset();

	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
#ifdef HAVE_CUDA
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);
#endif

	if (relaxMethod == XZebra && !bSemiCoarsening)
	{
		bSemiCoarsening = true;
		cout << "bSemiCoarsening is automatically set to true because of using an x-zebra relaxation" << endl;
	}

	PrintParamters();
	PrintRelaxMethod();

	double w = calcSmoothing(relaxMethod, epsilon, angle, false, false);
	cout << "w: " << w << endl;

	try
	{
  		//printInfo();
  		//deviceQuery();
  		//MyQuery();
  		//CacheTests();
  		//GPUCacheTests();
  		//CusparseTests();
  		//AllocateBuffer();

  		//TestKCycleSolveTime(); // old
  		//CalcKCycleNorms();
  		//CalcCGNorms(12);
  		//CheckKernelLaunchTime();
  		//TestRelaxationDurations();
  		//TestFixedNumberOfIterations(11);

		if (strcmp(testName, "") == 0 || strcmp(testName, "usage") == 0)
			PrintUsage(argv[0]);
		else if (strcmp(testName, "TestKCycleTime") == 0)
			TestKCycleTime(4, nLevels);
		else if (strcmp(testName, "TestKCycleSolveTimeAsFunctionOfEps") == 0)
			TestKCycleSolveTimeAsFunctionOfEps(nLevels);
		else if (strcmp(testName, "CompareToConjugateGradient") == 0)
		{
			//for (int nu = 2; nu <= 2; nu++)
				//CompareToConjugateGradient(nu, nu);
			int nu = 2;
			CompareToConjugateGradient(nu, nu);
		}
  		//test_gmres(5);
		if (strcmp(testName, "CompareAnglesNoCG") == 0)
			CompareAngles();

  		//TestCyclicReduction();
	}
	catch (const char* ex)
	{
	    cout << "Exception: " << ex << endl;
	}

	if (cublasHandle != nullptr)
		cublasDestroy(cublasHandle);
#ifdef HAVE_CUDA
	if (cusparseHandle)
		cusparseDestroy(cusparseHandle);
#endif

	overAllTimer.StopMeasure();
	cout << "overall time: " << overAllTimer.GetDuration() << endl;

// no need to wait for the user on linux, and this would disable the automation
// (in case we want to run more than one test)
#if defined(WIN32)
	cout << "Press <Enter> to quit" << endl;
	getchar();
#endif

	return 0;
}
