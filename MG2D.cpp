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

#ifdef HAVE_CUDA
bool CudaApplyStencill(tfloat* dest, const tfloat* x, const tfloat* stencil, int n, int stride);
bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, int level);
bool CudaRestrict(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest);
bool CudaProlong(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest, int nSrc);
bool CudaSmoothJacobiWithOneBlock(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, tfloat h2);
bool CudaSmoothJacobiWithOneBlock(tfloat* temp, const tfloat* rhs, tfloat* x, const tfloat* stencil, int n, int stride, tfloat w, int nIterations);
bool CudaSmoothJacobiWithOneBlock6Times(tfloat* temp, const tfloat* rhs, tfloat* x, const tfloat* stencil, int n, int stride, tfloat w);
bool CudaRedBlackGSKernel(tfloat* dest, const tfloat* rhs, const tfloat* stencil, tfloat w, int n, int stride);
bool CudaRedBlackKernel(tfloat* dest, const tfloat* rhs, tfloat* temp, const tfloat* stencil, tfloat w, int n, int stride);
bool CudaRedBlackBorder(tfloat* dest, const tfloat* rhs, tfloat* temp, const tfloat* stencil, tfloat w, int n, int stride);
bool CudaJacobi(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, tfloat w, int level);
bool CudaSolve(tfloat* dest, const tfloat* rhs, int stride, tfloat w);
bool MemCopyJacobiStencil(const tfloat* stencil);
bool MemCopyResidualStencil(const tfloat* stencil, int iLevel);
bool MemCopyJacobiStencil(const tfloat* stencil, int iLevel);
bool CreateCyclicReductionValues(tfloat* stencil, int iLevel);
bool CudaYZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level, tfloat* tmp);

// rectangle cuda functions (nx may be different than ny)
bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int nx, int ny, int stride, int level);
bool CudaJacobi(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int nx, int ny, int stride, tfloat w, int level);
bool CudaJacobiSingleBlock(tfloat* x, const tfloat* rhs, const tfloat* stencil, int nx, int ny, int stride, tfloat w, int level, int n_times);
bool CudaRestrictY(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx); // semi-coarsening
bool CudaProlongY(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx); // semi-coarsening
bool CudaXZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level, tfloat* tmp1, tfloat* tmp2, int tmp_stride);
bool CudaXZebraWithCusparse(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level, tfloat* buffer, int tmp_stride);
bool CudaInPlaceXZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level);

#else
bool CudaApplyStencill(tfloat* dest, const tfloat* x, const tfloat* stencil, int n, int stride) {return false;}
bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, int level) {return false;}
bool CudaRestrict(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest) {return false;}
bool CudaProlong(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nDest, int nSrc) {return false;}
bool CudaSmoothJacobiWithOneBlock(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, tfloat h2) {return false;}
bool CudaSmoothJacobiWithOneBlock(tfloat* temp, const tfloat* rhs, tfloat* x, const tfloat* stencil, int n, int stride, tfloat w, int nIterations) {return false;}
bool CudaSmoothJacobiWithOneBlock6Times(tfloat* temp, const tfloat* rhs, tfloat* x, const tfloat* stencil, int n, int stride, tfloat w) {return false;}
bool CudaRedBlackGSKernel(tfloat* dest, const tfloat* rhs, const tfloat* stencil, tfloat w, int n, int stride) {return false;}
bool CudaRedBlackKernel(tfloat* dest, const tfloat* rhs, tfloat* temp, const tfloat* stencil, tfloat w, int n, int stride) {return false;}
bool CudaRedBlackBorder(tfloat* dest, const tfloat* rhs, tfloat* temp, const tfloat* stencil, tfloat w, int n, int stride) {return false;}
bool CudaJacobi(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, tfloat w, int level) {return false;}
bool CudaSolve(tfloat* dest, const tfloat* rhs, int stride, tfloat w) {return false;}
bool MemCopyJacobiStencil(const tfloat* stencil) {return false;}
bool MemCopyResidualStencil(const tfloat* stencil, int iLevel) {return false;}
bool MemCopyJacobiStencil(const tfloat* stencil, int iLevel) {return false;}
bool CreateCyclicReductionValues(tfloat* stencil, int iLevel) {return false;}

// rectangle cuda functions (nx may be different than ny)
bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int nx, int ny, int stride, int level) {return false;}
bool CudaJacobi(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int nx, int ny, int stride, tfloat w, int level) {return false;}
bool CudaJacobiSingleBlock(tfloat* x, const tfloat* rhs, const tfloat* stencil, int nx, int ny, int stride, tfloat w, int level, int n_times) { return false; }
bool CudaRestrictY(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx) {return false;}
bool CudaProlongY(tfloat* __restrict__ dest, const tfloat* __restrict__ src, int dStride, int sStride, int nyDest, int nx) {return false;}
bool CudaXZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level)  {return false;}
bool CudaInPlaceXZebra(tfloat* x, const tfloat* rhs, int nx, int ny, int stride, int level) { return false; }

#endif
void CheckCudaError(cudaError_t cudaStatus, const char *const file, int const line);

extern bool bUseGpu;
extern bool bGalerkin;
extern bool bSemiCoarsening;
extern bool bSPAI0;
extern bool bHybrid;
extern bool bRelaxSolve;

void GalerkinIteration(tfloat out_op[3][3], const tfloat in_op[3][3]);
void GalerkinIterationY(tfloat out_op[3][3], const tfloat in_op[3][3]);
void ZeroStencil(tfloat s[3][3]);
void MultiplyStencil(tfloat s[3][3], tfloat a);
void AddStencil(tfloat a[3][3], tfloat b[3][3]);
void CyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n);
void IterativeCyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n);
int CalcStride(int n);

// vars for xzebra
tfloat* d;
tfloat* dl;
tfloat* du;

void MG2D::CalcStencil(tfloat result[3][3], tfloat finerStencil[3][3], tfloat hx, tfloat hy)
{
	// Lu = Uss+ep*Utt
	if (finerStencil && bGalerkin)
	{
		if (bSemiCoarsening)
			GalerkinIterationY(result, finerStencil); // generate coarser level stencil using Galerkin
		else
			GalerkinIteration(result, finerStencil); // generate coarser level stencil using Galerkin
	}
	else
	{
		const tfloat C = cos(ang);
		const tfloat S = sin(ang);
		const tfloat C2 = C * C;
		const tfloat S2 = S * S;

		tfloat Uxx[3][3] = { 0.0, 0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0 };
		tfloat Uyy[3][3] = { 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0 };
		tfloat Uxy[3][3] = { -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0 };
		MultiplyStencil(Uxy, 0.25);

		MultiplyStencil(Uxx, (C2 + ep * S2) / (hx * hx));
		MultiplyStencil(Uyy, (ep * C2 + S2) / (hy * hy));
		MultiplyStencil(Uxy, 2*(1-ep)*C*S / (hx * hy));

		ZeroStencil(result);

		AddStencil(result, Uxx);
		AddStencil(result, Uyy);
		AddStencil(result, Uxy);
	}
}

bool MG2D::Build(int n, tfloat h)
{
	bool bGpu = bUseGpu;
	tfloat stencil[3][3];
	bool bPrintSize = false;

	tfloat hx = h, hy = h;
	int nx = n, ny = n;

	CalcStencil(stencil, nullptr, hx, hy);

	/*
	// print the stencils for debugging:
	cout << "finest_level_stencil:" << endl;
	PrintStencil(stencil);
	GalerkinIteration(stencil, finest_level_stencil);
	cout << "next_level_stencil:" << endl;
	PrintStencil(stencil);
	for (int i = 0; i < 10; i++)
		GalerkinIteration(stencil, stencil);
	cout << "next_level_stencil:" << endl;
	PrintStencil(stencil);
	*/

	int iLevelIndex = 0;

	delete finestLevel;
	finestLevel = new Level;
	Level* pl = finestLevel;
	if (bHybrid && n <= 32) // last 5 levels on the CPU
		bGpu = false;
	int access_policy = bGpu ? UM_ACCESS_GPU : UM_ACCESS_CPU;

	/*
	// temporarily here - SPAI-0 for galerkin
	tfloat w = 0.0;
	//if (bGalerkin)
	{
		tfloat sum2 = 0.0;
		for (int i = 0; i < 9; i++)
		{
			tfloat val = stencil[i / 3][i % 3];
			current_stencil[i] = val;
			sum2 += val * val;
		}

		// SPAI-0
		w = stencil[1][1] * stencil[1][1] / sum2;
	}
	*/

	if (bPrintSize)
	{
		int stride = CalcStride(nx+1);
		size_t size = stride * (ny+1);
		cout << n << " levels => " << sizeof(tfloat) * size << " bytes per array" << endl;
		return false;
	}

	if (!pl->BuildLevel(nx+1, ny+1, bGpu, stencil, w, iLevelIndex++, access_policy))
	{
		cout << "Finest level building failed!" << endl;
		return false;
	}
	while (ny > 2) // n=2 is the last level, with 1 unknown and a direct solver
	{
		if (bRelaxSolve && ny <= 16)
			break;
		ny /= 2;
		hy *= 2.0;
		if (!bSemiCoarsening)
		{
			nx /= 2;
			hx *= 2.0;
		}
		pl->courserLevel = new Level;
		if (pl->courserLevel == nullptr)
			return false;

		//if (bHybrid && n <= 16) // last 4 levels on the CPU
		if (bHybrid && n <= 32) // last 5 levels on the CPU
		//if (bHybrid && n <= 64) // last 6 levels on the CPU
			bGpu = false;
		int access_policy = bGpu ? UM_ACCESS_GPU : UM_ACCESS_CPU;
		if (pl->bGpu && !bGpu) // here pl refers to the next finer level
			access_policy = UM_ACCESS_BOTH;
		pl = pl->courserLevel;
		//if (n <= 2)
			//bGpu = false; // run last level (Solve only) on the CPU

		CalcStencil(stencil, stencil, hx, hy);

		tfloat ww = w;
		// SPAI-0
		if (bSPAI0)
		{
			tfloat sum2 = 0.0;
			for (int i = 0; i < 9; i++)
			{
				tfloat val = stencil[i / 3][i % 3];
				sum2 += val * val;
			}

			ww = stencil[1][1]*stencil[1][1] / sum2;
		}

		if (bRelaxSolve && ny <= 32)
			w = 1.0; // undamped jacobi for solving coarsest level
		if (!pl->BuildLevel(nx+1, ny+1, bGpu, stencil, ww, iLevelIndex++, access_policy))
			return false;
	}
	for (pl = finestLevel; pl != nullptr; pl = pl->courserLevel)
	{
		if (pl->bGpu)
		{
			MemCopyResidualStencil(pl->residualStencil, pl->iLevelIndex);
			MemCopyJacobiStencil(pl->jacobiStencil, pl->iLevelIndex);
			CreateCyclicReductionValues(pl->residualStencil, pl->iLevelIndex);
		}
		if (!pl->courserLevel)
			pl->bLastLevel = true;
	}
	if (finestLevel->bGpu)
	{
		MemCopyJacobiStencil(finestLevel->jacobiStencil);
		//int stride = CalcStride(2*n + 1);
		//if (!tmp1.AllocateMemory(2*n + 1, n + 1, stride, true))
			//return false;
		//if (!tmp2.AllocateMemory(2*n + 1, n + 1, stride, true))
			//return false;

		/*
		tfloat a = finestLevel->residualStencil[4];
		tfloat b = finestLevel->residualStencil[3];
		tfloat c = finestLevel->residualStencil[5];

		stride = CalcStride(n + 1);

		d = (tfloat*)um_malloc(stride * sizeof(tfloat), UM_ACCESS_GPU);
		dl = (tfloat*)um_malloc(stride * sizeof(tfloat), UM_ACCESS_GPU);
		du = (tfloat*)um_malloc(stride * sizeof(tfloat), UM_ACCESS_GPU);

		for (int i = 0; i < stride; i++)
		{
			d[i] = a;
			dl[i] = b;
			du[i] = c;
		}
		dl[0] = 0.0;
		du[n-1-1] = 0.0; //last element
		*/
	}

	return true;
}

bool MG2D::Destroy()
{
	delete finestLevel;
	finestLevel = nullptr;
	return true;
}

void MG2D::KCycle(int k, tfloat h)
{
	KCycle(k, h, finestLevel);
}

void MG2D::KCycle(int k, tfloat h, Level* pLevel, int iLevel)
{
	if (pLevel == nullptr)
		return;

	nCalls++;

	if (pLevel->courserLevel == nullptr)
	{
		Solve(pLevel, pLevel->u, pLevel->f);
		//PrintResidualNorm(pLevel);
		return;
	}

	Timer timer;
	timer.StartMeasure();

	Relax(pLevel, nu1);

	//cout << "Level " << iLevel  << ". ";
	//PrintResidualNorm(pLevel);

	if (pLevel->courserLevel != nullptr)
	{
		Residual(pLevel, pLevel->u, pLevel->f, pLevel->temp);
		Restrict(pLevel, pLevel->temp, pLevel->courserLevel->f);

		//pLevel->courserLevel->f.MakeZero(); // for debugging
		//pLevel->courserLevel->f.Multiply(1.6);

		pLevel->courserLevel->u.MakeZero();

		if (pLevel->bGpu && !pLevel->courserLevel->bGpu)
		{
			cudaError_t cudaStatus = cudaDeviceSynchronize();
			CUDA_ERROR(cudaStatus);
		}

		// Recursive call/s
		if (k > 0)
			KCycle(k, 2 * h, pLevel->courserLevel, iLevel + 1);
		if (k > 1 && !pLevel->courserLevel->bLastLevel)
		//if (k > 1)
			KCycle(k - 1, 2 * h, pLevel->courserLevel, iLevel + 1);

		//cout << "Level " << iLevel  << ". ";
		//PrintResidualNorm(pLevel);

		Prolong(pLevel, pLevel->courserLevel->u, pLevel->temp);

		//pLevel->temp.MakeZero(); //for debugging
		//cudaError_t cudaStatus = cudaDeviceSynchronize();
		//CUDA_ERROR(cudaStatus);
		//pLevel->temp.ZeroBorder(); //for debugging

		pLevel->u.Add(pLevel->temp);
	}

	//cout << "Level " << iLevel  << ". ";
	//PrintResidualNorm(pLevel);

	Relax(pLevel, nu2);

	timer.StopMeasure();

	// important: this calculation is not valid when using the GPU:
	if (countDurationForLevel == iLevel && !pLevel->bGpu)
	{
		durationCounter += timer.GetDuration();
	}
}

// For Ax=b, this function computes r = b-Ax
// When A is SPD, solving Ax=b is identical to minimizing [f(x)=(1/2)*xT*A*x-bT*x]. Then -r=Ax-b is the gradient of f.
void MG2D::Residual(Level* pLevel, const Matrix2D& u, const Matrix2D& f, Matrix2D& r)
{
	cudaError_t cudaStatus;

	if (u.bGpu)
	{
		tfloat* stencil = pLevel->residualStencil;

		tfloat* dest = r.p;
		const tfloat* rhs = f.p;
		const tfloat* x = u.p;
		CudaResidual(dest, rhs, x, stencil, r.nx, r.ny, r.stride, pLevel->iLevelIndex);
	}
	else
	{
		tfloat* __restrict__ stencil = pLevel->residualStencil;

		tfloat* __restrict__ dest = r.p;
		const tfloat* __restrict__ rhs = f.p;
		const tfloat* __restrict__ x = u.p;

		int stride = r.stride;

		for (int j = 1; j < r.ny - 1; j++)
		{
			for (int i = 1, idx = j*stride+1; i < r.nx - 1; i++, idx++)
			{
				dest[idx] = rhs[idx] -
					(     stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
						+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
						+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
			}
		}
	}
}



void MG2D::Relax(Level* pLevel, int nu)
{
	Matrix2D& u = pLevel->u;
	const Matrix2D& f = pLevel->f;
	if (relaxMethod == XYZebra || relaxMethod == YXZebra)
		nu /= 2;
	for (int i = 0; i < nu; i++)
		Relax(pLevel);
}


void MG2D::Relax(Level* pLevel)
{
	if (relaxMethod == OptimalPointJacobi)
		Jacobi(pLevel, pLevel->u, pLevel->f);
	else if (relaxMethod == PointRB)
		RedBlackGS(pLevel, pLevel->u, pLevel->f);
	else if (relaxMethod == XZebra)
		XZebraRelax(pLevel, pLevel->u, pLevel->f);
	else if (relaxMethod == YZebra)
		YZebraRelax(pLevel, pLevel->u, pLevel->f);
	else if (relaxMethod == XYZebra)
		XYZebraRelax(pLevel, pLevel->u, pLevel->f);
	else if (relaxMethod == YXZebra)
		YXZebraRelax(pLevel, pLevel->u, pLevel->f);
}

// u is both input and output
void MG2D::Jacobi(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	if (u.bGpu)
	{
		tfloat* stencil = pLevel->jacobiStencil;

		//pLevel->temp.MakeZero();
		int nx = pLevel->temp.nx;
		int ny = pLevel->temp.ny;
		int stride = pLevel->temp.stride;
		tfloat* dest = pLevel->temp.p;
		const tfloat* rhs = f.p;
		const tfloat* x = u.p;

		if (bGalerkin)
		{
			CudaJacobi(dest, rhs, x, stencil, nx, ny, stride, pLevel->w / pLevel->stencil[4], pLevel->iLevelIndex);
			
			// note: when using swap, we must take care to copy the boundary conditions!!!
			u.Swap(pLevel->temp); // OK only when the boundary conditions are 0
		}
		else
		{
			CudaJacobi(dest, rhs, x, stencil, nx, ny, stride, pLevel->w / pLevel->stencil[4], 0);
			// nx=ny only! for bandwidth tests:
			//CudaJacobi(dest, rhs, x, stencil, nx, stride, pLevel->w, 0);

			// note: when using swap, we must take care to copy the boundary conditions!!!
			u.Swap(pLevel->temp); // OK only when the boundary conditions are 0
		}


		/*pLevel->temp.CoptyTo(u);
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
		pLevel->temp.ZeroBorder();*/
	}
	else
	{
		//Residual(pLevel, u, f, pLevel->temp);
		//u.Add(pLevel->temp, w*h2 / diag);

		tfloat* __restrict__ stencil = pLevel->jacobiStencil;

		tfloat* __restrict__ dest = pLevel->temp.p;
		const tfloat* __restrict__ rhs = f.p;
		const tfloat* __restrict__ x = u.p;

		const tfloat ww = pLevel->w / pLevel->stencil[4];

		const int stride = u.stride;

//#pragma omp parallel for schedule(static)
//#pragma omp parallel for schedule(static) if (u.n > 33) // u.n == 33 means 5 levels
//#pragma omp for schedule(static) // bad performance if here
		for (int j = 1; j < u.ny - 1; j++)
		{
			for (int i = 1, idx = j*stride + 1; i < u.nx - 1; i++, idx++)
			{
				dest[idx] = ww * rhs[idx] -
					(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
					+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
					+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);
			}
		}
//#pragma omp single // bad performance if here
		//u.Swap(pLevel->temp);

		// note: when using swap, we must take care to copy the boundary conditions!!!
		u.Swap(pLevel->temp); // OK only when the boundary conditions are 0
	}
}

// u is both input and output
void MG2D::XZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	const int MAX_N = 16 * 1024 + 2;
	tfloat ff[MAX_N];

	if (u.bGpu)
	{
		int nx = u.nx;
		int ny = u.ny;
		int stride = u.stride;
		const tfloat* rhs = f.p;
		tfloat* x = u.p;

		//cout << nx - 2 << endl;
		//CudaXZebra(x, rhs, nx, ny, stride, pLevel->iLevelIndex, tmp1.p, tmp2.p, tmp1.stride);
		//CudaXZebraWithCusparse(x, rhs, nx, ny, stride, pLevel->iLevelIndex, NULL, 0);
		CudaInPlaceXZebra(x, rhs, nx, ny, stride, pLevel->iLevelIndex);
		//pLevel->temp.MakeZero(); // only for now, need a better solution!
	}
	else
	{
		const tfloat* __restrict__ rhs = f.p;
		tfloat* __restrict__ x = u.p;

		const int stride = u.stride;
		tfloat a = pLevel->stencil[4];
		tfloat b = pLevel->stencil[3];
		tfloat c = pLevel->stencil[5];

		//#pragma omp parallel
		for (int j = 1; j < u.ny - 1; j += 2)
		{
			int idx = j * stride + 1;

			//#pragma omp for schedule(static)
			for (int i = 1; i < u.nx-1; i++, idx++)
			{
				// r is the sum of weighted values for the row before and the row after i
				tfloat r = 0.0;
				r += pLevel->stencil[0] * x[idx - stride - 1];
				r += pLevel->stencil[1] * x[idx - stride];
				r += pLevel->stencil[2] * x[idx - stride + 1];
				r += pLevel->stencil[6] * x[idx + stride - 1];
				r += pLevel->stencil[7] * x[idx + stride];
				r += pLevel->stencil[8] * x[idx + stride + 1];
				ff[i] = rhs[idx] - r;
			}

			//CyclicReduction(a, b, c, ff+1, x + j * stride, u.nx - 2);
			IterativeCyclicReduction(a, b, c, ff+1, x + j * stride, u.nx - 2);
		}

		//#pragma omp parallel
		for (int j = 2; j < u.ny - 1; j += 2)
		{
			int idx = j * stride + 1;

			//#pragma omp for schedule(static)
			for (int i = 1; i < u.nx-1; i++, idx++)
			{
				// r is the sum of weighted values for the row before and the row after i
				tfloat r = 0.0;
				r += pLevel->residualStencil[0] * x[idx - stride - 1];
				r += pLevel->residualStencil[1] * x[idx - stride];
				r += pLevel->residualStencil[2] * x[idx - stride + 1];
				r += pLevel->residualStencil[6] * x[idx + stride - 1];
				r += pLevel->residualStencil[7] * x[idx + stride];
				r += pLevel->residualStencil[8] * x[idx + stride + 1];
				ff[i] = rhs[idx] - r;
			}

			//CyclicReduction(a, b, c, ff+1, x + j * stride, u.nx - 2);
			IterativeCyclicReduction(a, b, c, ff + 1, x + j * stride, u.nx - 2);
		}
	}
}

// u is both input and output
void MG2D::YZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	const int MAX_N = 16 * 1024 + 2;
	tfloat ff[MAX_N];
	tfloat uu[MAX_N];

	if (u.bGpu)
	{
		/*
		int nx = u.nx;
		int ny = u.ny;
		int stride = u.stride;
		const tfloat* rhs = f.p;
		tfloat* x = u.p;

		CudaYZebra(x, rhs, nx, ny, stride, pLevel->iLevelIndex, pLevel->temp.p);
		*/
		Matrix2D& t = pLevel->temp;
		u.Transpose(t);
		f.Transpose(u);
		int nx = t.nx;
		int ny = t.ny;
		int stride = t.stride;
		const tfloat* rhs = u.p;
		tfloat* x = t.p;
		CudaInPlaceXZebra(x, rhs, nx, ny, stride, MAX_LEVELS+pLevel->iLevelIndex);
		t.Transpose(u);
	}
	else
	{
		const tfloat* __restrict__ rhs = f.p;
		tfloat* __restrict__ x = u.p;

		const int stride = u.stride;
		tfloat a = pLevel->stencil[4];
		tfloat b = pLevel->stencil[1];
		tfloat c = pLevel->stencil[7];

		for (int ix = 1; ix < u.nx - 1; ix += 2)
		{
			for (int iy = 1; iy < u.ny-1; iy++)
			{
				int idx = iy * stride + ix;
				// r is the sum of weighted values for the row before and the row after i
				tfloat r = 0.0;
				r += pLevel->stencil[0] * x[idx - stride - 1];
				r += pLevel->stencil[2] * x[idx - stride + 1];
				r += pLevel->stencil[3] * x[idx - 1];
				r += pLevel->stencil[5] * x[idx + 1];
				r += pLevel->stencil[6] * x[idx + stride - 1];
				r += pLevel->stencil[8] * x[idx + stride + 1];
				ff[iy] = rhs[idx] - r;
			}

			IterativeCyclicReduction(a, b, c, ff+1, uu, u.ny - 2);
			for (int iy = 1; iy < u.ny-1; iy++)
			{
				int idx = iy * stride + ix;
				x[idx] = uu[iy];
			}
		}

		for (int ix = 2; ix < u.nx - 1; ix += 2)
		{
			for (int iy = 1; iy < u.ny-1; iy++)
			{
				int idx = iy * stride + ix;
				// r is the sum of weighted values for the row before and the row after i
				tfloat r = 0.0;
				r += pLevel->stencil[0] * x[idx - stride - 1];
				r += pLevel->stencil[2] * x[idx - stride + 1];
				r += pLevel->stencil[3] * x[idx - 1];
				r += pLevel->stencil[5] * x[idx + 1];
				r += pLevel->stencil[6] * x[idx + stride - 1];
				r += pLevel->stencil[8] * x[idx + stride + 1];
				ff[iy] = rhs[idx] - r;
			}

			IterativeCyclicReduction(a, b, c, ff+1, uu, u.ny - 2);
			for (int iy = 1; iy < u.ny-1; iy++)
			{
				int idx = iy * stride + ix;
				x[idx] = uu[iy];
			}
		}
	}
}

// u is both input and output
void MG2D::XYZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	XZebraRelax(pLevel, u, f);
	YZebraRelax(pLevel, u, f);
}

// u is both input and output
void MG2D::YXZebraRelax(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	YZebraRelax(pLevel, u, f);
	XZebraRelax(pLevel, u, f);
}

void HalfRedBlack(tfloat* dest, const tfloat* x, const tfloat* rhs, const tfloat* stencil, tfloat w, int s, int stride, int n)
{
	for (int j = 1; j < n - 1; j++)
	{
		for (int i = 1, idx = j*stride + 1; i < n - 1; i++, idx++)
		{
			if ((i + j + s) % 2 == 0)
				dest[idx] = x[idx];
			else
			{
				tfloat Ax =
					(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
						+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
						+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);

				dest[idx] = x[idx] + w * (rhs[idx] - Ax);
			}
		}
	}
}

void HalfRedBlack(tfloat* x, const tfloat* rhs, const tfloat* stencil, tfloat w, int s, int stride, int n)
{
	for (int j = 1; j < n - 1; j++)
	{
		for (int i = 1, idx = j*stride + 1; i < n - 1; i++, idx++)
		{
			if ((i + j + s) % 2 == 0)
				continue;

			tfloat Ax =
				(stencil[0] * x[idx - stride - 1] + stencil[1] * x[idx - stride] + stencil[2] * x[idx - stride + 1]
					+ stencil[3] * x[idx - 1] + stencil[4] * x[idx] + stencil[5] * x[idx + 1]
					+ stencil[6] * x[idx + stride - 1] + stencil[7] * x[idx + stride] + stencil[8] * x[idx + stride + 1]);

			x[idx] += w * (rhs[idx] - Ax);
		}
	}
}

// u is both input and output
void MG2D::RedBlackGS(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	if (u.bGpu)
	{
	}
	else
	{
	}
}

// Only valid when just 1 unknown/row remains (unless bRelaxSolve is true)
void MG2D::Solve(Level* pLevel, Matrix2D& u, const Matrix2D& f)
{
	// The Jacobi with weight of 1 seems to be a problem since the stencil is already weighted.
	// This still works becasue u is initiated to zero in the finer level. This doesn't works if we have only one level.
	// When Jacobi is called with 1 unknown, weight of w=1, and u=0 
	// it is equivalent to just setting u = (w*h2 / diag) * rhs = rhs / (diag/h2)
	// w*h2 / diag
	//Jacobi(pLevel, u, f, 1.0);

	if (bSemiCoarsening)
	{
		// This call will give the exact solution when just 1 row remains:
		XZebraRelax(pLevel, u, f);
		return;
	}


	int stride = pLevel->u.stride;
	tfloat* dest = pLevel->u.p;
	const tfloat* rhs = f.p;

	if (u.bGpu)
	{
		if (bRelaxSolve)
		{
			//Relax(pLevel, 100);

			tfloat* stencil = pLevel->jacobiStencil;
			CudaJacobiSingleBlock(u.p, f.p, stencil, u.nx, u.ny, u.stride, pLevel->w / pLevel->stencil[4], pLevel->iLevelIndex, 100);
		}
		else
			CudaSolve(dest, rhs, stride, 1.0 / pLevel->stencil[4]);
	}
	else
	{
		dest[stride + 1] = rhs[stride + 1] / pLevel->stencil[4];
	}
}

void MG2D::Restrict(Level* pLevel, const Matrix2D& src, Matrix2D& dest)
{
	cudaError_t cudaStatus;

	bool bGpu = pLevel->bGpu;

	if (bGpu)
	{
		tfloat* pDest = dest.p;
		tfloat* pSrc = src.p;
		if (bSemiCoarsening)
			CudaRestrictY(pDest, pSrc, dest.stride, src.stride, dest.ny, dest.nx);
		else
			CudaRestrict(pDest, pSrc, dest.stride, src.stride, dest.ny);
	}
	else
	{
		int ny = dest.ny;
		int nx = dest.nx;
		int dStride = dest.stride;
		int sStride = src.stride;

		if (bSemiCoarsening)
		{
			for (int i = 1; i < ny - 1; i++)
			{
				for (int j = 1; j < nx - 1; j++)
				{
					dest.p[i * dStride + j] = 0.5 * (
						src.p[2 * i * sStride + j] +
						0.5 * (src.p[2 * i * sStride - sStride + j] + src.p[2 * i * sStride + sStride + j])
						);
				}
			}
		}
		else
		{
			for (int i = 1; i < ny - 1; i++)
			{
				for (int j = 1; j < nx - 1; j++)
				{
					dest.p[j + i * dStride] = 0.25 * (
						src.p[2 * j + 2 * i * sStride] +
						0.5 * (src.p[2 * j - 1 + 2 * i * sStride] + src.p[2 * j + 1 + 2 * i * sStride] + src.p[2 * j + 2 * i * sStride - sStride] + src.p[2 * j + 2 * i * sStride + sStride]) +
						0.25 * (src.p[2 * j - 1 + 2 * i * sStride - sStride] + src.p[2 * j - 1 + 2 * i * sStride + sStride] + src.p[2 * j + 1 + 2 * i * sStride - sStride] + src.p[2 * j + 1 + 2 * i * sStride + sStride])
						);
				}
			}
		}
	}
}

void MG2D::Prolong(Level* pLevel, const Matrix2D& src, Matrix2D& dest)
{
	cudaError_t cudaStatus;

	if (pLevel->bGpu)
	{
		tfloat* pDest = dest.p;
		tfloat* pSrc = src.p;
		if (bSemiCoarsening)
			CudaProlongY(pDest, pSrc, dest.stride, src.stride, dest.ny, dest.nx);
		else
			CudaProlong(pDest, pSrc, dest.stride, src.stride, dest.ny, src.ny);
	}
	else
	{
		tfloat* pDest = dest.p;
		const tfloat* pSrc = src.p;
		int nDestx = dest.nx;
		int nDesty = dest.ny;
		int nSrcx = dest.nx;
		int nSrcy = dest.ny;
		int dStride = dest.stride;
		int sStride = src.stride;

		/*
		bi-linear interpolation, valid only when the boundary is always zero
		example with
		nSrc = 2 ^ 8 + 1 = 257
		nDest = 2 ^ 9 + 1 = 513
		i = 255 (max i)
		2i = 510 (max 2i)

		dest[0] = src[0] = 0
		dest[1] = 0.5(src[0] + src[1])
		dest[510] = src[255]
		dest[511] = 0.5(src[255] + src[256])

		dest[512] and src[256] must always be 0. The method doesn't change dest[512]
		*/

		if (bSemiCoarsening)
		{
			for (int j = 0; j < nSrcy - 1; j++)
			{
				for (int i = 0; i < nSrcx - 1; i++)
				{
					int didx = i + 2 * j * dStride;
					int sidx = i + j * sStride;

					if (i >= nDestx - 1 || 2 * j >= nDesty - 1)
						continue;

					pDest[didx] = pSrc[sidx];
					pDest[didx + dStride] = 0.5 * (pSrc[sidx] + pSrc[sidx + sStride]);
				}
			}
		}
		else
		{
			for (int j = 0; j < nSrcy - 1; j++)
			{
				for (int i = 0; i < nSrcx - 1; i++)
				{
					int didx = 2 * i + 2 * j * dStride;
					int sidx = i + j * sStride;

					if (2 * i >= nDestx - 1 || 2 * j >= nDesty - 1)
						continue;

					pDest[didx] = pSrc[sidx];
					//if (2 * i + 1 < nDest - 1) // this whould always be true anyway
					pDest[didx + 1] = 0.5 * (pSrc[sidx] + pSrc[sidx + 1]);
					//if (2 * j + 1 < nDest - 1) // this whould always be true anyway
					pDest[didx + dStride] = 0.5 * (pSrc[sidx] + pSrc[sidx + sStride]);
					//if (2 * i + 1 < nDest - 1 && 2 * j + 1 < nDest - 1) // this whould always be true anyway
					pDest[didx + dStride + 1] = 0.25 * (pSrc[sidx] + pSrc[sidx + 1] + pSrc[sidx + sStride] + pSrc[sidx + sStride + 1]);
				}
			}
		}
	}
}

void MG2D::RandomU()
{
	finestLevel->u.MakeRandom();
	finestLevel->u.ZeroBorder();
}

double MG2D::UNorm()
{
	double result = finestLevel->u.Norm();
	//result /= (finestLevel->f.n - 1); // normalize the norm (only valid for 2D with nx=ny)
	result /= sqrt((finestLevel->f.nx - 1) * (finestLevel->f.ny - 1)); // normalize the norm
	return result;
}

double MG2D::FNorm()
{
	double result = finestLevel->f.Norm();
	result /= sqrt((finestLevel->f.nx - 1) * (finestLevel->f.ny - 1)); // normalize the norm
	return result;
}


double MG2D::ResidualNorm()
{
	Level* pLevel = finestLevel;
	Residual(pLevel, pLevel->u, pLevel->f, pLevel->temp);
	double result = pLevel->temp.Norm();
	result /= sqrt((pLevel->f.nx - 1) * (pLevel->f.ny - 1)); // normalize residual norm
	return result;
}

double MG2D::ErrorNorm()
{
	// Empty solution vector means the solution is the zero vector, in this case the error norm is the u norm:
	if (solution.count == 0)
		return UNorm();

	Level* pLevel = finestLevel;

	pLevel->u.CoptyTo(pLevel->temp);
	if (solution.count > 0)
		pLevel->temp.Add(solution, -1);

	double result = pLevel->temp.Norm();
	pLevel->temp.MakeZero(); // make sure the boundary is zero. TODO: this is an expensive call, and it is probably not needed because u=solution on the boundary
	result /= sqrt((pLevel->f.nx - 1) * (pLevel->f.ny - 1)); // normalize the norm
	return result;
}


void MG2D::PrintResidualNorm(Level* pLevel)
{
	Residual(pLevel, pLevel->u, pLevel->f, pLevel->temp);

	if (!pLevel->u.bGpu)
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaDeviceSynchronize();
		CUDA_ERROR(cudaStatus);
	}

	double norm = pLevel->temp.Norm();
	//norm /= pLevel->n; // normalize. This is valid only for the case n=nx=ny. we need to multiply by sqrt(h^d), where d is the number of dimensions. in this case d=2, and h=1/n. page 55 in multigrid tutorial book.
	norm /= sqrt(pLevel->temp.nx * pLevel->temp.ny); // normalize. we need to multiply by sqrt(hx*hy)
	cout << "residual norm: " << norm << endl;
}
