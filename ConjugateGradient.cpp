#include "Classes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <memory.h>
#include <iostream>
#include <iomanip>

bool CudaApplyStencill(tfloat* dest, const tfloat* x, const tfloat* stencil, int n, int stride);
bool CudaResidual(tfloat* dest, const tfloat* rhs, const tfloat* x, const tfloat* stencil, int n, int stride, int level);
int CalcStride(int n);

bool ConjugateGradient::Build(int n, tfloat h, tfloat w)
{

	mg.w = w;
	mg.Build(n, h);

	return true;
}

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
/*
function [x] = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
*/

void ApplyStencil(Matrix2D &Ap, const Matrix2D &x, Level* pLevel)
{
	tfloat* stencil = pLevel->residualStencil;

	if (Ap.bGpu)
		CudaApplyStencill(Ap.p, x.p, stencil, x.ny, x.stride);
	else
	{
		int nx = Ap.nx;
		int ny = Ap.ny;
		int stride = Ap.stride;
		for (int j = 1; j < ny - 1; j++)
		{
			int idx = 1 + j * stride;
			for (int i = 1; i < nx - 1; i++, idx++)
			{
				Ap.p[idx] = (stencil[0] * x.p[idx - stride - 1] + stencil[1] * x.p[idx - stride] + stencil[2] * x.p[idx - stride + 1]
					+ stencil[3] * x.p[idx - 1] + stencil[4] * x.p[idx] + stencil[5] * x.p[idx + 1]
					+ stencil[6] * x.p[idx + stride - 1] + stencil[7] * x.p[idx + stride] + stencil[8] * x.p[idx + stride + 1]);
			}
		}
	}

}

void ConjugateGradient::Run(tfloat h)
{
	int MAX_ITERATIONS = 5;
	Level* pLevel = mg.finestLevel;
	bool bGpu = pLevel->bGpu;
	int access_policy = bGpu ? UM_ACCESS_GPU : UM_ACCESS_CPU;

	Matrix2D& x = pLevel->u;
	Matrix2D r; // residual. r  = b-A*x. In each step, this will hold the (opposite of the) gradient.
	Matrix2D p; // = r (at init). In each step, this will be the direction we go.
	Matrix2D Ap; // Ap = A*p

	int stride = CalcStride(x.nx);

	r.AllocateMemory(x.nx, x.ny, stride, bGpu);
	p.AllocateMemory(x.nx, x.ny, stride, bGpu);
	Ap.AllocateMemory(x.nx, x.ny, stride, bGpu);

	mg.Residual(pLevel, x, pLevel->f, r); // r is the residual. -r is the gradient.
	r.CoptyTo(p);
	//rsold = r'*r
	tfloat rsold = r.Norm();
	cout << "Initial residual norm: " << rsold << endl;
	rsold = rsold * rsold;

	for (int iIter = 0; iIter < MAX_ITERATIONS; iIter++)
	{
		// Ap = A*p
		ApplyStencil(Ap, p, pLevel);
		tfloat alpha = rsold / p.DotProduct(Ap); // = rsold / (p' * Ap). Calc the best coefficient in -p direction.
		x.Add(p, alpha); // x = x + alpha * p;
		r.Add(Ap, -alpha); // r = r - alpha * Ap; Update residual/gradient for current iteration.
		tfloat rsnew = r.Norm(); // rsnew = r' * r; here it is after applying sqrt() to it
		cout << "(" << (iIter + 1) << ") Residual norm: " << rsnew << endl;
		if (rsnew < 1e-10)
		{
			rsold = rsnew * rsnew;
			break;
		}
		rsnew = rsnew * rsnew;
		//p = r + (rsnew / rsold) * p;
		p.Multiply(rsnew / rsold);
		p.Add(r);
		rsold = rsnew;
	}
	cout << "Final residual norm: " << sqrt(rsold) << endl; // here rsold is the squared norm
}

// if bStopByResidualRatio is true the method stops when the residual norm reaches the specific ratio
// if bStopByResidualRatio is false the method stops when the error norm reaches the specific ratio
// ratioWanted - residual/error norm reduce factor wanted
void ConjugateGradient::RunWithPreconditioner(tfloat h, double ratioWanted, bool bStopByResidualRatio)
{
	int MAX_ITERATIONS = 300;
	Level* pLevel = mg.finestLevel;
	bool bGpu = pLevel->bGpu;
	int access_policy = bGpu ? UM_ACCESS_GPU : UM_ACCESS_CPU;
	double threshold = 1e-10; // will be overwritten below
	double errNorm = 0.0;
	double startErrNorm = 0.0;
	double finishErrNorm = 0.0;
	Timer reductionTimer;

	int stride = pLevel->u.stride;

	Matrix2D x;
	x.AllocateMemory(pLevel->u.nx, pLevel->u.ny, stride, bGpu);
	pLevel->u.CoptyTo(x);
	Matrix2D r; // residual. r  = b-A*x. In each step, this will hold the (opposite of the) gradient.
	//Matrix2D z; // z = M^-1 * r. The residual after multiplying by the residual matrix. The opposite direction of the gradient of the preconditioned system.
	Matrix2D p; // = r (at init). In each step, this will be the direction we go.
	//Matrix2D Ap; // Ap = A*p
	//Matrix2D err; // err = x-u. when u is the solution
	Matrix2D tmp;

	r.AllocateMemory(x.nx, x.ny, stride, bGpu);
	//z.AllocateMemory(x.nx, x.ny, stride, bGpu);
	p.AllocateMemory(x.nx, x.ny, stride, bGpu);
	//Ap.AllocateMemory(x.nx, x.ny, stride, bGpu);
	//err.AllocateMemory(x.nx, x.ny, stride, bGpu);
	tmp.AllocateMemory(x.nx, x.ny, stride, bGpu);

	mg.Residual(pLevel, x, pLevel->f, r); // r is the residual. -r is the gradient.  r = b-Ax.
	ApplyPreconditioner(p, r, h); //z=M^-1 * r
	//tmp.CoptyTo(p);
	//rsold = r'*r
	tfloat rsold = r.Norm(); // this norm is not normalized
	cout << "Initial residual norm: " << rsold << endl;

	if (mg.solution.count > 0)
	{
		x.CoptyTo(tmp);
		tmp.Add(mg.solution, -1);
		startErrNorm = errNorm = tmp.Norm(); // this norm is not normalized
	}
	else
		startErrNorm = errNorm = x.Norm(); // this norm is not normalized

	cout << "Initial error norm: " << startErrNorm << endl;
	if (bStopByResidualRatio)
		threshold = rsold / ratioWanted;
	else
		threshold = startErrNorm / ratioWanted;
	rsold = rsold * rsold;

	tfloat rDotZOld = r.DotProduct(p);
	int iIter = 0;

	for (iIter = 0; iIter < MAX_ITERATIONS; iIter++)
	{
		// Ap = A*p
		ApplyStencil(tmp, p, pLevel);
		tfloat alpha = rDotZOld / p.DotProduct(tmp); // = rsold / (p' * Ap)
		x.Add(p, alpha); // x = x + alpha * p;
		r.Add(tmp, -alpha); // r = r - alpha * Ap;
		reductionTimer.StartMeasure();
		tfloat rsnew = r.Norm(); // rsnew = r' * r; here it is after applying sqrt() to it
		if (isnan(rsnew))
		{
			cout << "Error: residual norm = " << rsnew << endl;
			break;
		}
		reductionTimer.StopMeasure();

		if (mg.solution.count > 0)
		{
			x.CoptyTo(tmp);
			tmp.Add(mg.solution, -1);
			reductionTimer.StartMeasure();
			errNorm = tmp.Norm();
			reductionTimer.StopMeasure();
		}
		else
		{
			reductionTimer.StartMeasure();
			errNorm = x.Norm();
			reductionTimer.StopMeasure();
		}

		errorNorms[iIter] = errNorm;

		tfloat currentNorm = 0;
		if (bStopByResidualRatio)
			currentNorm = rsnew;
		else
			currentNorm = errNorm;

		if (currentNorm < threshold)
		{
			rsold = rsnew * rsnew;
			iIter++;
			break;
		}

		ApplyPreconditioner(tmp, r, h); //z=M^-1 * r
		reductionTimer.StartMeasure();
		tfloat rDotZNew = r.DotProduct(tmp);
		reductionTimer.StopMeasure();
		rsnew = rsnew * rsnew;
		//p = r + (rsnew / rsold) * p;
		p.Multiply(rDotZNew / rDotZOld);
		p.Add(tmp);
		rsold = rsnew;
		rDotZOld = rDotZNew;
	}
	nIteraions = iIter;
	cout << "Final residual norm: " << sqrt(rsold) << endl; // here rsold is the squared norm
	finishErrNorm = errNorm;
	cout << "Final error norm: " << finishErrNorm << endl;
	cout << "Error norm ratio: " << finishErrNorm / startErrNorm << endl;
	//cout << "Total reduction runtimes: " << reductionTimer.GetDuration() << " seconds" << endl; // about 10% of the time appears to be used in reductions

	x.CoptyTo(pLevel->u);
}

void ConjugateGradient::ApplyPreconditioner(Matrix2D& dest, Matrix2D& src, tfloat h)
{
	//src.CoptyTo(dest); //no preconditioner

	/* */
	Level* pLevel = mg.finestLevel;
	//src.CoptyTo(pLevel->f);
	src.Swap(pLevel->f); // The finset right-side values are not changed in the MG cycle
	dest.Swap(pLevel->u); // we don't use original pLevel->f and pLevel->u on finest level at all (always use src and dest)
	pLevel->u.MakeZero();
	mg.KCycle(k, h);
	src.Swap(pLevel->f); // Swap f back
	//pLevel->u.CoptyTo(dest);
	dest.Swap(pLevel->u);
	/* */
}
