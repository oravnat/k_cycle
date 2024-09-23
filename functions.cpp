#include "Classes.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <memory.h>
#include <iostream>

// b must be surounded by zeros:
// the boundary of result is not modified by the function
template <int n>
void conv2d(tfloat result[n][n], const tfloat a[3][3], const tfloat b[n][n])
{
	for (int i = 1; i < n - 1; i++)
	{
		for (int j = 1; j < n - 1; j++)
		{
			tfloat sum = 0.0;
			sum += a[0][0] * b[i - 1][j - 1] + a[0][1] * b[i - 1][j] + a[0][2] * b[i - 1][j + 1];
			sum += a[1][0] * b[i][j - 1] + a[1][1] * b[i][j] + a[1][2] * b[i][j + 1];
			sum += a[2][0] * b[i + 1][j - 1] + a[2][1] * b[i + 1][j] + a[2][2] * b[i + 1][j + 1];
			result[i][j] = sum;
		}
	}

}

// Calc one coarser 9-point stencil using galerkin rule, assuming bi-linear prolongation and full weighting restriction
// in_op and out_op can be the same array
void GalerkinIteration(tfloat out_op[3][3], const tfloat in_op[3][3])
{
	// LH = R*L*P
	tfloat LP[9][9] = { 0.0 };
	tfloat RLP[9][9] = { 0.0 };
	const tfloat P[3][3] = { 0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25 }; //prolongation stencil
	tfloat padded_in_op[9][9] = { 0.0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			padded_in_op[i + 3][j + 3] = in_op[i][j];
		}
	}

	conv2d<9>(LP, P, padded_in_op);
	conv2d<9>(RLP, P, LP);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			out_op[i][j] = RLP[1 + 2 * i + 1][1 + 2 * j + 1] / 4.0; // the first +1 is because of padding, the last +1 is because we want odd indices
		}
	}
}

// Calc one coarser 9-point stencil using galerkin rule with semi-coarsening, assuming linear prolongation on y axis and full weighting restriction
// in_op and out_op can be the same array
void GalerkinIterationY(tfloat out_op[3][3], const tfloat in_op[3][3])
{
	// LH = R*L*P
	tfloat LP[9][9] = { 0.0 };
	tfloat RLP[9][9] = { 0.0 };
	const tfloat P[3][3] = { 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0 }; //1D prolongation stencil
	tfloat padded_in_op[9][9] = { 0.0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			padded_in_op[i + 3][j + 3] = in_op[i][j];
		}
	}

	conv2d<9>(LP, P, padded_in_op);
	conv2d<9>(RLP, P, LP);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			out_op[i][j] = RLP[1 + 2 * i + 1][1 + j + 2] / 2.0; // the first +1 is because of padding, the last +1 is because we want odd indices
		}
	}
}

void PrintStencil(tfloat stencil[3][3])
{
	cout << stencil[0][0] << " | " << stencil[0][1] << " | " << stencil[0][2] << endl;
	cout << stencil[1][0] << " | " << stencil[1][1] << " | " << stencil[1][2] << endl;
	cout << stencil[2][0] << " | " << stencil[2][1] << " | " << stencil[2][2] << endl;
}

void MultiplyStencil(tfloat s[3][3], tfloat a)
{
	for (int i = 0; i < 9; i++)
		s[i / 3][i % 3] *= a;
}

void AddStencil(tfloat a[3][3], tfloat b[3][3])
{
	for (int i = 0; i < 9; i++)
		a[i / 3][i % 3] += b[i / 3][i % 3];
}

void ZeroStencil(tfloat s[3][3])
{
	for (int i = 0; i < 9; i++)
		s[i / 3][i % 3] = 0.0;
}


// cyclic reduction for fixed a, b, c
// n is the number of unknows. n + 1 must be a power of 2. n is always an odd number.
// f is treated as an array of size n.
// The size of u must be at least n + 2. 
// When the function returns, u[0] = u[n+1] = 0, the answer is in indices 1 to n in u.
/*
   b*u(i-2) + a*u(i-1) + c*u(i)                       = f(i-1)
			  b*u(i-1) + a*u(i) + c*u(i+1)            = f(i)
						 b*u(i) + a*u(i+1) + c*u(i+2) = f(i+1)
=>
+  (b*b/a)*u(i-2) + (b)*u(i-1) + (b*c/a)*u(i)                               = f(i-1)*b/a
-		            (b)*u(i-1) + (a)*u(i)     + (c)*u(i+1)                  = f(i)
+				                 (b*c/a)*u(i) + (c)*u(i+1) + (c*c/a)*u(i+2) = f(i+1)*c/a
-----------------------------------------------------------------------------------------
   (b*b/a)*u(i-2) + [2*(b*c/a)-(a)]*u(1) + (c*c/a)*u(i+2) = f(i-1)*b/a - f(i) + f(i+1)*c/a
*/
void CyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n)
{
	const int MAX_N = 8 * 1024 + 2;
	tfloat aa, bb, cc;
	tfloat ff[MAX_N], uu[MAX_N];

	int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n

	if (n == 1)
	{
		u[0] = 0.0;
		u[1] = f[0] / a;
		u[2] = 0.0;
		return;
	}

	//cout << a << " "; // for debug
	bb = b * b / a;
	cc = c * c / a;
	aa = 2 * b * c / a - a;
	for (int ii = 0, i = 1; ii < nn; ii++, i += 2)
	{
		ff[ii] = f[i - 1] * b / a - f[i] + f[i + 1] * c / a;
	}
	CyclicReduction(aa, bb, cc, ff, uu, nn);
	for (int ii = 0, i = 0; ii < nn + 2; ii++, i += 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
	{
		u[i] = uu[ii];
	}
	for (int ii = 0, i = 1; ii < nn + 1; ii++, i += 2)
	{
		u[i] = (f[i - 1] - b * uu[ii] - c * uu[ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
	}
}

void IterativeCyclicReduction(tfloat a, tfloat b, tfloat c, const tfloat* f, tfloat* u, int n)
{
	const int MAX_LEVELS = 14;
	const int MAX_N = 2 * 8 * 1024 + 2;
	tfloat as[MAX_LEVELS], bs[MAX_LEVELS], cs[MAX_LEVELS];
	tfloat fs[MAX_N], us[MAX_N];

	//#pragma omp for schedule(static)
	for (int i = 0; i < n; i++)
		fs[i] = f[i];

	int s = 0; // current level start index 
	int ic = n + 2; //coarser level start index

	int iLevel = 0;
	while (n > 1)
	{
		int nn = n / 2; // n is an odd number, so nn is rounded down, nn*2+1=n
		tfloat fl = b / a;
		tfloat fu = c / a;

		/*for (int ii = 0, i = 1; ii < nn; ii++, i += 2)
		{
			fs[ic + ii] = fl * fs[s + i - 1] - fs[s + i] + fu * fs[s + i + 1];
		}*/
		//#pragma omp for schedule(static)
		for (int ii = 0; ii < nn; ii++)
		{
			int i = 1 + 2 * ii;
			fs[ic + ii] = fl * fs[s + i - 1] - fs[s + i] + fu * fs[s + i + 1];
		}
		as[iLevel] = a;
		bs[iLevel] = b;
		cs[iLevel] = c;

		// prepare next level:
		a = 2 * b * fu - a; // OK only for constant values of a, b, c
		b = fl * b;
		c = fu * c;

		iLevel++;
		n = nn;

		s = ic;
		ic += nn + 2;
	}

	if (n == 1) // must be true here
	{
		us[s+0] = 0.0;
		us[s+1] = fs[s+0] / a;
		us[s+2] = 0.0;
	}

	while (--iLevel >= 0)
	{
		int nn = n;
		n = n * 2 + 1;
		ic = s;
		s -= n + 2;
		a = as[iLevel];
		b = bs[iLevel];
		c = cs[iLevel];
		/*for (int ii = 0, i = 0; ii < nn + 2; ii++, i += 2) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		{
			us[s + i] = us[ic + ii];
		}*/
		//#pragma omp for schedule(static)
		for (int ii = 0; ii < nn + 2; ii++) //including u[0] = uu[0] = 0 and u[n+1]=uu[nn+1]=0
		{
			int i = 2 * ii;
			us[s+i] = us[ic + ii];
		}
		/*for (int ii = 0, i = 1; ii < nn + 1; ii++, i += 2)
		{
			us[s + i] = (fs[s + i - 1] - b * us[ic + ii] - c * us[ic + ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}*/
		//#pragma omp for schedule(static)
		for (int ii = 0; ii < nn + 1; ii++)
		{
			int i = 1 + 2 * ii;
			us[s+i] = (fs[s + i - 1] - b * us[ic + ii] - c * us[ic + ii + 1]) / a; // u[1] is the first value in u, while f[0] is the first in f
		}
	}

	//#pragma omp for schedule(static)
	for (int i = 0; i < n + 2; i++)
		u[i] = us[i];
}

//void IterativeCyclicReductionInPlace(tfloat* __restrict__ B, tfloat a, tfloat b, tfloat c, int n)
void IterativeCyclicReductionInPlace(tfloat* B, tfloat a, tfloat b, tfloat c, int n)
{
	const int MAX_LEVELS = 14;
	tfloat as[MAX_LEVELS], bs[MAX_LEVELS], cs[MAX_LEVELS];
	int nn = n;
	int stride = 1;
	int iLevel = 0;
	while (nn > 1)
	{
		tfloat fl = b / a;
		tfloat fu = c / a;
		nn = nn / 2; // nn is an odd number, so nn is rounded down, nn(new)*2+1=nn(old)
		for (int ii = 0; ii < nn; ii += 1)
		{
			int i = (ii+1) * 2 * stride; // the additional +1 is because the first non-zero value is at index 1 in B
			B[i] = fl * B[i - stride] - B[i] + fu * B[i + stride];
		}
		as[iLevel] = a;
		bs[iLevel] = b;
		cs[iLevel] = c;

		// prepare next level:
		a = 2 * b * fu - a; // OK only for constant values of a, b, c
		b = fl * b;
		c = fu * c;

		iLevel++;
		stride *= 2;
	}

	//if (nn == 1) // must be true here
	B[stride] = B[stride] / a;

	while (--iLevel >= 0)
	{
		stride /= 2;
		a = as[iLevel];
		b = bs[iLevel];
		c = cs[iLevel];
		for (int ii = 0; ii < nn + 1; ii += 1)
		{
			B[stride + 2 * ii * stride] = (B[stride + 2 * ii * stride] - b * B[2 * ii * stride] - c * B[2 * (ii + 1) * stride]) / a;
		}
		nn = nn * 2 + 1;
	}
}