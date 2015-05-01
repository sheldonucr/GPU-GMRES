/*!	\file
	\brief the definations for the functions used for the gmres solving routine
*/

#include "gmres.h"

// zky
float difftime(timeval &st, timeval &et){
	float difftime = 0.0f;
	difftime = (et.tv_sec-st.tv_sec)*1000.0 + (et.tv_usec - st.tv_usec)/1000.0;
	return difftime;
}


// Note the transpose storage in array
float mat_get(const float *A, const int row, const int col,
		const  int numRows, const  int numCols)
{
	return *(A + col*numRows + row);
}

// Note the transpose storage in array
void mat_set(float *A, const float alpha, const int row, const int col,
		const  int numRows, const  int numCols)
{
	*(A + col*numRows + row) = alpha;
}

// set vector to a value
void vec_initial(float *v, const float alpha, const  int n)
{
	for(int i=0; i<n; i++) {
		v[i] = alpha;
	}
}

void copy(float *x, float *y, const int n)
{
	for (int i=0; i<n; i++)
		x[i] = y[i];
}

// v = alpha*x
void sscal(float *v, const float *x, const float alpha, const  int n)
{
	for(int i=0; i<n; i++) {
		v[i] = alpha*x[i];
	}
}

// y = alpha*x + y
void sapxy(float *y, const float *x, const float alpha, const  int n)
{
	for(int i=0; i<n; i++) {
		y[i] = alpha*x[i] + y[i];
	}
}

float norm2(const float *v, const  int n)
{
	float tmp = 0;
	for(int i=0; i<n; i++)
		tmp += v[i] * v[i];
	return sqrt(tmp);
}

float dot(const float *x, const float *y, const  int n)
{
	float tmp = 0;
	for(int i=0; i<n; i++)
		tmp += x[i] * y[i];
	return tmp;
}

// y = alpha*A*x + beta*y
void sgemv(float *v,
		const float *val, const  int *rowIndices, const  int *indices,
		const float alpha, const float *x, const float beta, const float *y,
		const  int numRows, const  int numCols)
{
	float *tmp_vec = (float*) malloc(numRows*sizeof(float));
	computeSpMV(tmp_vec, val, rowIndices, indices, x, numRows);
	for(int i=0; i<numRows; i++) {
		v[i] = alpha*tmp_vec[i] + beta*y[i];
	}
	free(tmp_vec);
}


//! original update operations
	void 
Update(float *x, const int k, const float *H, const int m,
		const float *s, const float *v,
		const int n)
{
	int i=0, j=0;
	float *y = (float*) malloc((k+1)*sizeof(float));
	for (i=0; i<k+1; i++)  
		y[i] = s[i];

	// Back substituation, H is upper triangle matrix
	// solve y, where H*y = s
	for (i = k; i >= 0; i--) {
		y[i] /= *(H + i + i*(m+1));// the diagonal element
		for (j=i-1; j >= 0; j--)
			y[j] -= *(H + j + i*(m+1)) * y[i];
	}

	// x = v*y
	for (j = 0; j <= k; j++)
		for (i=0; i < n; i++)
			x[i] += *(v + i + j*n) * y[j];

	free(y);
}



//! this function is for right DIAG precondition
	void 
Update_precondition(float *x, const int k, const float *H, const int m,
		const float *s, const float *v,
		const int n, 
		const float* m_val, const  int* m_rowIndices, const  int* m_indices)
{
	int i=0, j=0;
	float *y = (float*) malloc((k+1)*sizeof(float));
	for (i=0; i<k+1; i++)  
		y[i] = s[i];

	// Back substituation, H is upper triangle matrix
	// solve y, where H*y = s
	for (i = k; i >= 0; i--) {
		y[i] /= *(H + i + i*(m+1));// the diagonal element
		for (j=i-1; j >= 0; j--)
			y[j] -= *(H + j + i*(m+1)) * y[i];
	}

	//---------- precondition operation ----------
	float *z = (float*) malloc(n*sizeof(float));
	float *z1 = (float*) malloc(n*sizeof(float));

	// z = v*y
	for (j = 0; j <= k; j++)
		for (i=0; i < n; i++)
			z[i] = *(v + i + j*n) * y[j];

	// z = (M^-1)*z
	computeSpMV(z1, m_val, m_rowIndices, m_indices, z, n);


	// x += z
	for(i = 0; i < n; ++i)
		x[i] += z1[i];

	free(y);
	free(z);
	free(z1);
}



	void 
Update_GPU(float *d_x, const int k, const float *H, const int m,
		const float *s, const float *d_v, const int n)
{
	int i=0, j=0;
	float *h_y = (float*) malloc((k+1)*sizeof(float));
	float *d_y; cudaMalloc((void**) &d_y, (k+1)*sizeof(float));
	for (i=0; i<k+1; i++){
		h_y[i] = s[i];
	}

	// Backsolve:
	for (i = k; i >= 0; i--) {
		h_y[i] /= *(H+i+i*(m+1));
		for (j=i-1; j >= 0; j--)
			h_y[j] -= *(H+j+i*(m+1)) * h_y[i];
	}

	// copy to device mem for cuBLAS computation of x = x + V*y
	cudaMemcpy(d_y, h_y, (k+1)*sizeof(float), cudaMemcpyHostToDevice);
	cublasSgemv('n', n, k+1, 1.0, d_v, n,
			d_y, 1,
			1.0, d_x, 1);

	free(h_y);  cudaFree(d_y);
}


void ApplyPlaneRotation(float *dx, float *dy, float cs, float sn)
{
	float temp  =  cs * (*dx) + sn * (*dy);
	*dy = -sn * (*dx) + cs * (*dy);
	*dx = temp;
}


void GeneratePlaneRotation(float dx, float dy, float *cs, float *sn)
{
	if (dy == 0.0) {
		*cs = 1.0;
		*sn = 0.0;
	}
	else if (fabs(dy) > fabs(dx)) {
		float temp = dx / dy;
		*sn = 1.0 / sqrt( 1.0 + temp*temp );
		*cs = temp * (*sn);
	}
	else {
		float temp = dy / dx;
		*cs = 1.0 / sqrt( 1.0 + temp*temp );
		*sn = temp * (*cs);
	}
}


//! the GMRES with left diagonal preconditioner on CPU side
int 
GMRES_leftDiag(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const float *m_val, const  int *m_rowIndices, const  int *m_indices)// n: rowNum, m: restart threshold, with m is a inverse matrix forum
{
	float resid;
	int i, j = 1, k;

	//Vector s(m+1), cs(m+1), sn(m+1), w;
	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *ww = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *rr = (float*) malloc(n*sizeof(float));
	float *bb = (float*) malloc(n*sizeof(float));

	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = norm2(b, n);
	computeSpMV(bb, m_val, m_rowIndices, m_indices, b, n);
	float normb = norm2(bb, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	computeSpMV(r, m_val, m_rowIndices, m_indices, rr, n);

	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	resid = norm2(r, n) / normb;

	if ((resid = norm2(r, n) / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;

		free(s);
		free(cs);
		free(sn);
		free(w);
		free(ww);
		free(bb);
		free(H);
		free(v);

		cout<<endl;
		return 0;
	}

	while (j <= *max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(s, 0.0, m+1);
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);
			//computeSpMV(w, val, rowIndices, indices, v+i*n, n);
			computeSpMV(ww, val, rowIndices, indices, v+i*n, n);
			computeSpMV(w, m_val, m_rowIndices, m_indices, ww, n);

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);
				Update(x, i, H, m, s, v, n);

				*tol = resid;
				*max_iter = j;

				free(s);
				free(cs);
				free(sn);
				free(w);
				free(ww);
				free(bb);
				free(H);
				free(v);

				cout<<endl;
				return 0;
			}
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;

		}// end of for (i = 0; i < m && j <= *max_iter; i++, j++)

		Update(x, m-1, H, m, s, v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		computeSpMV(r, m_val, m_rowIndices, m_indices, rr, n);


		beta = norm2(r, n);
		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(s);
			free(cs);
			free(sn);
			free(w);
			free(ww);
			free(bb);
			free(H);
			free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	*tol = resid;

	free(s);
	free(cs);
	free(sn);
	free(w);
	free(ww);
	free(bb);
	free(H);
	free(v);

	return 1;
}

/*
void myAinvPrecond_old(const MyAINV_old &myAinv, const float *i_array, int n, 
		float *o_array){

	assert(myAinv.dim == n);
	float *temp = new float[n];

	// o_array = w_t * i_array
	computeSpMV(o_array, myAinv.w_t_val, myAinv.w_t_rowIndices, myAinv.w_t_indices, i_array, n);

	// temp = (D^-1) * w_t * i_array
	for(int i=0; i<n; ++i){
		assert(!Equal(myAinv.diag_val[i], 0.0f));
		temp[i] = o_array[i] / myAinv.diag_val[i];
	}

	// o_array = z * (D^-1) * w_t * i_array
	computeSpMV(o_array, myAinv.z_val, myAinv.z_rowIndices, myAinv.z_indices, temp, n);

	delete [] temp;

}

void myAinvPrecond(const MyAINV_old &myAinv, const float *i_array, int n, 
		float *o_array){

	assert(myAinv.dim == n);
	float *temp = new float[n];

	// o_array = w_t * i_array
	computeSpMV(o_array, myAinv.z_val, myAinv.z_rowIndices, myAinv.z_indices, i_array, n);

	// temp = (D^-1) * w_t * i_array
	for(int i=0; i<n; ++i){
		//assert(!Equal(myAinv.diag_val[i], 0.0f));
		temp[i] = o_array[i] * myAinv.diag_val[i];
	}

	// o_array = z * (D^-1) * w_t * i_array
	computeSpMV(o_array, myAinv.w_t_val, myAinv.w_t_rowIndices, myAinv.w_t_indices, temp, n);

	delete [] temp;

}

int 
GMRES_cpu_AINV(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const MyAINV_old &myAinv)// n: rowNum, m: restart threshold, with m is a inverse matrix forum
{
	float resid;
	int i, j = 1, k;

	//Vector s(m+1), cs(m+1), sn(m+1), w;
	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *ww = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *rr = (float*) malloc(n*sizeof(float));
	float *bb = (float*) malloc(n*sizeof(float));

	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	myAinvPrecond(myAinv, b, n, bb);
	float normb = norm2(bb, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	myAinvPrecond(myAinv, rr, n, r);

	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	resid = norm2(r, n) / normb;

	if ((resid = norm2(r, n) / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;

		free(s);
		free(cs);
		free(sn);
		free(w);
		free(ww);
		free(bb);
		free(H);
		free(v);

		cout<<endl;
		return 0;
	}

	while (j <= *max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(s, 0.0, m+1);
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);
			//computeSpMV(w, val, rowIndices, indices, v+i*n, n);
			computeSpMV(ww, val, rowIndices, indices, v+i*n, n);
			myAinvPrecond(myAinv, ww, n, w);

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);
				Update(x, i, H, m, s, v, n);

				*tol = resid;
				*max_iter = j;

				free(s);
				free(cs);
				free(sn);
				free(w);
				free(ww);
				free(bb);
				free(H);
				free(v);

				cout<<endl;
				return 0;
			}
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;
		}// end of for (i = 0; i < m && j <= *max_iter; i++, j++)

		Update(x, m-1, H, m, s, v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		myAinvPrecond(myAinv, rr, n, r);


		beta = norm2(r, n);
		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(s);
			free(cs);
			free(sn);
			free(w);
			free(ww);
			free(bb);
			free(H);
			free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	*tol = resid;

	free(s);
	free(cs);
	free(sn);
	free(w);
	free(ww);
	free(bb);
	free(H);
	free(v);

	cout<<endl;
	return 1;
}

*/

//! the GMRES method with left ILU0 preconditioner on CPU side
int 
GMRES_leftILU0(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const float *l_val, const int *l_rowIndices, const int *l_indices, 
		const float *u_val, const int *u_rowIndices, const int *u_indices)
{
	float resid;
	int i, j = 1, k;

	//Vector s(m+1), cs(m+1), sn(m+1), w;
	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *ww = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *rr = (float*) malloc(n*sizeof(float));
	float *bb = (float*) malloc(n*sizeof(float));

	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = norm2(b, n);
	LUSolve_ignoreZero(bb, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices, b, n);
	float normb = norm2(bb, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	LUSolve_ignoreZero(r, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices, rr, n);


	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	resid = norm2(r, n) / normb;

	if ((resid = norm2(r, n) / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;

		free(s);
		free(cs);
		free(sn);
		free(w);
		free(ww);
		free(bb);
		free(H);
		free(v);

		cout<<endl;
		return 0;
	}

	while (j <= *max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(s, 0.0, m+1);
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);
			//computeSpMV(w, val, rowIndices, indices, v+i*n, n);
			computeSpMV(ww, val, rowIndices, indices, v+i*n, n);
			LUSolve_ignoreZero(w, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices, ww, n);

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);
				Update(x, i, H, m, s, v, n);

				*tol = resid;
				*max_iter = j;

				free(s);
				free(cs);
				free(sn);
				free(w);
				free(ww);
				free(bb);
				free(H);
				free(v);

				cout<<endl;
				return 0;
			}
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;

		}// end of for (i = 0; i < m && j <= *max_iter; i++, j++)

		Update(x, m-1, H, m, s, v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		LUSolve_ignoreZero(r, l_val, l_rowIndices, l_indices, u_val, u_rowIndices, u_indices, rr, n);


		beta = norm2(r, n);
		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(s);
			free(cs);
			free(sn);
			free(w);
			free(ww);
			free(bb);
			free(H);
			free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	*tol = resid;

	free(s);
	free(cs);
	free(sn);
	free(w);
	free(ww);
	free(bb);
	free(H);
	free(v);

	cout<<endl;
	return 1;
}


//! the original GMRES method without preconditioner on CPU side
int 
GMRES(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol)// n: rowNum, m: restart
{
	float resid;
	int i, j = 1, k;

	//Vector s(m+1), cs(m+1), sn(m+1), w;
	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	float normb = norm2(b, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);

	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	if ((resid = norm2(r, n) / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;

		free(s);
		free(cs);
		free(sn);
		free(w);
		free(H);
		free(v);

		cout<<endl;
		return 0;
	}


	while (j <= *max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(s, 0.0, m+1);
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {

			// XXLiu: w = M.solve(A * v[i]);
			computeSpMV(w, val, rowIndices, indices, v+i*n, n);

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);
				Update(x, i, H, m, s, v, n);

				*tol = resid;
				*max_iter = j;

				free(s);
				free(cs);
				free(sn);
				free(w);
				free(H);
				free(v);

				cout<<endl;
				return 0;
			}
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;

		}// end of for (i = 0; i < m && j <= *max_iter; i++, j++)

		Update(x, m-1, H, m, s, v, n);

		// XXLiu: r = M.solve(b - A * x);
		sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		beta = norm2(r, n);
		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(s);
			free(cs);
			free(sn);
			free(w);
			free(H);
			free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	*tol = resid;

	free(s);
	free(cs);
	free(sn);
	free(w);
	free(H);
	free(v);

	cout<<endl;
	return 1;
}



//! the right diag preconditioned GMRES on CPU side
/*!
  \m_val	the value array of the right diagonal matrix
  \m_rowIndices	the row ptr of the right diagnoal matrix
  \m_indices	the row indices of the right diagonal matrix
 */
int 
GMRES_right_diag(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter,
		float *tol, 
		const float *m_val, const  int *m_rowIndices, const  int *m_indices)// n: rowNum, m: restart threshold
{

	float resid;
	int i, j = 1, k;

	//Vector g(m+1), cs(m+1), sn(m+1), w;
	float *g = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	float *z = (float*) malloc(n*sizeof(float));// used for precondition, added by zky

	// XXLiu:  normb = norm( M.solve(b) )
	float normb = norm2(b, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);

	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	if ((resid = norm2(r, n) / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;

		free(g);
		free(cs);
		free(sn);
		free(w);
		free(H);
		free(v);

		cout<<endl;
		return 0;
	}

	while (j <= *max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(g, 0.0, m+1);
		g[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);

			//---------- precondition operation ----------
			//computeSpMV(w, val, rowIndices, indices, v+i*n, n);
			computeSpMV(z, m_val, m_rowIndices, m_indices, v+i*n, n);// z = (M^-1) * v[i]
			computeSpMV(w, val, rowIndices, indices, z, n);// w = A * z
			//--------------------------------------------


			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( g+i, g+(i+1), cs[i], sn[i]);

			if ((resid = fabs(g[i+1]) / normb) < *tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);


				//---------- precondition operation ----------
				Update(x, i, H, m, g, v, n);
				Update_precondition(x, m-1, H, m, g, v, n, m_val, m_rowIndices, m_indices);// added by zky
				//--------------------------------------------

				*tol = resid;
				*max_iter = j;

				free(g);
				free(cs);
				free(sn);
				free(w);
				free(H);
				free(v);

				cout<<endl;
				return 0;
			}
			printf("HOST---resid: %6.4e < %6.4e\n",resid, *tol);
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;

		}// end of for (i = 0; i < m && j <= *max_iter; i++, j++)


		//---------- precondition operation ----------
		Update(x, m-1, H, m, g, v, n);
		Update_precondition(x, m-1, H, m, g, v, n, m_val, m_rowIndices, m_indices);// added by zky
		//--------------------------------------------


		// XXLiu: r = M.solve(b - A * x);
		sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		beta = norm2(r, n);
		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(g);
			free(cs);
			free(sn);
			free(w);
			free(H);
			free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	*tol = resid;

	free(g);
	free(cs);
	free(sn);
	free(w);
	free(H);
	free(v);

	cout<<endl;
	return 1;
}


//==============================================
// v = alpha*A*x + beta*y
void sgemv_GPU(float *v,
		const SpMatrixGPU *Sparse, const SpMatrix *spm,
		dim3 *grid, dim3 *block,
		const float alpha, const float *x,
		const float beta, const float *y,
		const  int numRows, const  int numCols)
{
	cublasScopy(numRows, y, 1, v, 1); // y --> v 
	cublasSscal(numRows, beta, v, 1); // v*beta --> v

	float *tmp_vector; // Device Memory
	cudaMalloc((void**) &tmp_vector, numRows*sizeof(float));
	//cudaThreadSynchronize();
	SpMV<<<*grid,*block>>>(tmp_vector, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices,
			x, numRows, numCols, spm->numNZEntries);
	cudaThreadSynchronize();

	cublasSaxpy(numRows, alpha, tmp_vector, 1, v, 1);

	cudaFree(tmp_vector);
}

//! pre-process the matrix to get the level of the triangle matrix, useless after the use of cusparse library
void getLevel(bool isLmatrix, const float* val, const  int* rowIndices, const  int* indices, int* level, int n){
	for(int i=0; i<n; ++i){// initialize the level
		level[i] = 0;
	}

	if(isLmatrix == true){// decide the level of the L matrix
		for(int i=1; i<n; ++i){
			int lb = rowIndices[i];
			int ub = rowIndices[i+1];
			for(int j=lb; j<ub; ++j){
				if(Equal(val[j], 0)){
					break;// based on the fact that all the padded element are on the end of a row
					//continue;
				}
				else if(indices[j] >= i){// based on the assumption that all the element on the row are sorted by the column index
					break;
				}
				else if(indices[j] < i){
					if((level[indices[j]] + 1) > level[i]){
						level[i] = (level[indices[j]] + 1);
					}
				}
			}
		}
	}
	else{// decide the level of the U matrix
		for(int i=n-2; i>=0; --i){
			int lb = rowIndices[i];
			int ub = rowIndices[i+1];
			for(int j=ub-1; j>=lb; --j){
				if(Equal(val[j], 0)){
					continue;// the padded element on the end, just ignore
				}
				else if(indices[j] <= i){// traversal to the L matrix
					break;
				}
				else{
					if((level[indices[j]] + 1) > level[i]){
						level[i] = (level[indices[j]] + 1);
					}
				}
			}
		}
	}
}


//! the original solving of GMRES on GPU without preconditioner
int 
GMRES_GPU(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter, float *tol)
{
	float resid;
	int i, j = 1, k;

	float *d_r;  cudaMalloc((void**) &d_r, n*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	float normb = cublasSnrm2(n, d_b, 1);
	if (normb == 0.0)  normb = 1;

	// XXLiu: Vector r = M.solve(b - A * x);
	sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	float beta = cublasSnrm2(n, d_r, 1);
	if ((resid = beta / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;
		cudaFree(d_r);
		cout<<endl;
		return 0;
	}

	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *H = (float*) malloc(m*(m+1)*sizeof(float));

	float *d_v; cudaMalloc((void**) &d_v, (m+1)*n*sizeof(float));
	float *d_w; cudaMalloc((void**) &d_w, n*sizeof(float));

	while (j <= *max_iter) {
		cublasScopy(n, d_r, 1, d_v, 1);
		cublasSscal(n, 1.0/beta, d_v, 1); // XXLiu: v[0] = r * (1.0 / beta);

		vec_initial(s, 0.0, m+1); s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);
			SpMV<<<*grid, *block>>>(d_w, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices,
					d_v+i*n, n, n, spm->numNZEntries);
			cudaThreadSynchronize();

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = cublasSdot(n, d_w, 1, d_v+k*n, 1); // XXLiu: H(k, i) = dot(w, v[k]);
				cublasSaxpy(n, -*(H+k+i*(m+1)), d_v+k*n, 1, d_w, 1);// XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = cublasSnrm2(n, d_w, 1); // XXLiu: H(i+1, i) = norm(w);
			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			cublasScopy(n, d_w, 1, d_v+(i+1)*n, 1);
			cublasSscal(n, 1.0/(*(H+(i+1)+i*(m+1))), d_v+(i+1)*n, 1);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("DEV---BREAK: %6.4e\n",resid);
				Update_GPU(d_x, i, H, m, s, d_v, n);

				*tol = resid;
				*max_iter = j;

				free(H);   free(s);   free(cs);   free(sn);
				cudaFree(d_v);	cudaFree(d_w);	cudaFree(d_r);
				cout<<endl;
				return 0;
			}
			cout<<"DEV---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;
		}
		Update_GPU(d_x, m-1, H, m, s, d_v, n);

		// XXLiu: r = M.solve(b - A * x);
		sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
		beta = cublasSnrm2(n, d_r, 1);

		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(H);      free(s);      free(cs);      free(sn);
			cudaFree(d_v);      cudaFree(d_w);      cudaFree(d_r);
			cout<<endl;
			return 0;
		}
	}

	*tol = resid;

	free(H);  free(s);  free(cs);  free(sn);
	cudaFree(d_v);  cudaFree(d_w);  cudaFree(d_r);

	cout<<endl;
	return 1;
}


int 
GMRES_GPU_leftDiag(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter, float *tol, 
		const float* m_val, const int* m_rowIndices, const int* m_indices)
{

	float resid;
	int i, j = 1, k;

	int m_nnz = m_rowIndices[n];
	float *d_m_val;
	int *d_m_rowIndices, *d_m_indices;
	cudaMalloc((void**)&d_m_val, m_nnz * sizeof(float));
	cudaMalloc((void**)&d_m_rowIndices, (n + 1) * sizeof(int));
	cudaMalloc((void**)&d_m_indices, m_nnz * sizeof(int));

	cudaMemcpy(d_m_val, m_val, m_nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m_rowIndices, m_rowIndices, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m_indices, m_indices, m_nnz * sizeof(int), cudaMemcpyHostToDevice);


	float *d_r;
	cudaMalloc((void**) &d_r, n*sizeof(float));
	float *d_rr;
	cudaMalloc((void**) &d_rr, n*sizeof(float));

	float *d_bb;
	cudaMalloc((void**)& d_bb, n*sizeof(float));

	float *d_temp;
	cudaMalloc((void**)& d_temp, n*sizeof(float));


	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = cublasSnrm2(n, d_b, 1);
	SpMV<<<*grid, *block>>>(d_bb, d_m_val, d_m_rowIndices, d_m_indices, d_b, n, n, n);
	cudaThreadSynchronize();
	float normb = cublasSnrm2(n, d_bb, 1);
	if (normb == 0.0)  
		normb = 1;

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	SpMV<<<*grid, *block>>>(d_r, d_m_val, d_m_rowIndices, d_m_indices, d_rr, n, n, n);
	cudaThreadSynchronize();

	float beta = cublasSnrm2(n, d_r, 1);
	if ((resid = beta / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;
		cudaFree(d_r);
		cout<<endl;
		return 0;
	}

	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *H = (float*) malloc(m*(m+1)*sizeof(float));

	float *d_v; cudaMalloc((void**) &d_v, (m+1)*n*sizeof(float));
	float *d_w; cudaMalloc((void**) &d_w, n*sizeof(float));
	float *d_ww; cudaMalloc((void**) &d_ww, n*sizeof(float));

#if CUSPARSE_FLAG
	float floatOne[1] = {1};
	float floatZero[1] = {0};
	float floatMinusOne[1] = {-1};
#endif

	while (j <= *max_iter) {

		timeSentence(cublasScopy(n, d_r, 1, d_v, 1), time_cublas);
		timeSentence(cublasSscal(n, 1.0/beta, d_v, 1), time_cublas); // XXLiu: v[0] = r * (1.0 / beta);

		vec_initial(s, 0.0, m+1); 
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {

			// XXLiu: w = M.solve(A * v[i]);
#if SUB_TIMER
			gettimeofday(&st, NULL);
#endif


#if CUSPARSE_FLAG
			status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, spm->numNZEntries, floatOne, A_des, Sparse->d_val, (const int*)Sparse->d_rowIndices, (const int*)Sparse->d_indices, d_v+i*n, floatZero, d_ww);
			assert(status == CUSPARSE_STATUS_SUCCESS);
#else
			SpMV<<<*grid, *block>>>(d_ww, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices, d_v+i*n, n, n, spm->numNZEntries);
#endif
			cudaThreadSynchronize();


#if SUB_TIMER
			gettimeofday(&et, NULL);
			time_spmv += difftime(st, et);
#endif

			SpMV<<<*grid, *block>>>(d_w, d_m_val, d_m_rowIndices, d_m_indices, d_ww, n, n, n);
			cudaThreadSynchronize();


			for (k = 0; k <= i; k++) {
				timeSentence(*(H+k+i*(m+1)) = cublasSdot(n, d_w, 1, d_v+k*n, 1), time_cublas); // XXLiu: H(k, i) = dot(w, v[k]);
				timeSentence(cublasSaxpy(n, -*(H+k+i*(m+1)), d_v+k*n, 1, d_w, 1), time_cublas);// XXLiu: w -= H(k, i) * v[k];
			}
			timeSentence(*(H+(i+1)+i*(m+1)) = cublasSnrm2(n, d_w, 1), time_cublas); // XXLiu: H(i+1, i) = norm(w);
			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			timeSentence(cublasScopy(n, d_w, 1, d_v+(i+1)*n, 1), time_cublas);
			timeSentence(cublasSscal(n, 1.0/(*(H+(i+1)+i*(m+1))), d_v+(i+1)*n, 1), time_cublas);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("DEV---BREAK: %6.4e\n",resid);
				Update_GPU(d_x, i, H, m, s, d_v, n);

				*tol = resid;
				*max_iter = j;

#if SUB_TIMER
				summary_time();
#endif

				free(H);   free(s);   free(cs);   free(sn);
				cudaFree(d_v);	cudaFree(d_w);	cudaFree(d_r);
				cout<<endl;
				return 0;
			}
			cout<<"DEV---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;
		}
		Update_GPU(d_x, m-1, H, m, s, d_v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);

#if SUB_TIMER
		gettimeofday(&st, NULL);
#endif

#if CUSPARSE_FLAG
		// copy d_b -> d_ww
		cublasScopy(n, d_b, 1, d_rr, 1);
		// d_rr = (-1)*A*x + 1*b
		status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, spm->numNZEntries, floatMinusOne, A_des, Sparse->d_val, (const int*)Sparse->d_rowIndices, (const int*)Sparse->d_indices, d_x, floatOne, d_rr);
		assert(status == CUSPARSE_STATUS_SUCCESS);
#else
		sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
#endif
		cudaThreadSynchronize();

#if SUB_TIMER
		gettimeofday(&et, NULL);
		time_spmv += difftime(st, et);
#endif


		SpMV<<<*grid, *block>>>(d_r, d_m_val, d_m_rowIndices, d_m_indices, d_rr, n, n, n);
		cudaThreadSynchronize();

		beta = cublasSnrm2(n, d_r, 1);

		if ((resid = beta / normb) < *tol) {

#if SUB_TIMER
			summary_time();
#endif
			*tol = resid;
			*max_iter = j;

			free(H);      free(s);      free(cs);      free(sn);
			cudaFree(d_v);      cudaFree(d_w);      cudaFree(d_r);
			cout<<endl;
			return 0;
		}
	}

#if SUB_TIMER
	summary_time();
#endif

	*tol = resid;

	free(H);  free(s);  free(cs);  free(sn);

	cudaFree(d_v);  cudaFree(d_w);  cudaFree(d_r);

	cudaFree(d_m_val); cudaFree(d_m_rowIndices); cudaFree(d_m_indices);

	cout<<endl;
	return 1;
}


// solve the equation of L*U*x = y on GPU, where the L and U matrix are stored in (val, rowIndices, indices)
/*!
  \brief	this function employs cusparse library to solve the sparse triangle system, the set up and analisis part were carried out out of this function to save time, because these parts only need to carried out one time
  \L_des	the description of the L matrix
  \U_des	the description of the U matrix
  \handle	the handle for the use of cusparse library
  \status	the status of cusparse execution
  \L_info	the information of the L matrix
  \U_info the information of the U matrix
 */
void LUSolve_gpu(float *x, 
		const float *l_val,	const int *l_rowIndices, const int *l_indices, 
		const float *u_val,	const int *u_rowIndices, const int *u_indices, 
		const float *y, const int numRows, const float* alpha, 
		cusparseMatDescr_t& L_des, cusparseMatDescr_t& U_des, cusparseHandle_t& handle, cusparseStatus_t& status, 
		cusparseSolveAnalysisInfo_t& L_info, cusparseSolveAnalysisInfo_t& U_info,  
		float * v)// the intermidate vector to save the malloc time
{
#if SUB_TIMER
	gettimeofday(&st, NULL);
#endif

	status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, alpha, L_des, l_val, l_rowIndices, l_indices, L_info, y, v);

	status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, alpha, U_des, u_val, u_rowIndices, u_indices, U_info, v, x);

	cudaThreadSynchronize();

#if SUB_TIMER
	gettimeofday(&et, NULL);
	time_trisolve += difftime(st, et);
#endif

	if(status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to SOLVE the matrix!!!\n");
		printf("Error code %d\n", status);
		assert(false);
	}

}


//! this function employs left ILU0 preconditioner too solve linear system with GMRES, the precondition part is implemented by zky
/*!
  \l_val	the value array of L matrix part of preconditioner
  \l_rowIndices	the row ptr of L matrix part of preconditioner
  \l_indices	the column array of L matrix part of preconditioner
  \return 0:	if GMRES succed to convergence within a reasonable number of iterations, 1:	otherwise
 */
int 
GMRES_GPU_leftILU0(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter, float *tol, 
		const float* l_val, const int* l_rowIndices, const int* l_indices, 
		const float* u_val, const int* u_rowIndices, const int* u_indices)
{

	float resid;
	int i, j = 1, k;

	float *d_r;
	cudaMalloc((void**) &d_r, n*sizeof(float));
	float *d_rr;
	cudaMalloc((void**) &d_rr, n*sizeof(float));

	float *d_bb;
	cudaMalloc((void**)& d_bb, n*sizeof(float));

	float *d_temp;
	cudaMalloc((void**)& d_temp, n*sizeof(float));

	cusparseStatus_t status;
	cusparseHandle_t handle;
	cusparseCreate(&handle);

	int cusparse_version;
	cusparseGetVersion(handle, &cusparse_version);

	cusparseSolveAnalysisInfo_t L_info, U_info;
	assert((status = cusparseCreateSolveAnalysisInfo(&L_info)) == CUSPARSE_STATUS_SUCCESS);
	assert((status = cusparseCreateSolveAnalysisInfo(&U_info)) == CUSPARSE_STATUS_SUCCESS);

	float alpha = 1.0f;

	cusparseMatDescr_t L_des, U_des, A_des;
	assert(cusparseCreateMatDescr(&L_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(&U_des) == CUSPARSE_STATUS_SUCCESS);
	assert(cusparseCreateMatDescr(&A_des) == CUSPARSE_STATUS_SUCCESS);

	cusparseSetMatType(L_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(L_des, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(L_des, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseSetMatIndexBase(L_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(U_des, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
	cusparseSetMatFillMode(U_des, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(U_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(U_des, CUSPARSE_INDEX_BASE_ZERO);

	cusparseSetMatType(A_des, CUSPARSE_MATRIX_TYPE_GENERAL);
	//cusparseSetMatFillMode(A_des, CUSPARSE_FILL_MODE_UPPER);
	//cusparseSetMatDiagType(A_des, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(A_des, CUSPARSE_INDEX_BASE_ZERO);



	int l_nnz = l_rowIndices[n];
	int u_nnz = u_rowIndices[n];

	float *d_l_val, *d_u_val;
	int *d_l_rowIndices, *d_l_indices, *d_u_rowIndices, *d_u_indices;
	cudaMalloc((void**)&d_l_val, sizeof(float)*l_nnz);
	cudaMalloc((void**)&d_l_rowIndices, sizeof(int)*(n+1));
	cudaMalloc((void**)&d_l_indices, sizeof(int)*l_nnz);
	cudaMalloc((void**)&d_u_val, sizeof(float)*u_nnz);
	cudaMalloc((void**)&d_u_rowIndices, sizeof(int)*(n+1));
	cudaMalloc((void**)&d_u_indices, sizeof(int)*u_nnz);


	cudaMemcpy(d_l_val, l_val, sizeof(float)*l_nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_l_rowIndices, l_rowIndices, sizeof(int)*(n+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_l_indices, l_indices, sizeof(int)*l_nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_val, u_val, sizeof(float)*u_nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_rowIndices, u_rowIndices, sizeof(int)*(n+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_indices, u_indices, sizeof(int)*u_nnz, cudaMemcpyHostToDevice);

	status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, l_nnz, L_des, d_l_val, d_l_rowIndices, d_l_indices, L_info);
	status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, u_nnz, U_des, d_u_val, d_u_rowIndices, d_u_indices, U_info);

	if(status != CUSPARSE_STATUS_SUCCESS){
		perr;
		printf("Failed to ANALYSIS the matrix!!!\n");
		printf("Error code %d\n", status);
		assert(false);
	}


	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = cublasSnrm2(n, d_b, 1);
	LUSolve_gpu(d_bb, d_l_val, d_l_rowIndices, d_l_indices, d_u_val, d_u_rowIndices, d_u_indices, d_b, n, &alpha, L_des, U_des, handle, status, L_info, U_info, d_temp);
	cudaThreadSynchronize();
	float normb = cublasSnrm2(n, d_bb, 1);
	if (normb == 0.0)  
		normb = 1;

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	LUSolve_gpu(d_r, d_l_val, d_l_rowIndices, d_l_indices, d_u_val, d_u_rowIndices, d_u_indices, d_rr, n, &alpha, L_des, U_des, handle, status, L_info, U_info, d_temp);
	cudaThreadSynchronize();

	float beta = cublasSnrm2(n, d_r, 1);
	if ((resid = beta / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;
		cudaFree(d_r);
		cout<<endl;
		return 0;
	}

	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *H = (float*) malloc(m*(m+1)*sizeof(float));

	float *d_v; cudaMalloc((void**) &d_v, (m+1)*n*sizeof(float));
	float *d_w; cudaMalloc((void**) &d_w, n*sizeof(float));
	float *d_ww; cudaMalloc((void**) &d_ww, n*sizeof(float));

#if CUSPARSE_FLAG
	float floatOne[1] = {1};
	float floatZero[1] = {0};
	float floatMinusOne[1] = {-1};
#endif

	while (j <= *max_iter) {

		timeSentence(cublasScopy(n, d_r, 1, d_v, 1), time_cublas);
		timeSentence(cublasSscal(n, 1.0/beta, d_v, 1), time_cublas); // XXLiu: v[0] = r * (1.0 / beta);

		vec_initial(s, 0.0, m+1); 
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {

			// XXLiu: w = M.solve(A * v[i]);
#if SUB_TIMER
			gettimeofday(&st, NULL);
#endif


#if CUSPARSE_FLAG
			status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, spm->numNZEntries, floatOne, A_des, Sparse->d_val, (const int*)Sparse->d_rowIndices, (const int*)Sparse->d_indices, d_v+i*n, floatZero, d_ww);
			assert(status == CUSPARSE_STATUS_SUCCESS);
#else
			SpMV<<<*grid, *block>>>(d_ww, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices, d_v+i*n, n, n, spm->numNZEntries);
#endif
			cudaThreadSynchronize();


#if SUB_TIMER
			gettimeofday(&et, NULL);
			time_spmv += difftime(st, et);
#endif


			LUSolve_gpu(d_w, d_l_val, d_l_rowIndices, d_l_indices, d_u_val, d_u_rowIndices, d_u_indices, d_ww, n, &alpha, L_des, U_des, handle, status, L_info, U_info, d_temp);
			cudaThreadSynchronize();


			for (k = 0; k <= i; k++) {
				timeSentence(*(H+k+i*(m+1)) = cublasSdot(n, d_w, 1, d_v+k*n, 1), time_cublas); // XXLiu: H(k, i) = dot(w, v[k]);
				timeSentence(cublasSaxpy(n, -*(H+k+i*(m+1)), d_v+k*n, 1, d_w, 1), time_cublas);// XXLiu: w -= H(k, i) * v[k];
			}
			timeSentence(*(H+(i+1)+i*(m+1)) = cublasSnrm2(n, d_w, 1), time_cublas); // XXLiu: H(i+1, i) = norm(w);
			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			timeSentence(cublasScopy(n, d_w, 1, d_v+(i+1)*n, 1), time_cublas);
			timeSentence(cublasSscal(n, 1.0/(*(H+(i+1)+i*(m+1))), d_v+(i+1)*n, 1), time_cublas);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("DEV---BREAK: %6.4e\n",resid);
				Update_GPU(d_x, i, H, m, s, d_v, n);

				*tol = resid;
				*max_iter = j;

#if SUB_TIMER
				summary_time();
#endif

				free(H);   free(s);   free(cs);   free(sn);
				cudaFree(d_v);	cudaFree(d_w);	cudaFree(d_r);
				cout<<endl;
				return 0;
			}
			cout<<"DEV---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;
		}
		Update_GPU(d_x, m-1, H, m, s, d_v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);

#if SUB_TIMER
		gettimeofday(&st, NULL);
#endif

#if CUSPARSE_FLAG
		// copy d_b -> d_ww
		cublasScopy(n, d_b, 1, d_rr, 1);
		// d_rr = (-1)*A*x + 1*b
		status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, spm->numNZEntries, floatMinusOne, A_des, Sparse->d_val, (const int*)Sparse->d_rowIndices, (const int*)Sparse->d_indices, d_x, floatOne, d_rr);
		assert(status == CUSPARSE_STATUS_SUCCESS);
#else
		sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
#endif
		cudaThreadSynchronize();

#if SUB_TIMER
		gettimeofday(&et, NULL);
		time_spmv += difftime(st, et);
#endif


		LUSolve_gpu(d_r, d_l_val, d_l_rowIndices, d_l_indices, d_u_val, d_u_rowIndices, d_u_indices, d_rr, n, &alpha, L_des, U_des, handle, status, L_info, U_info, d_temp);
		cudaThreadSynchronize();

		beta = cublasSnrm2(n, d_r, 1);

		if ((resid = beta / normb) < *tol) {

#if SUB_TIMER
			summary_time();
#endif
			*tol = resid;
			*max_iter = j;

			free(H);      free(s);      free(cs);      free(sn);
			cudaFree(d_v);      cudaFree(d_w);      cudaFree(d_r);
			cout<<endl;
			return 0;
		}
	}

#if SUB_TIMER
	summary_time();
#endif

	*tol = resid;

	cusparseDestroySolveAnalysisInfo(L_info);
	cusparseDestroySolveAnalysisInfo(U_info);
	cusparseDestroyMatDescr(L_des);
	cusparseDestroyMatDescr(U_des);

	free(H);  free(s);  free(cs);  free(sn);
	cudaFree(d_v);  cudaFree(d_w);  cudaFree(d_r);

	cout<<endl;
	return 1;
}


int
GMRES_GPU_ainv(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		//const Preconditioner &M, Matrix &H,
		const  int m, int *max_iter, float *tol, 
		cusp::precond::nonsym_bridson_ainv<float, cusp::device_memory> &ainv_M)
{

	float resid;
	int i, j = 1, k;

	float *d_r;
	cudaMalloc((void**) &d_r, n*sizeof(float));
	float *d_rr;
	cudaMalloc((void**) &d_rr, n*sizeof(float));

	float *d_bb;
	cudaMalloc((void**)& d_bb, n*sizeof(float));

	float *d_temp;
	cudaMalloc((void**)& d_temp, n*sizeof(float));


	cusp::array1d<float, cusp::device_memory> in_array;
	cusp::array1d<float, cusp::device_memory> out_array;
	in_array.resize(n);
	out_array.resize(n);
	float *pin_array = thrust::raw_pointer_cast(&in_array[0]);
	float *pout_array = thrust::raw_pointer_cast(&out_array[0]);


	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = cublasSnrm2(n, d_b, 1);

	cudaMemcpy(pin_array, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);
	ainv_M(in_array, out_array);
	cudaMemcpy(d_bb, pout_array, n * sizeof(float), cudaMemcpyDeviceToDevice);

	float normb = cublasSnrm2(n, d_bb, 1);
	if (normb == 0.0)  
		normb = 1;

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);

	cudaMemcpy(pin_array, d_rr, n * sizeof(float), cudaMemcpyDeviceToDevice);
	ainv_M(in_array, out_array);
	cudaMemcpy(d_r, pout_array, n * sizeof(float), cudaMemcpyDeviceToDevice);

	float beta = cublasSnrm2(n, d_r, 1);
	if ((resid = beta / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;
		cudaFree(d_r);
		cout<<endl;
		return 0;
	}

	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *H = (float*) malloc(m*(m+1)*sizeof(float));

	float *d_v; cudaMalloc((void**) &d_v, (m+1)*n*sizeof(float));
	float *d_w; cudaMalloc((void**) &d_w, n*sizeof(float));
	float *d_ww; cudaMalloc((void**) &d_ww, n*sizeof(float));

#if CUSPARSE_FLAG
	float floatOne[1] = {1};
	float floatZero[1] = {0};
	float floatMinusOne[1] = {-1};
#endif

	while (j <= *max_iter) {

		timeSentence(cublasScopy(n, d_r, 1, d_v, 1), time_cublas);
		timeSentence(cublasSscal(n, 1.0/beta, d_v, 1), time_cublas); // XXLiu: v[0] = r * (1.0 / beta);

		vec_initial(s, 0.0, m+1); 
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {

			// XXLiu: w = M.solve(A * v[i]);
#if SUB_TIMER
			gettimeofday(&st, NULL);
#endif


#if CUSPARSE_FLAG
			status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, spm->numNZEntries, floatOne, A_des, Sparse->d_val, (const int*)Sparse->d_rowIndices, (const int*)Sparse->d_indices, d_v+i*n, floatZero, d_ww);
			assert(status == CUSPARSE_STATUS_SUCCESS);
#else
			SpMV<<<*grid, *block>>>(d_ww, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices, d_v+i*n, n, n, spm->numNZEntries);
#endif
			cudaThreadSynchronize();


#if SUB_TIMER
			gettimeofday(&et, NULL);
			time_spmv += difftime(st, et);
#endif

			cudaMemcpy(pin_array, d_ww, n * sizeof(float), cudaMemcpyDeviceToDevice);
			ainv_M(in_array, out_array);
			cudaMemcpy(d_w, pout_array, n * sizeof(float), cudaMemcpyDeviceToDevice);



			for (k = 0; k <= i; k++) {
				timeSentence(*(H+k+i*(m+1)) = cublasSdot(n, d_w, 1, d_v+k*n, 1), time_cublas); // XXLiu: H(k, i) = dot(w, v[k]);
				timeSentence(cublasSaxpy(n, -*(H+k+i*(m+1)), d_v+k*n, 1, d_w, 1), time_cublas);// XXLiu: w -= H(k, i) * v[k];
			}
			timeSentence(*(H+(i+1)+i*(m+1)) = cublasSnrm2(n, d_w, 1), time_cublas); // XXLiu: H(i+1, i) = norm(w);
			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			timeSentence(cublasScopy(n, d_w, 1, d_v+(i+1)*n, 1), time_cublas);
			timeSentence(cublasSscal(n, 1.0/(*(H+(i+1)+i*(m+1))), d_v+(i+1)*n, 1), time_cublas);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("DEV---BREAK: %6.4e\n",resid);
				Update_GPU(d_x, i, H, m, s, d_v, n);

				*tol = resid;
				*max_iter = j;

#if SUB_TIMER
				summary_time();
#endif

				free(H);   free(s);   free(cs);   free(sn);
				cudaFree(d_v);	cudaFree(d_w);	cudaFree(d_r);
				cout<<endl;
				return 0;
			}
			cout<<"DEV---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;
		}
		Update_GPU(d_x, m-1, H, m, s, d_v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);

#if SUB_TIMER
		gettimeofday(&st, NULL);
#endif

#if CUSPARSE_FLAG
		// copy d_b -> d_ww
		cublasScopy(n, d_b, 1, d_rr, 1);
		// d_rr = (-1)*A*x + 1*b
		status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, spm->numNZEntries, floatMinusOne, A_des, Sparse->d_val, (const int*)Sparse->d_rowIndices, (const int*)Sparse->d_indices, d_x, floatOne, d_rr);
		assert(status == CUSPARSE_STATUS_SUCCESS);
#else
		sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
#endif
		cudaThreadSynchronize();

#if SUB_TIMER
		gettimeofday(&et, NULL);
		time_spmv += difftime(st, et);
#endif

		cudaMemcpy(pin_array, d_rr, n * sizeof(float), cudaMemcpyDeviceToDevice);
		ainv_M(in_array, out_array);
		cudaMemcpy(d_r, pout_array, n * sizeof(float), cudaMemcpyDeviceToDevice);

		beta = cublasSnrm2(n, d_r, 1);

		if ((resid = beta / normb) < *tol) {

#if SUB_TIMER
			summary_time();
#endif
			*tol = resid;
			*max_iter = j;

			free(H);      free(s);      free(cs);      free(sn);
			cudaFree(d_v);      cudaFree(d_w);      cudaFree(d_r);
			cout<<endl;
			return 0;
		}
	}

#if SUB_TIMER
	summary_time();
#endif

	*tol = resid;

	free(H);  free(s);  free(cs);  free(sn);
	cudaFree(d_v);  cudaFree(d_w);  cudaFree(d_r);

	cout<<endl;
	return 1;
}



int 
GMRES(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		const  int m, int *max_iter,
		float *tol, 
		Preconditioner &preconditioner)// n: rowNum, m: restart threshold
{
	float resid;
	int i, j = 1, k;

	//Vector s(m+1), cs(m+1), sn(m+1), w;
	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *ww = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *rr = (float*) malloc(n*sizeof(float));
	float *bb = (float*) malloc(n*sizeof(float));

	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = norm2(b, n);
	preconditioner.HostPrecond(b, bb);
	float normb = norm2(bb, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	preconditioner.HostPrecond(rr, r);

	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	resid = norm2(r, n) / normb;

	if ((resid = norm2(r, n) / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;

		free(s);
		free(cs);
		free(sn);
		free(w);
		free(ww);
		free(bb);
		free(H);
		free(v);

		cout<<endl;
		return 0;
	}

	while (j <= *max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(s, 0.0, m+1);
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);
			//computeSpMV(w, val, rowIndices, indices, v+i*n, n);
			computeSpMV(ww, val, rowIndices, indices, v+i*n, n);
			preconditioner.HostPrecond(ww, w);

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);
				Update(x, i, H, m, s, v, n);

				*tol = resid;
				*max_iter = j;

				free(s);
				free(cs);
				free(sn);
				free(w);
				free(ww);
				free(bb);
				free(H);
				free(v);

				cout<<endl;
				return 0;
			}
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;

		}// end of for (i = 0; i < m && j <= *max_iter; i++, j++)

		Update(x, m-1, H, m, s, v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		preconditioner.HostPrecond(rr, r);


		beta = norm2(r, n);
		if ((resid = beta / normb) < *tol) {
			*tol = resid;
			*max_iter = j;

			free(s);
			free(cs);
			free(sn);
			free(w);
			free(ww);
			free(bb);
			free(H);
			free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	*tol = resid;

	free(s);
	free(cs);
	free(sn);
	free(w);
	free(ww);
	free(bb);
	free(H);
	free(v);

	return 1;
}

int 
GMRES_tran(const float *val, const  int *rowIndices, const  int *indices,
		float *x, const float *b, const  int n,
		const  int m, const int max_iter,
		const float tol, 
		Preconditioner &preconditioner)// n: rowNum, m: restart threshold
{
	float resid;
	int i, j = 1, k;

	//Vector s(m+1), cs(m+1), sn(m+1), w;
	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *w = (float*) malloc(n*sizeof(float));
	float *ww = (float*) malloc(n*sizeof(float));
	float *r = (float*) malloc(n*sizeof(float));
	float *rr = (float*) malloc(n*sizeof(float));
	float *bb = (float*) malloc(n*sizeof(float));

	float *H = (float*) malloc(((m+1)*m)*sizeof(float));
	float *v = (float*) malloc(((m+1)*n)*sizeof(float));

	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = norm2(b, n);
	preconditioner.HostPrecond(b, bb);
	float normb = norm2(bb, n);

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
	preconditioner.HostPrecond(rr, r);

	float beta = norm2(r, n);

	if (normb == 0.0)  normb = 1;

	resid = norm2(r, n) / normb;

	if ((resid = norm2(r, n) / normb) <= tol) {

		free(s); free(cs); free(sn); free(w); free(ww);
		free(r); free(rr); free(bb); free(H); free(v);

		cout<<endl;
		return 0;
	}

	while (j <= max_iter) {
		// XXLiu: v[0] = r * (1.0 / beta);    // ??? r / beta
		sscal(v, r, 1.0/beta, n);

		vec_initial(s, 0.0, m+1);
		s[0] = beta;

		for (i = 0; i < m && j <= max_iter; i++, j++) {
			// XXLiu: w = M.solve(A * v[i]);
			//computeSpMV(w, val, rowIndices, indices, v+i*n, n);
			computeSpMV(ww, val, rowIndices, indices, v+i*n, n);
			preconditioner.HostPrecond(ww, w);

			for (k = 0; k <= i; k++) {
				*(H+k+i*(m+1)) = dot(w, v+k*n, n); // XXLiu: H(k, i) = dot(w, v[k]);
				sapxy(w, v+k*n, -(*(H+k+i*(m+1))), n); // XXLiu: w -= H(k, i) * v[k];
			}
			*(H+(i+1)+i*(m+1)) = norm2(w,n); // XXLiu: H(i+1, i) = norm(w);

			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			sscal(v+(i+1)*n, w, 1.0/(*(H+(i+1)+i*(m+1))), n);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < tol) {
				//printf("HOST---BREAK: %6.4e\n",resid);
				Update(x, i, H, m, s, v, n);

				free(s); free(cs); free(sn); free(w); free(ww);
				free(r); free(rr); free(bb); free(H); free(v);

				cout<<endl;
				return 0;
			}
			cout<<"HOST---resid: "<<scientific<<resid<<" < "<<tol<<'\r'<<flush;

		}// end of for (i = 0; i < m && j <= max_iter; i++, j++)

		Update(x, m-1, H, m, s, v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv(r, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		sgemv(rr, val, rowIndices, indices, -1.0, x, 1, b, n, n);
		preconditioner.HostPrecond(rr, r);


		beta = norm2(r, n);
		if ((resid = beta / normb) < tol) {

			free(s); free(cs); free(sn); free(w); free(ww);
			free(r); free(rr); free(bb); free(H); free(v);

			cout<<endl;
			return 0;
		}
	}// end of while(j <= *max_iter)

	free(s); free(cs); free(sn); free(w); free(ww);
	free(r); free(rr); free(bb); free(H); free(v);

	cout<<endl;
	return 1;
}


	int 
GMRES_GPU(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		const  int m, int *max_iter, float *tol, 
		Preconditioner &preconditioner)
{

	float resid;
	int i, j = 1, k;


	float *d_r;
	cudaMalloc((void**) &d_r, n*sizeof(float));
	float *d_rr;
	cudaMalloc((void**) &d_rr, n*sizeof(float));

	float *d_bb;
	cudaMalloc((void**)& d_bb, n*sizeof(float));

	float *d_temp;
	cudaMalloc((void**)& d_temp, n*sizeof(float));


	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = cublasSnrm2(n, d_b, 1);
	preconditioner.DevPrecond(d_b, d_bb);
	float normb = cublasSnrm2(n, d_bb, 1);
	if (normb == 0.0)  
		normb = 1;

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	preconditioner.DevPrecond(d_rr, d_r);

	float beta = cublasSnrm2(n, d_r, 1);
	if ((resid = beta / normb) <= *tol) {
		*tol = resid;
		*max_iter = 0;
		cudaFree(d_r);
		cout<<endl;
		return 0;
	}

	float *s = (float*) malloc((m+1)*sizeof(float));
	float *cs = (float*) malloc((m+1)*sizeof(float));
	float *sn = (float*) malloc((m+1)*sizeof(float));
	float *H = (float*) malloc(m*(m+1)*sizeof(float));

	float *d_v; cudaMalloc((void**) &d_v, (m+1)*n*sizeof(float));
	float *d_w; cudaMalloc((void**) &d_w, n*sizeof(float));
	float *d_ww; cudaMalloc((void**) &d_ww, n*sizeof(float));

	while (j <= *max_iter) {

		timeSentence(cublasScopy(n, d_r, 1, d_v, 1), time_cublas);
		timeSentence(cublasSscal(n, 1.0/beta, d_v, 1), time_cublas); // XXLiu: v[0] = r * (1.0 / beta);

		vec_initial(s, 0.0, m+1); 
		s[0] = beta;

		for (i = 0; i < m && j <= *max_iter; i++, j++) {

			// XXLiu: w = M.solve(A * v[i]);
			SpMV<<<*grid, *block>>>(d_ww, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices, d_v+i*n, n, n, spm->numNZEntries);
			cudaThreadSynchronize();
			preconditioner.DevPrecond(d_ww, d_w);


			for (k = 0; k <= i; k++) {
				timeSentence(*(H+k+i*(m+1)) = cublasSdot(n, d_w, 1, d_v+k*n, 1), time_cublas); // XXLiu: H(k, i) = dot(w, v[k]);
				timeSentence(cublasSaxpy(n, -*(H+k+i*(m+1)), d_v+k*n, 1, d_w, 1), time_cublas);// XXLiu: w -= H(k, i) * v[k];
			}
			timeSentence(*(H+(i+1)+i*(m+1)) = cublasSnrm2(n, d_w, 1), time_cublas); // XXLiu: H(i+1, i) = norm(w);
			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			timeSentence(cublasScopy(n, d_w, 1, d_v+(i+1)*n, 1), time_cublas);
			timeSentence(cublasSscal(n, 1.0/(*(H+(i+1)+i*(m+1))), d_v+(i+1)*n, 1), time_cublas);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( H+k+i*(m+1), H+(k+1)+i*(m+1), cs[k], sn[k]);

			GeneratePlaneRotation( *(H +i+i*(m+1)), *(H+(i+1)+i*(m+1)), cs+i, sn+i);
			ApplyPlaneRotation( H+i+i*(m+1), H+(i+1)+i*(m+1), cs[i], sn[i]);
			ApplyPlaneRotation( s+i, s+(i+1), cs[i], sn[i]);

			if ((resid = fabs(s[i+1]) / normb) < *tol) {
				//printf("DEV---BREAK: %6.4e\n",resid);
				Update_GPU(d_x, i, H, m, s, d_v, n);

				*tol = resid;
				*max_iter = j;

				free(H);   free(s);   free(cs);   free(sn);
				cudaFree(d_v);	cudaFree(d_w);	cudaFree(d_r);
				cout<<endl;
				return 0;
			}
			cout<<"DEV---resid: "<<scientific<<resid<<" < "<<*tol<<'\r'<<flush;
		}
		Update_GPU(d_x, m-1, H, m, s, d_v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
		sgemv_GPU(d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
		cudaThreadSynchronize();
		preconditioner.DevPrecond(d_rr, d_r);

		beta = cublasSnrm2(n, d_r, 1);


		if ((resid = beta / normb) < *tol) {

			*tol = resid;
			*max_iter = j;

			free(H);      free(s);      free(cs);      free(sn);
			cudaFree(d_v);      cudaFree(d_w);      cudaFree(d_r);
			cout<<endl;
			return 0;
		}
	}


	*tol = resid;

	free(H);  free(s);  free(cs);  free(sn);

	cudaFree(d_v);  cudaFree(d_w);  cudaFree(d_r);

	cout<<endl;
	return 1;
}



	int 
GMRES_GPU_tran(SpMatrixGPU *Sparse, SpMatrix *spm, dim3 *grid, dim3 *block,
		float *d_x, const float *d_b, const  int n,
		const  int m, const int max_iter, 
		const float tol, 
		Preconditioner &preconditioner, GMRES_GPU_Data &ggd)
{

	float resid;
	int i, j = 1, k;

	// XXLiu:  normb = norm( M.solve(b) )
	//float normb = cublasSnrm2(n, d_b, 1);
	preconditioner.DevPrecond(d_b, ggd.d_bb);
	float normb = cublasSnrm2(n, ggd.d_bb, 1);
	if (normb == 0.0)  
		normb = 1;

	// XXLiu: Vector r = M.solve(b - A * x);
	//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	sgemv_GPU(ggd.d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
	preconditioner.DevPrecond(ggd.d_rr, ggd.d_r);

	float beta = cublasSnrm2(n, ggd.d_r, 1);
	if ((resid = beta / normb) <= tol) {

		cout<<endl;
		return 0;
	}

	while (j <= max_iter) {

		timeSentence(cublasScopy(n, ggd.d_r, 1, ggd.d_v, 1), time_cublas);
		timeSentence(cublasSscal(n, 1.0/beta, ggd.d_v, 1), time_cublas); // XXLiu: v[0] = r * (1.0 / beta);

		vec_initial(ggd.s, 0.0, m+1); 
		ggd.s[0] = beta;

		for (i = 0; i < m && j <= max_iter; i++, j++) {

			// XXLiu: w = M.solve(A * v[i]);
			SpMV<<<*grid, *block>>>(ggd.d_ww, Sparse->d_val, Sparse->d_rowIndices, Sparse->d_indices, ggd.d_v+i*n, n, n, spm->numNZEntries);
			cudaThreadSynchronize();
			preconditioner.DevPrecond(ggd.d_ww, ggd.d_w);


			for (k = 0; k <= i; k++) {
				timeSentence(*(ggd.H+k+i*(m+1)) = cublasSdot(n, ggd.d_w, 1, ggd.d_v+k*n, 1), time_cublas); // XXLiu: H(k, i) = dot(w, v[k]);
				timeSentence(cublasSaxpy(n, -*(ggd.H+k+i*(m+1)), ggd.d_v+k*n, 1, ggd.d_w, 1), time_cublas);// XXLiu: w -= H(k, i) * v[k];
			}
			timeSentence(*(ggd.H+(i+1)+i*(m+1)) = cublasSnrm2(n, ggd.d_w, 1), time_cublas); // XXLiu: H(i+1, i) = norm(w);
			// XXLiu: v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)
			timeSentence(cublasScopy(n, ggd.d_w, 1, ggd.d_v+(i+1)*n, 1), time_cublas);
			timeSentence(cublasSscal(n, 1.0/(*(ggd.H+(i+1)+i*(m+1))), ggd.d_v+(i+1)*n, 1), time_cublas);

			for (k = 0; k < i; k++)
				ApplyPlaneRotation( ggd.H+k+i*(m+1), ggd.H+(k+1)+i*(m+1), ggd.cs[k], ggd.sn[k]);

			GeneratePlaneRotation( *(ggd.H +i+i*(m+1)), *(ggd.H+(i+1)+i*(m+1)), ggd.cs+i, ggd.sn+i);
			ApplyPlaneRotation( ggd.H+i+i*(m+1), ggd.H+(i+1)+i*(m+1), ggd.cs[i], ggd.sn[i]);
			ApplyPlaneRotation( ggd.s+i, ggd.s+(i+1), ggd.cs[i], ggd.sn[i]);

			if ((resid = fabs(ggd.s[i+1]) / normb) < tol) {
				//printf("DEV---BREAK: %6.4e\n",resid);
				Update_GPU(d_x, i, ggd.H, m, ggd.s, ggd.d_v, n);

				cout<<endl;
				return 0;
			}
			cout<<"DEV---resid: "<<scientific<<resid<<" < "<<tol<<'\r'<<flush;
		}
		Update_GPU(d_x, m-1, ggd.H, m, ggd.s, ggd.d_v, n);

		// XXLiu: r = M.solve(b - A * x);
		//sgemv_GPU(d_r, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
		sgemv_GPU(ggd.d_rr, Sparse, spm, grid, block, -1.0, d_x, 1.0, d_b, n, n);
		cudaThreadSynchronize();
		preconditioner.DevPrecond(ggd.d_rr, ggd.d_r);

		beta = cublasSnrm2(n, ggd.d_r, 1);


		if ((resid = beta / normb) < tol) {

			cout<<endl;
			return 0;
		}
	}

	cout<<"Failure"<<endl;
	return 1;
}



