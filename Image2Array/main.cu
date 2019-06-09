#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

#pragma once
#ifdef __INTELLISENSE__
	void __syncthreads();
#endif

#define MAX_KERNEL_WIDTH 441
__constant__ double K[MAX_KERNEL_WIDTH];

__global__ void Gaussian(double*, double*, int, int, int, int);
__host__ void generateGaussian(vector<double>&, int, int);
__host__ void errCatch(cudaError_t);
template<typename T> size_t vBytes(const typename vector<T>&);

int main() {
	vector<double> hIn, hKernel, hOut;
	double* dIn, * dOut;
	int inCols, inRows;
	int kDim, kRadius;
	int outCols, outRows;
	int max = 0;
	double bw = 8;

	/*
	 * Load image into OpenCV matrix and transfer to vector as linearized matrix
	*/

	Mat image = imread("Dude.jpg", IMREAD_GRAYSCALE);
	if (!image.data || !image.isContinuous()) {
		cout << "Could not open image file." << endl;
		exit(EXIT_FAILURE);
	}
	hIn.assign(image.data, image.data + image.total());
	inCols = image.cols;
	inRows = image.rows;
	hOut.resize(inCols * inRows, 0);

	/*
	 * Set mask dimensions and determine whether image and masks dimensions are compatible
	*/

	kDim = 5; // Kernel is square and odd in dimension, should be variable at some point
	if ((inRows < 2 * kDim + 1) || (inCols < 2 * kDim + 1)) {
		cout << "Image is too small to apply kernel effectively." << endl;
		exit(EXIT_FAILURE);
	}
	kRadius = floor(kDim / 2.0); // Radius of odd kernel doesn't consider middle index
	hKernel.resize(pow(kDim, 2), 0);
	generateGaussian(hKernel, kDim, kRadius);

	// Trim output matrix to account for kernel size
	outCols = inCols - (kDim-1);
	outRows = inRows - (kDim-1);

	/*
	 * Device matrices allocation and copying
	*/

	errCatch(cudaMalloc((void**)& dIn, vBytes(hIn)));
	errCatch(cudaMemcpy(dIn, hIn.data(), vBytes(hIn), cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& dOut, vBytes(hOut)));
	errCatch(cudaMemcpy(dOut, hOut.data(), vBytes(hOut), cudaMemcpyHostToDevice));
	errCatch(cudaMemcpyToSymbol(K, hKernel.data(), vBytes(hKernel)));

	/*
	 * Kernel configuration and launch
	*/

	int bwHalo = bw + (kDim-1); // Increase number of threads per block to account for halo cells
	dim3 dimBlock(bwHalo, bwHalo);
	dim3 dimGrid(ceil(inCols / bw), ceil(inRows / bw)); 
	Gaussian <<<dimGrid, dimBlock, bwHalo*bwHalo*sizeof(double)>>>(dIn, dOut, kDim, inCols, outCols, outRows);
	errCatch(cudaDeviceSynchronize());
	errCatch(cudaMemcpy(hOut.data(), dOut, vBytes(hOut), cudaMemcpyDeviceToHost));
	errCatch(cudaDeviceSynchronize());

	/*
	 * Normalizing output matrix values
	*/

	for (auto& value : hOut)
		max = (value > max) ? value : max;
	for (auto& value : hOut)
		value = (value * 255) / max;

	/*
	 * Converting output matrix to OpenCV Mat type
	*/

	vector<int> toInt(hOut.begin(), hOut.end()); // Converting from double to integer matrix
	Mat blurImg = Mat(toInt).reshape(0, inRows);
	blurImg.convertTo(blurImg, CV_8UC1);
	Mat cropImg = blurImg(Rect(0, 0, outCols, outRows));

	/*
	 * Display blurred image
	*/

	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", image);
	namedWindow("Cropped Image", WINDOW_AUTOSIZE);
	imshow("Cropped Image", cropImg);
	waitKey(0);

	image.release();
	errCatch(cudaFree(dIn));
	errCatch(cudaFree(dOut));

	exit(EXIT_SUCCESS);
}

// CUDA kernel, it performs the image convolution
__global__
void Gaussian(double* In, double* Out, int kDim, int inWidth, int outWidth, int outHeight) {
	extern __shared__ double loadIn[];

	// trueDim is tile dimension without halo cells
	int trueDimX = blockDim.x - (kDim-1);
	int trueDimY = blockDim.y - (kDim-1);

	// trueDim used in place of blockDim so Grid step/stride does not consider halo cells
	int col = (blockIdx.x * trueDimX) + threadIdx.x;
	int row = (blockIdx.y * trueDimY) + threadIdx.y;

	if (col < outWidth && row < outHeight) { // Filter out-of-bounds threads

		// Load input tile into shared memory for the block
		loadIn[threadIdx.y * blockDim.x + threadIdx.x] = In[row * inWidth + col];
		__syncthreads();

		if (threadIdx.y < trueDimY && threadIdx.x < trueDimX) { // Filter extra threads used for halo cells
			double acc = 0;
			for (int i = 0; i < kDim; ++i)
				for (int j = 0; j < kDim; ++j)
					acc += loadIn[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * K[(i * kDim) + j];
			Out[row * inWidth + col] = acc;
		}
	} else
		loadIn[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
}

// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 2D Gaussian distribution
__host__
void generateGaussian(vector<double> & K, int dim, int radius) {
	double stdev = 1.0;
	double pi = 355.0 / 113.0;
	double constant = 1.0 / (2.0 * pi * pow(stdev, 2));

	for (int i = -radius; i < radius + 1; ++i)
		for (int j = -radius; j < radius + 1; ++j)
			K[(i + radius) * dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
}

// Catches errors returned from CUDA functions
__host__
void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}

// Returns the size in bytes of any type of vector
template<typename T>
size_t vBytes(const typename vector<T> & v) {
	return sizeof(T)* v.size();
}
