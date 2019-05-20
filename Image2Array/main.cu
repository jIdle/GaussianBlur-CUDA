#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// Returns the size in bytes of any type of vector
template<typename T>
size_t vBytes(const typename vector<T>& v) {
	return sizeof(T)*v.size();
}

// Catches errors returned from CUDA functions
__host__
void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
		exit(EXIT_FAILURE);
	}
}

// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 2D Gaussian distribution
__host__
void generateGaussian(vector<double>& K, int dim, int radius) {
	double stdev = 5.0;
	double pi = 355.0 / 113.0;
	double constant = 1.0 / (2.0 * pi * pow(stdev, 2));

	// The following for-loops will iterate through the rows and columns
	// of the mask, assigning different values to each element depending 
	// on its 2D location in the matrix
	for (int i = -radius; i < radius + 1; ++i)
		for (int j = -radius; j < radius + 1; ++j)
			K[(i + radius) * dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
}

// CUDA kernel, it performs the image convolution
__global__
void Gaussian(double* In, double* K, double* Out, int kDim, int inWidth, int outWidth) {
	int outCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	int outRow = (blockIdx.y * blockDim.y) + threadIdx.y;

	double acc = 0; 
	for (int i = 0; i < kDim; ++i)
		for (int j = 0; j < kDim; ++j)
			acc += In[(outRow + i) * inWidth + (outCol + j)] * K[(i * kDim) + j]; // Sum of input elements multiplied by mask elements
	Out[outRow * outWidth + outCol] = acc;
}

int main() {
	vector<double> hIn, hKernel, hOut;
	int inCols, inRows;
	int kDim, kRadius;
	int outCols, outRows;
	int max = 0;
	double* dIn, * dKernel, * dOut;
	double bw = 8;

	/*
	 * Load image into OpenCV matrix and transfer to vector as linearized matrix
	*/

	Mat image = imread("Pikachu.jpg", IMREAD_GRAYSCALE);
	if (!image.data) {
		cout << "Could not open image file." << endl;
		return -1;
	}
	
	if (image.isContinuous())
		hIn.assign(image.data, image.data + image.total());
	inCols = image.cols;
	inRows = image.rows;

	/*
	 * Set mask dimensions and determine whether image and masks dimensions are compatible
	*/

	kDim = 9; // Kernel is square and odd in dimension, should be variable at some point
	if ((inRows < 2 * kDim + 1) || (inCols < 2 * kDim + 1)) {
		cout << "Image is too small to apply kernel effectively." << endl;
		return -1;
	}
	kRadius = floor(kDim / 2.0); // Radius of odd kernel doesn't consider middle index
	hKernel.resize(pow(kDim, 2), 0);
	generateGaussian(hKernel, kDim, kRadius);

	// Trim output matrix to account for kernel radius
	outCols = inCols - (2 * kRadius);
	outRows = inRows - (2 * kRadius);
	// Trim output matrix even further to match multiple of block dimensions
	// Performing this trim removes the possibility of any blocks exceeding image boundaries
	outCols = bw * floor(outCols / bw);
	outRows = bw * floor(outRows / bw);
	hOut.resize(outCols * outRows, 0);

	/*
	 * Device matrices allocation and copying
	*/

	errCatch(cudaMalloc((void**)& dIn, vBytes(hIn)));
	errCatch(cudaMemcpy(dIn, hIn.data(), vBytes(hIn), cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& dKernel, vBytes(hKernel)));
	errCatch(cudaMemcpy(dKernel, hKernel.data(), vBytes(hKernel), cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& dOut, vBytes(hOut)));
	errCatch(cudaMemcpy(dOut, hOut.data(), vBytes(hOut), cudaMemcpyHostToDevice));

	/*
	 * Kernel configuration and launch
	*/

	dim3 dimBlock(bw, bw);
	dim3 dimGrid(outCols / bw, outRows / bw);
	Gaussian <<<dimGrid, dimBlock>>> (dIn, dKernel, dOut, kDim, inCols, outCols);
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
	Mat blurImg = Mat(toInt).reshape(0, outRows);
	blurImg.convertTo(blurImg, CV_8UC1);

	/*
	 * Display blurred image
	*/

	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", image);
	namedWindow("Blurred Image", WINDOW_AUTOSIZE);
	imshow("Blurred Image", blurImg);
	waitKey(0);

	image.release();
	errCatch(cudaFree(dIn));
	errCatch(cudaFree(dKernel));
	errCatch(cudaFree(dOut));
}
