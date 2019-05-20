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
	double stdev = 1.0;
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
void Gaussian(double* In, double* K, double* Out, int kDim, int kRadius, int inWidth, int inHeight, int outWidth, int outHeight) {
	int outCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	int outRow = (blockIdx.y * blockDim.y) + threadIdx.y;
	int inCol = outCol + kRadius; // Mapping from the output matrix elements to the input matrix elements
	int inRow = outRow + kRadius;

	if(outCol < outWidth && outRow < outHeight) { // Filter threads outside the range of the image
		double acc = 0; 
		int rowStart = inRow - kRadius; // Specifies top left element as the location to begin applying mask
		int colStart = inCol - kRadius;

		// Loop through mask elements
		for (int i = 0; i < kDim; ++i)
			for (int j = 0; j < kDim; ++j)
				acc += In[(rowStart + i) * inWidth + (colStart + j)] * K[(i * kDim) + j];
		Out[outRow * outWidth + outCol] = acc;
	}
}

int main() {

	/*
	 * Load image into OpenCV matrix and transfer to vector as linearized matrix
	*/

	Mat image = imread("Dude.jpg", IMREAD_GRAYSCALE);
	if (!image.data) {
		cout << "Could not open image file." << endl;
		return -1;
	}
	
	vector<double> hIn;
	if (image.isContinuous())
		hIn.assign(image.data, image.data + image.total());
	int inCols = image.cols;
	int inRows = image.rows;

	/*
	 * Set mask dimensions and determine whether image and masks dimensions are compatible
	*/

	int kDim = 21; // Kernel is square and odd in dimension, should be variable at some point
	if ((inRows < 2 * kDim + 1) || (inCols < 2 * kDim + 1)) {
		cout << "Image is too small to apply kernel effectively." << endl;
		return -1;
	}
	int kRadius = floor(kDim / 2.0); // Radius of odd kernel doesn't consider middle index
	vector<double> hKernel(pow(kDim, 2));
	generateGaussian(hKernel, kDim, kRadius); // 

	int outCols = inCols - (2 * kRadius);
	int outRows = inRows - (2 * kRadius);
	vector<double> hOut(outCols * outRows);
	fill(hOut.begin(), hOut.end(), 0);

	/*
	 * Device matrices allocation and copying
	*/

	double* dIn, * dKernel, * dOut;
	errCatch(cudaMalloc((void**)& dIn, vBytes(hIn)));
	errCatch(cudaMemcpy(dIn, hIn.data(), vBytes(hIn), cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& dKernel, vBytes(hKernel)));
	errCatch(cudaMemcpy(dKernel, hKernel.data(), vBytes(hKernel), cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& dOut, vBytes(hOut)));
	errCatch(cudaMemcpy(dOut, hOut.data(), vBytes(hOut), cudaMemcpyHostToDevice));

	/*
	 * Kernel configuration and launch
	*/

	dim3 dimBlock(8, 8);
	dim3 dimGrid(ceil(outCols / 8.0), ceil(outRows / 8.0));
	Gaussian <<<dimGrid, dimBlock>>> (dIn, dKernel, dOut, kDim, kRadius, inCols, inRows, outCols, outRows);
	errCatch(cudaDeviceSynchronize());
	errCatch(cudaMemcpy(hOut.data(), dOut, vBytes(hOut), cudaMemcpyDeviceToHost));
	errCatch(cudaDeviceSynchronize());

	/*
	 * Normalizing output matrix values
	*/

	int max = 0;
	for (auto& value : hOut)
		max = (value > max) ? value : max;
	for (auto& value : hOut)
		value = (value * 255) / max;

	/*
	 * Converting output matrix to OpenCV Mat type
	*/

	vector<int> temp(hOut.begin(), hOut.end());
	Mat blurImg = Mat(temp).reshape(0, outRows);
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
