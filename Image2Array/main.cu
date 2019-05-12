// You finished making the gaussian kernel. The next part is to apply it to the matrix with the stuff you wrote down.

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

template<typename T>
size_t vBytes(const typename vector<T>& v) {
	return sizeof(T)*v.size();
}

__host__
void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
		exit(EXIT_FAILURE);
	}
}

__host__
void generateGaussian(vector<double>& K, int dim, int radius) {
	double stdev = 1.0;
	double pi = 355.0 / 113.0;
	double constant = 1.0 / (2.0 * pi * pow(stdev, 2));

	for (int i = -radius; i < radius + 1; ++i) {
		for (int j = -radius; j < radius + 1; ++j)
			K[(i + radius) * dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
	}
}

__global__
void Gaussian(double* M, double* K, double* Res, int kDim, int kRadius, int mWidth, int mHeight) {
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	if ((row - kRadius > 0) && (col - kRadius > 0) && (row + kRadius < mHeight) && (col + kRadius < mWidth)) {
		double acc = 0; 
		int rStart = row - kRadius, cStart = col - kRadius;
		for (int i = 0; i < kDim; ++i) {
			for (int j = 0; j < kDim; ++j)
				acc += M[(rStart + i) * mWidth + (cStart + j)] * K[(i * kDim) + j];
		}
		Res[row * mWidth + col] = acc;
	}
}

int main() {
	Mat image = imread("Pikachu.jpg", IMREAD_GRAYSCALE);
	if (!image.data) {
		cout << "Could not open image file." << endl;
		return -1;
	}

	int depth = image.depth();
	int channels = image.channels();
	int type = image.type();
	
	vector<double> hMatrix;
	if (image.isContinuous())
		hMatrix.assign(image.data, image.data + image.total());

	int kDim = 3; // Kernel is square and odd in dimension, should be variable at some point
	if ((image.rows < 2 * kDim + 1) || (image.cols < 2 * kDim + 1)) {
		cout << "Image is too small to apply kernel effectively." << endl;
		return -1;
	}
	int kRadius = floor(kDim / 2.0); // Radius of odd kernel doesn't consider middle index
	vector<double> hKernel(pow(kDim, 2));
	generateGaussian(hKernel, kDim, kRadius);

	//int rCols = image.cols - (2 * kRadius);
	//int rRows = image.rows - (2 * kRadius);
	//vector<double> hRes(rCols * rRows);
	vector<double> hRes(image.rows * image.cols);
	fill(hRes.begin(), hRes.end(), 0);

	// Device stuff
	double* dMatrix, * dKernel, * dRes;
	errCatch(cudaMalloc((void**)& dMatrix, vBytes(hMatrix)));
	errCatch(cudaMemcpy(dMatrix, hMatrix.data(), vBytes(hMatrix), cudaMemcpyHostToDevice));

	cudaMalloc((void**)& dKernel, vBytes(hKernel));
	cudaMemcpy(dKernel, hKernel.data(), vBytes(hKernel), cudaMemcpyHostToDevice);
	cudaMalloc((void**)& dRes, vBytes(hRes));
	cudaMemcpy(dRes, hRes.data(), vBytes(hRes), cudaMemcpyHostToDevice);

	dim3 dimBlock(8, 8);
	dim3 dimGrid(ceil(image.cols / 8.0), ceil(image.rows / 8.0));
	Gaussian <<<dimGrid, dimBlock>>> (dMatrix, dKernel, dRes, kDim, kRadius, image.cols, image.rows);
	cudaDeviceSynchronize();
	cudaMemcpy(hRes.data(), dRes, vBytes(hRes), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int max = 0;
	for (auto& value : hRes)
		max = (value > max) ? value : max;
	for (auto& value : hRes)
		value = (value * 255) / max;

	vector<int> temp(hRes.begin(), hRes.end());

	//Mat blurImg = Mat(rRows, rCols, CV_8UC1, temp.data(), sizeof(int)*(rCols));
	//Mat blurImg = Mat(image.rows, image.cols, CV_8UC1, temp.data(), sizeof(int)*(image.cols));
	Mat blurImg = Mat(temp).reshape(0, 475);
	blurImg.convertTo(blurImg, CV_8UC1);

	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", image);
	namedWindow("Blurred Image", WINDOW_AUTOSIZE);
	imshow("Blurred Image", blurImg);

	waitKey(0);
	image.release();
}
