//
// Created by ebubekir on 03.10.2017.
//

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// Created for question 1
void reduceIntensityLevel(Mat &srcImg, Mat &resImg)
{
	// Break, if source image is NOT 8-bit
	CV_Assert(srcImg.depth() != sizeof(uchar));

	// Ask desired intensity level to user
	cout << "Prompt desired intensity level: " << endl;
	int desiredLev;
	cin >> desiredLev;

	// Copy result image from original source
	resImg = srcImg.clone();

	for (int i = 0; i < resImg.rows - 1; ++i)
	{
		for (int j = 0; j < resImg.cols - 1; ++j)
		{
			uchar val  = srcImg.at<uchar>(i, j);
			float val2 = (1.0 * val / 255 * (desiredLev - 1));
			val2 = round((round(val2) * 255) / (desiredLev - 1));
			resImg.at<uchar>(i, j) = val2;
		}
	}
}

// Created for question 2-3
void zoomInOut(Mat &srcImg, Mat &resImg)
{
	// Break, if source image is NOT 8-bit
	CV_Assert(srcImg.depth() != sizeof(uchar));

	// Ask desired intensity level to user
	cout << "Prompt desired zoom factor: " << endl;
	float zoomFactor;
	cin >> zoomFactor;

	int interpolation;

	cout << "Promt number for interpolation method:" << endl;
	cout << "1-) Pixel replication method" << endl;
	cout << "2-) Bi-linear interpolation method  " << endl;
	cout << "3-) Bi-cubic interpolation method   " << endl;

	cin >> interpolation;

	// Calculate required dimesion by given zoom factor
	float xDim = srcImg.rows * zoomFactor;
	float yDim = srcImg.cols * zoomFactor;

	// Create image with given dimensions
	resImg.create((int) xDim, (int) yDim, CV_8U);

	// These variables will be used to measure distance
	float a, b;

	switch (interpolation)
	{
		case 1 :
			// Pixel replication algorithm
			for (int m = 0; m < resImg.rows; ++m)
			{
				for (int n = 0; n < resImg.cols; ++n)
				{
					float mm = 1.0f * m / zoomFactor;
					float nn = 1.0f * n / zoomFactor;

					mm += 0.5f;
					nn += 0.5f;

					resImg.at<uchar>(m, n) = srcImg.at<uchar>((int) mm, (int) nn);
				}
			}
			break;

		case 2 :
			// Bilinear interpolation algoritm
			for (int m = 0; m < resImg.rows; ++m)
			{
				for (int n = 0; n < resImg.cols; ++n)
				{
					float mm = 1.0f * m / zoomFactor;
					float nn = 1.0f * n / zoomFactor;

					a = nn - (int) nn;
					b = mm - (int) mm;

					mm = floor(mm);
					nn = floor(nn);

					uchar su  = srcImg.at<uchar>(mm, nn);
					uchar sa  = srcImg.at<uchar>(mm + 1, nn);
					uchar ssu = srcImg.at<uchar>(mm, nn + 1);
					uchar ssa = srcImg.at<uchar>(mm + 1, nn + 1);

					float vall = (1 - a) * (1 - b) * su +
								 a * (1 - b) * ssu +
								 (1 - a) * b * sa +
								 a * b * ssa;

					resImg.at<uchar>(m, n) = (uchar) vall;
				}
			}
			break;
		case 3:

			// Bicubic interpolation algoritm, mm -- nn is input, m -- n is output positions
			Mat F(resImg.rows, srcImg.cols, CV_8U);

			for (int i = 0; i <= resImg.rows-2; ++i)
			{
				for (int j = 0; j <= srcImg.cols; ++j)
				{
					float fmm = 1.0f * i / zoomFactor;

					b = fmm - (int)fmm;

					fmm = floor(fmm);

					if(fmm == 0) continue;

					uchar su  = srcImg.at<uchar>(fmm - 1, j);
					uchar sa  = srcImg.at<uchar>(fmm, j);
					uchar ssu = srcImg.at<uchar>(fmm + 1, j);
					uchar ssa = srcImg.at<uchar>(fmm + 2, j);

					float fVall = -b *(1-b) * (1-b) * su +
								  (1 - 2*b*b + b*b*b) * sa +
								  b * (1+b-b*b) * ssu -
								  (b*b *(1-b) * ssa);


					F.at<uchar>(i, j) = (uchar)fVall;
				}
			}

			for (int m = 0; m < resImg.rows; ++m)
			{
				for (int n = 0; n < resImg.cols; ++n)
				{
					float nn = 1.0f * n / zoomFactor;

					a = nn - (int)nn;

					nn = floor(nn);

					if(nn == 0) continue;

					uchar su  = F.at<uchar>(m , nn -1);
					uchar sa  = F.at<uchar>(m, nn);
					uchar ssu = F.at<uchar>(m, nn + 1);
					uchar ssa = F.at<uchar>(m, nn + 2);

					float vall = (-a * (1-a) * (1-a)) * su  +
							     (1- 2*a*a + a*a*a)   * sa  +
							     (a * (1+a-a*a))      * ssu -
							     (a*a * (1-a))        * ssa;


					resImg.at<uchar>(m, n) = (uchar) vall;
				}
			}


			break;
	}
}

// Affine transforms
void inverseAffine(Mat &srcImg, Mat &resImg)
{
	float t[3][3] = {{0.866f, 0.5f  , 0.0f},
					{-0.5f   , 0.866f, 0.0f},
					{0.0f   , 0.0f  , 1.0f}};

	float tttt[3][3] = {{2.0f, 0.0f  , 0.0f},
					 {0.0f   , 2.0f, 0.0f},
					 {0.0f   , 0.0f  , 1.0f}};

	Mat T(3,3, CV_32FC1, t);

	resImg.create(srcImg.rows,srcImg.cols, CV_8U);

	for (int i = 0; i < resImg.rows; ++i)
	{
		for (int j = 0; j < resImg.cols; ++j)
		{
			float out[1][3] = {i, j, 1};
			Mat pos(1,3,CV_32F, out);

			Mat res = pos * T.inv();

			int row = res.at<float>(0,0);
			int col = res.at<float>(0,1);
			resImg.at<uchar>(i, j) = srcImg.at<uchar>(row, col);
		}
	}
}
