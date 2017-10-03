#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// Forwards
void reduceIntensityLevel(Mat &srcImg, Mat &resImg);
void zoomInOut(Mat &srcImg, Mat &resImg);
void inverseAffine(Mat &srcImg, Mat &resImg);

int main()
{
	// Get image(s), Figure 2.21(a)
	Mat fig, result, result2;
	fig = imread("../fig221a.tif", CV_BGR2GRAY);

	if(fig.empty())
	{
		cout << "File is not loaded or corrupted";
	}

	//reduceIntensityLevel(fig, result);
	//zoomInOut(fig, result);
	//zoomInOut(result, result2);
	inverseAffine(fig, result);

	imshow("Original ", fig);
	imshow("Processed", result);
	//imshow("Processed 2", result2);

	waitKey(0);

	return 0;
}