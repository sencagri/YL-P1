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
void imageRegistration(Mat &srcImg, Mat &resImg);


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		vector<Point>* ptPtr = (vector<Point>*)userdata;
		ptPtr->push_back(Point(x,y));
	}
}

void CallBackFunc2(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		vector<Point>* ptPtr = (vector<Point>*)userdata;
		ptPtr->push_back(Point(x,y));
	}
}

int main()
{
	// Get image(s), Figure 2.21(a)
	Mat fig, result, result2;
	fig = imread("../fig236a.tif", CV_BGR2GRAY);

	if(fig.empty())
	{
		cout << "File is not loaded or corrupted";
	}


	//set the callback function for any mouse event


	//reduceIntensityLevel(fig, result);
	//zoomInOut(fig, result);
	//inverseAffine(fig, result);


	/*
	 * IMAGE REGISTRATION AREA
	 * */

	std::vector<Point> pointsOrg;
	std::vector<Point> pointsPro;

	imageRegistration(fig, result);

	namedWindow("Original");
	setMouseCallback("Original", CallBackFunc, (void *)&pointsOrg);


	while(pointsOrg.size() < 4)
	{
		imshow("Original", fig);

		waitKey(1);
	}



	///destroyWindow("Original");

	namedWindow("Processed");
	setMouseCallback("Processed", CallBackFunc2, (void *)&pointsPro);

	while(pointsPro.size() < 4)
	{
		imshow("Processed", result);
		waitKey(1);
	}

	// all control points are saved

	Mat_<float>kern1(4,1);
	kern1 << pointsPro[0].x, pointsPro[1].x, pointsPro[2].x, pointsPro[3].x;
	Mat X = kern1;

	Mat_<float>kern2(4,1);
	kern2 << pointsPro[0].y, pointsPro[1].y, pointsPro[2].y, pointsPro[3].y;
	Mat Y = kern2;


	Mat_<float>kernel(4,4);

	kernel << 		 pointsOrg[0].x, pointsOrg[0].y, pointsOrg[0].x * pointsOrg[0].y, 1,
					 pointsOrg[1].x, pointsOrg[1].y, pointsOrg[1].x * pointsOrg[1].y, 1,
					 pointsOrg[2].x, pointsOrg[2].y, pointsOrg[2].x * pointsOrg[2].y, 1,
					 pointsOrg[3].x, pointsOrg[3].y, pointsOrg[3].x * pointsOrg[3].y, 1;
	Mat C = kernel;

	Mat XRES;
	Mat YRES;

	XRES = C.inv() * X;
	YRES = C.inv() * Y;


	cout << XRES << " ";
	cout << YRES << endl;

	waitKey(0);

	return 0;
}