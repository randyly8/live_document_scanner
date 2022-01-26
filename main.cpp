#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

vector<Point> getContours(Mat& image) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(img, contours, -1, Scalar(255, 0, 255), 2);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	int maxArea=0;

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		//cout << area << endl;

		string objectType;

		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

			if (area > maxArea && conPoly[i].size()==4 ) {

				//drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);
				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
			}
			//drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 2);
			//rectangle(imgOriginal, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
		}
	}
	return biggest;
}

cv::Mat preProcessImage(Mat& imageOriginal)
{
    Mat res;
    // Gray out
    cvtColor(imageOriginal, res, COLOR_BGR2GRAY);
    // Blur
    GaussianBlur(res, res, Size(7, 7), 7, 0);
    // Canny
    Canny(res, res, 50, 150);

    // create kernal for dilation
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    dilate(res, res, kernel);
    return res;
    //cv::erode(res, res, kernel);
}

// Document scnner
int main()
{
    Mat imageOriginal, imageThre;
    imageOriginal = imread("resources/paper.png" , IMREAD_COLOR);
    if(! imageOriginal.data )
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    //resize(imageOriginal, imageOriginal, cv::Size(), 0.5, 0.5);

    // Preprocessing
    imageThre = preProcessImage(imageOriginal);

    // Get Contours - Biggest rectangle
    getContours(imageThre);

    // Warp

    imshow("imageOriginal", imageOriginal);
    imshow("imageThre", imageThre);

    waitKey(0);
    return 0;
}
