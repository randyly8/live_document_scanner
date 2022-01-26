#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

vector<Point> getContours(Mat& imageThre, Mat& imageOriginal) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(imageThre, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
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

			if (area > maxArea && conPoly[i].size() == 4 ) {

				//drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);
				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
			}
			drawContours(imageOriginal, conPoly, i, Scalar(255, 0, 255), 2);
			rectangle(imageOriginal, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
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

void drawPoints(vector<Point> points, Scalar color, Mat& image)
{
    for (int i = 0; i < points.size(); i++)
    {
        circle(image, points[i], 5, color, FILLED);
        putText(image, to_string(i), points[i], FONT_HERSHEY_PLAIN, 2, color, 2);
    }
}

vector<Point> reorder(vector<Point>& pts)
{

    vector<Point> newpts;
	vector<int>  sumpts, subpts;

	for (int i = 0; i < 4; i++)
	{
		sumpts.push_back(pts[i].x + pts[i].y);
		subpts.push_back(pts[i].x - pts[i].y);
	}

	newpts.push_back(pts[min_element(sumpts.begin(), sumpts.end()) - sumpts.begin()]); //0
	newpts.push_back(pts[max_element(subpts.begin(), subpts.end()) - subpts.begin()]); //1
	newpts.push_back(pts[min_element(subpts.begin(), subpts.end()) - subpts.begin()]); //2
	newpts.push_back(pts[max_element(sumpts.begin(), sumpts.end()) - sumpts.begin()]); //3

	return newpts;
}

// Document scnner
int main()
{
    Mat imageOriginal, imageThre;
    vector<Point> initialpts, docPts;

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
    initialpts = getContours(imageThre, imageOriginal);
    //drawPoints(initialpts, Scalar(0, 0, 255), imageOriginal);
    docPts = reorder(initialpts);
    drawPoints(docPts, Scalar(0, 255, 0), imageOriginal);
    // Warp

    imshow("imageOriginal", imageOriginal);
    imshow("imageThre", imageThre);

    waitKey(0);
    return 0;
}
