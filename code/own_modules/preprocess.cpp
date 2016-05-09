#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int preprocess (char* filename)
{
    Scalar color = Scalar(255, 255, 255);

    // Load the image
    Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    Mat result;

    // Convert to otsu
    threshold(image, result, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Remove specks
    Mat temp = result.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int mode = CV_RETR_LIST;
    int method = CV_CHAIN_APPROX_NONE;

    findContours(temp, contours, hierarchy, mode, method, Point(0, 0));

    for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
    {
      if (it->size() > 30)
          it = contours.erase(it);
      else it++;
    }
    for(unsigned i = 0; i < contours.size(); i++)
      drawContours(temp, contours, i, color, 2, 8, hierarchy, 0, Point() );

    result = result + temp;

    // Write to preprocecessed file
    string newName = "tmp/preprocessed.ppm";
    imwrite(newName, result);
    return 0;
}

int showPlaatje (char* filename)
{
  char* imageName = filename;
  Mat image;
  image = imread(imageName);

  if ( !image.data )
  {
    cout << "No image data \n";
    return -1;
  }
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image);

  waitKey(0);
  return 0;
}
