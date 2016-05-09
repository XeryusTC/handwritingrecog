#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int preprocess (char* filename)
{
    // Load the image
    Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    Mat result;
    // Convert to otsu
    threshold(image, result, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Other steps
    //...

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

void fContours (char* filename)
{
  RNG rng(12345);
  Scalar color = Scalar( 255, 255, 255 );

  Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  Mat otsu;
  threshold(image, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  int mode = CV_RETR_LIST;
  int method = CV_CHAIN_APPROX_NONE;

  findContours(otsu, contours, hierarchy, mode, method, Point(0, 0));

  Mat drawing = Mat::zeros( otsu.size(), CV_8UC3 );
  for(int i = 0; i < contours.size(); i++)
     {
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }

  namedWindow( "Contours", CV_WINDOW_NORMAL );
  imshow( "Contours", drawing );

  for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
  {
    if (it->size() < 30)
    {
        it = contours.erase(it);
    }
    else
        ++it;
  }

  Mat drawing2 = Mat::zeros( otsu.size(), CV_8UC3 );
  for(int i = 0; i < contours.size(); i++)
     {
       drawContours( drawing2, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }

  namedWindow( "Contours updated", CV_WINDOW_NORMAL );
  imshow( "Contours updated", drawing2 );
  waitKey(0);
  destroyAllWindows();
}
