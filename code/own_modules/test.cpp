#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void sayHello()
{
  cout << "Hello World\n";
}

int fact (int n)
{
  if (n <= 1) return 1;
    else return n*fact(n-1);
}

int showPlaatje (char* filename)
{
  char* imageName = filename;
  Mat image;
  image = imread(imageName);

  if ( !image.data )
  {
    printf("No image data \n");
    return -1;
  }
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image);

  waitKey(0);
  return 0;
}

void showGrayLena ()
{
  Mat image;
  image = imread("lena.jpg");
  Mat gray;
  cvtColor(image, gray, COLOR_BGR2GRAY);
  Mat otsu;
  threshold(gray, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image);
  namedWindow("Result Image", WINDOW_AUTOSIZE );
  imshow("Result Image", gray);
  namedWindow("Otsu Image", WINDOW_AUTOSIZE );
  imshow("Otsu Image", otsu);
  waitKey(0);
  destroyAllWindows();
}

void toOtsu (char* filename)
{
  Mat image;
  image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  Mat otsu;
  threshold(image, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  namedWindow("Original Image", WINDOW_NORMAL );
  imshow("Original Image", image);
  namedWindow("Otsu Image", WINDOW_NORMAL );
  imshow("Otsu Image", otsu);
  waitKey(0);
  destroyAllWindows();
}

void toClose (char* filename)
{
  int morph_elem = 2;
  int morph_size = 5;
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  Mat image;
  image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  Mat close;
  morphologyEx(image, close, 3, element);
  namedWindow("Original Image", WINDOW_NORMAL );
  imshow("Original Image", image);
  namedWindow("Closed Image", WINDOW_NORMAL );
  imshow("Closed Image", close);
  waitKey(0);
  destroyAllWindows();
}

void both (char* filename)
{
  Mat image;
  image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  Mat otsu;
  threshold(image, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

  int morph_elem = 2;
  int morph_size = 5;
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  Mat close;
  morphologyEx(otsu, close, 4, element);

  namedWindow("Otsu Image", WINDOW_NORMAL );
  imshow("Otsu Image", otsu);
  namedWindow("Closed Image", WINDOW_NORMAL );
  imshow("Closed Image", close);
  waitKey(0);
  destroyAllWindows();
}

void fContours (char* filename)
{
  RNG rng(12345);

  Mat image;
  image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  Mat otsu;
  threshold(image, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  int mode = CV_RETR_LIST;
  int method = CV_CHAIN_APPROX_NONE;
  //int thresh = 100;
  //Mat canny_output;

  //Canny( otsu, canny_output, thresh, thresh*2, 3 );
  findContours(otsu, contours, hierarchy, mode, method, Point(0, 0));
  ///printf("%d", contours.size());

  Mat drawing = Mat::zeros( otsu.size(), CV_8UC3 );
  for(int i = 0; i < contours.size(); i++)
     {
       Scalar color = Scalar( 255, 255, 255 );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }

  /// Show in a window
  printf("show me a window\n");
  namedWindow( "Contours", CV_WINDOW_NORMAL );
  imshow( "Contours", drawing );

  for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
  {
    printf("Length: %lu\n", it->size());
    if (it->size() < 30)
        it = contours.erase(it);
    else
        ++it;
  }

  namedWindow( "Contours updated", CV_WINDOW_NORMAL );
  imshow( "Contours updated", drawing );
  waitKey(0);
  destroyAllWindows();
}
