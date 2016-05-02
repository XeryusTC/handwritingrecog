#include <opencv2/opencv.hpp>
#include <iostream>

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
