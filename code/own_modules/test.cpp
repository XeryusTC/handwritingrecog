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
  image = imread(imageName, 1);

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
