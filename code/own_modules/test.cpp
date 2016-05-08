#include <opencv2/opencv.hpp>
#include <iostream>

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
