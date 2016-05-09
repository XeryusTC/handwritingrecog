 %module preprocess
 %{
 /* Includes the header in the wrapper code */
 #include "preprocess.h"
 %}

 /* Parse the header file to generate wrappers */
 %include "preprocess.h"
