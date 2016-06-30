# handwritingrecog
IMPORTANT:
  Python version: 2.x (2.7 if possible)
	Opencv version: 3.x
  Python-scikit: 0.17x

Installation
============
Installation is very easy, all that is needed is Python 2.7.11+ and virtualenv
(optional). Creating a virtualenv is very easy and the instructions can be
found online. OpenCV 3.x (3.1.0 or above recommended) also needs to be
installed, this can usually be done via your package manager or you can
retrieve it from the OpenCV website. When using a virtualenv then you need to
follow the instructions on how to install [on this website](https://medium.com/@manuganji/installation-of-opencv-numpy-scipy-inside-a-virtualenv-bf4d82220313#.b86rahc31).
Finally you need to execute the command` pip install -r requirements.txt` to
install the other necessary components. This will automatically install the
correct versions of all other dependencies.

Running
=======
All necessary files to run the code can be found in code/
First-timers first need to run the train command as following:
"python train.py --create_segments"

In the future, the "--create_segments" is no longer required.

After this, the recognizer can be run with
"python recognizer.py image.ppm input.words out.words"
