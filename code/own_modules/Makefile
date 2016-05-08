CC=gcc
CPP=c++
PYINCLUDE=-I/usr/include/python2.7 -I/usr/include/python2.6 -I/usr/include/python2.5 -I/usr/include/python2.4
OFLAGS=-Icocos_arnold -c -Wall -O4 -fPIC
SOFLAGS=-shared -Wall -O4 -fPIC $(PYINCLUDE)
LIBS = `pkg-config --libs opencv`
LIBS_opencv = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect

all: preprocess

preprocess: preprocess.o preprocess.i
	swig -c++ -python preprocess.i
	$(CPP) -fPIC -c preprocess.cpp preprocess_wrap.cxx $(SOFLAGS)
	$(CPP) -shared preprocess.o preprocess_wrap.o -o _preprocess.so $(LIBS_opencv)
	rm -f preprocess_wrap.o preprocess_wrap.cxx

clean:
	@rm -f *.o *.so test.py *.h.gch
	@rm -f *.pyc # Not created by this Makefile
