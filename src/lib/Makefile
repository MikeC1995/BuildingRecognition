## Makefile to build shared libraries used by Python

CC = g++

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# compiler flags:
CPPFLAGS = -ggdb -std=c++11
CPPFLAGS += $(shell pkg-config --cflags opencv)

# OpenCV libraries to link:
LIBS = surf.cpp
LIBS += saveable_matcher.cpp
LIBS += $(shell pkg-config --libs opencv)

% : %.cpp
	$(CC) -shared -o /root/server/src/$@.so $(CPPFLAGS) -I$(PYTHON_INCLUDE) -fPIC $< $(LIBS) -lpython$(PYTHON_VERSION) -lboost_python

