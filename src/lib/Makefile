## Makefile to build shared libraries used by Python

CC = g++

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# compiler flags:
CPPFLAGS = -std=c++11 -fopenmp
CPPFLAGS += $(shell pkg-config --cflags opencv)

# OpenCV libraries to link:
LIBS = engine.cpp
LIBS += saveable_matcher.cpp
LIBS += $(shell pkg-config --libs opencv)

% : %.cpp
	$(CC) $< -shared -o /root/server/bin/$@.so $(CPPFLAGS) -I $(PYTHON_INCLUDE) -I $(BOOST_ROOT) -fPIC $(LIBS) -lpython$(PYTHON_VERSION) -lboost_python -lboost_system -lboost_filesystem
