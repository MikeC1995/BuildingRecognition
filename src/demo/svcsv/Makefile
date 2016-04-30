## Usage:
##	make <filename with no extension>

CC = g++

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# compiler flags:
CPPFLAGS = -ggdb -std=c++11
CPPFLAGS += $(shell pkg-config --cflags opencv)

# OpenCV libraries to link:
LIBS = /root/server/src/lib/engine.cpp
LIBS += $(shell pkg-config --libs opencv)

% : %.cpp
	$(CC) -o $@ $(CPPFLAGS) $< -I$(PYTHON_INCLUDE) $(LIBS) -lpython$(PYTHON_VERSION) -lboost_python
