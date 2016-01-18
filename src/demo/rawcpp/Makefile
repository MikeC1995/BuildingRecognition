## Usage:
##	make <filename with no extension>

CC = g++

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# compiler flags:
CPPFLAGS = -ggdb -std=c++11
CPPFLAGS += $(shell pkg-config --cflags opencv)

# OpenCV libraries to link:
LIBS = ~/server/src/lib/surf.cpp
LIBS += ~/server/src/lib/recogniser.cpp
LIBS += $(shell pkg-config --libs opencv)

% : %.cpp
	$(CC) $(CPPFLAGS) $< $(LIBS) ~/server/src/lib/bin/saveable_matcher.so -o $@
