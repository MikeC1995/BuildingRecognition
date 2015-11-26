## Usage:
##	make <filename with no extension>

CC = g++

# compiler flags:
CPPFLAGS = -ggdb -std=c++11
CPPFLAGS += $(shell pkg-config --cflags opencv)

# OpenCV libraries to link:
LIBS = surf.cpp
LIBS += saveable_matcher.cpp
LIBS += $(shell pkg-config --libs opencv)

% : %.cpp
	$(CC) $(CPPFLAGS) $< $(LIBS) -o $@
