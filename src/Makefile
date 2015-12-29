## Usage:
##	make <filename with no extension>

CC = g++

# compiler flags:
CPPFLAGS = -ggdb -std=c++11
CPPFLAGS += $(shell pkg-config --cflags opencv)

# OpenCV libraries to link:
LIBS = lib/surf.cpp
LIBS += lib/saveable_matcher.cpp
LIBS += lib/recogniser.cpp
LIBS += $(shell pkg-config --libs opencv)

% : %.cpp
	$(CC) $(CPPFLAGS) $< $(LIBS) -o $@
