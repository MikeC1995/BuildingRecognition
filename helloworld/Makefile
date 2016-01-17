# location of the Python header files

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# compile mesh classes
TARGET = hello_ext

$(TARGET): $(TARGET).cpp
	g++ -shared -o $(TARGET).so -fPIC $(TARGET).cpp -I$(PYTHON_INCLUDE) -lpython$(PYTHON_VERSION) -lboost_python
