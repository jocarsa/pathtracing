#!/bin/bash

# Exit on error
set -e

# Define file names
SOURCE_FILE="main.cpp"
OUTPUT_BINARY="simulacion"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file $SOURCE_FILE not found."
    echo "Please save the C++ code to a file named $SOURCE_FILE first."
    exit 1
fi

# Check for compiler
if command -v g++ &> /dev/null; then
    COMPILER="g++"
elif command -v clang++ &> /dev/null; then
    COMPILER="clang++"
else
    echo "Error: Neither g++ nor clang++ compiler found. Please install one of them."
    exit 1
fi

echo "Using compiler: $COMPILER"

# Check for OpenCV
if pkg-config --exists opencv4; then
    CV_FLAGS=$(pkg-config --cflags --libs opencv4)
    echo "Found OpenCV 4"
elif pkg-config --exists opencv; then
    CV_FLAGS=$(pkg-config --cflags --libs opencv)
    echo "Found OpenCV"
else
    echo "Error: OpenCV not found. Please install OpenCV development package."
    echo "On Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "On Fedora: sudo dnf install opencv-devel"
    echo "On macOS: brew install opencv"
    exit 1
fi

echo "Compiling $SOURCE_FILE..."

# Compile with OpenMP support if available
if $COMPILER -fopenmp -dM -E - < /dev/null 2>/dev/null | grep -q "OPENMP"; then
    echo "OpenMP support detected, enabling parallel rendering..."
    $COMPILER -std=c++11 -O3 -fopenmp $SOURCE_FILE -o $OUTPUT_BINARY $CV_FLAGS -lm
else
    echo "OpenMP not supported by compiler, continuing without parallel rendering..."
    $COMPILER -std=c++11 -O3 $SOURCE_FILE -o $OUTPUT_BINARY $CV_FLAGS -lm
fi

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful! Run the program with: ./$OUTPUT_BINARY"
else
    echo "Compilation failed."
    exit 1
fi

# Make the binary executable
chmod +x $OUTPUT_BINARY

echo "Done!"
