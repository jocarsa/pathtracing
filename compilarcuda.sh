#!/bin/bash

# Name of the source file and output binary
SOURCE="main.cu"
OUTPUT="pathtracer"

# Compilation command
echo "Compiling $SOURCE with nvcc..."
nvcc "$SOURCE" -o "$OUTPUT" `pkg-config --cflags --libs opencv4` -std=c++17

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Output binary: $OUTPUT"
else
    echo "Compilation failed."
    exit 1
fi

