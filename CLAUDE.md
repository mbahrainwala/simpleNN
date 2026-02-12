# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Java-based neural network implementation designed as an educational example. The project demonstrates how to build neural networks from scratch, with examples for classifying handwritten digits (MNIST dataset) and other classification tasks.

## Repository Structure

- `src/main/java/ca/behrainwala/mustafa/` - Main source code
- `src/test/java/ca/behrainwala/mustafa/` - Unit tests
- `data/` - Dataset files (MNIST training/testing data, book texts, pet images)
- Root directory contains build configuration and documentation

## Architecture Overview

The neural network implementation follows a layered architecture:

1. **Core Components**:
   - `NeuralNetwork` - Main orchestrator that manages layers and training
   - `Layer` - Abstract base class for different layer types
   - `ConnectedLayer` - Fully connected neural network layer with ReLU activation
   - `NetworkBuilder` - Builder pattern for constructing neural networks

2. **Key Features**:
   - Forward propagation through connected layers
   - Backpropagation with gradient descent
   - ReLU activation function with leaky derivative
   - Support for multi-layer architectures
   - MNIST digit recognition capabilities

3. **Data Processing**:
   - `DataReader` - Reads CSV datasets
   - `DirectoryReader` - Reads image datasets from directories
   - Various utility classes for image processing and matrix operations

## Development Commands

### Build and Run
```bash
mvn compile
mvn exec:java -Dexec.mainClass="ca.behrainwala.mustafa.SimpleNN"
```

### Testing
```bash
mvn test
```

### Run Specific Tests
```bash
mvn -Dtest=NeuralNetworkTest#testGetOutput test
```

### Clean Build
```bash
mvn clean install
```

## Data Preparation

Before running MNIST examples, extract the compressed dataset files:
- Extract `data/mnist_train.7z` and `data/mnist_test.7z` to the same data folder

## Key Implementation Details

1. **Training Process**: Uses stochastic gradient descent with backpropagation
2. **Activation Function**: ReLU with leaky derivative for smoother gradients
3. **Learning Rate**: Configurable per layer (default 0.1)
4. **Bias Handling**: Small constant bias applied in activation function
5. **Scale Factor**: Input normalization handled in NeuralNetwork class

The main entry point is `SimpleNN.java` which demonstrates various classification examples including OR, XOR, and adult/child classification based on height/weight measurements.