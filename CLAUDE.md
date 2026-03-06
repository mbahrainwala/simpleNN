# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A from-scratch Java neural network library (Java 17, Maven) used for educational examples: logic gates (OR/XOR), height/weight classification, MNIST digit recognition (OCR), pet image classification, and a minimal language model (SimpleLLM) that trains on classic literature.

## Development Commands

```bash
mvn compile                    # Compile
mvn test                       # Run all tests (JUnit 5 + JaCoCo coverage)
mvn -Dtest=ConnectedLayerTest#testForwardPass test   # Single test method
mvn clean install              # Full clean build
mvn exec:java -Dexec.mainClass="ca.behrainwala.mustafa.SimpleNN"     # Run basic examples
mvn exec:java -Dexec.mainClass="ca.behrainwala.mustafa.OCR"          # Run MNIST (requires extracted data)
mvn exec:java -Dexec.mainClass="ca.behrainwala.mustafa.SimpleLLM"    # Run language model
```

MNIST data: extract `data/mnist_train.7z` and `data/mnist_test.7z` before running OCR.

## Architecture

### Layer Pipeline

All layers extend `Layer` (abstract), which defines the contract: `getOutput()`, `backPropagate()`, dimension queries, and optional `saveWeights()`/`restoreWeights()`. Layers form a doubly-linked chain via `prevLayer`/`nextLayer` pointers set by `NeuralNetwork.linkLayers()`.

Forward pass is recursive: each layer's `getOutput()` computes its result, then calls `getNextLayer().getOutput()`. Backpropagation flows in reverse: the last layer's `backPropagate()` calls `getPrevLayer().backPropagate()`.

**Layer types:**
- `ConnectedLayer` — fully connected with ReLU activation (leaky derivative, leak=0.01, bias=0.01, lr=0.1). Weights initialized with Gaussian N(0,1) from seed 123.
- `ConvolutionLayer` — single-filter convolution. Must be the first layer. Learns filter via backprop (lr=0.01).
- `MaxPoolLayer` — max pooling with position tracking for backprop gradient routing. No trainable weights.
- `WordVectorGraphLayer` — co-occurrence-based word embeddings (not trained by gradient descent). Converts token IDs to concatenated embedding vectors. Must be the first layer when used.

### Network Construction

`NetworkBuilder` uses the builder pattern. It infers each layer's input size from the previous layer's output. `NeuralNetwork` is the top-level orchestrator: it scales inputs by `1/scaleFactor`, drives forward/backward passes, and provides `saveWeights()`/`restoreWeights()` for epoch rollback.

### Data Pipeline

- `DataReader` — reads CSV datasets (MNIST format: label in first column, pixel values following)
- `DirectoryReader` — reads images from labeled subdirectories
- `Image` — record holding `double[][] data` and `int label`
- `ImageConverter` / `EdgeDetection` / `EdgeFilter` — image preprocessing utilities
- `BookDataLoader` — loads `.txt` files from a directory, builds word frequency vocabularies, tokenizes text, finds valid training positions
- `TextToken` — bidirectional word-to-integer token mapping (token 0 = unknown, tokens assigned sequentially starting at 1)

### Entry Points

- `SimpleNN` — OR, XOR, adult/child classification demos using raw `ConnectedLayer` and `NetworkBuilder`
- `OCR` — MNIST digit recognition: MaxPool → Connected(160) → Connected(10)
- `AnimalClassifier` — pet image classification from directory of labeled images
- `SimpleLLM` — minimal language model: WordVectorGraph → Connected(256) → Connected(vocabSize), trains on classic literature with temperature-based sampling and interactive chat mode

### Key Conventions

- `MatrixUtils` provides all linear algebra: weight initialization, scalar/matrix operations, vector↔matrix conversions, argmax (`getMaxIndex`)
- Training error is computed as `output - one_hot(target)` (target index set to -1, then added to output)
- Input normalization via `scaleFactor` is applied inside `NeuralNetwork`, except when `WordVectorGraphLayer` is present (it handles scaling internally, so `NeuralNetwork` gets scaleFactor=1)
