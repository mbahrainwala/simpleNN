package ca.behrainwala.mustafa.network;

import ca.behrainwala.mustafa.layer.ConnectedLayer;
import ca.behrainwala.mustafa.layer.Layer;
import ca.behrainwala.mustafa.layer.WordVectorGraphLayer;
import ca.behrainwala.mustafa.utils.MatrixUtils;

import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers;
    private final double scaleFactor;

    /** Pre-allocated error array for train(), avoids 3 allocations per training step */
    private double[] trainError;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {
        if(layers.size() < 2)
            return;

        for (int i = 0; i < layers.size()-1; i++) {
            if(i>0)
                layers.get(i).setPrevLayer(layers.get(i-1));

            layers.get(i).setNextLayer(layers.get(i+1));
        }
        layers.get(layers.size()-1).setPrevLayer(layers.get(layers.size()-2));
    }

    private double[] scaleInput(double[] input) {
        return scaleFactor == 1.0 ? input : MatrixUtils.multiplyScalar(input, 1/scaleFactor);
    }

    public int getOutput(double[] input) {
        return MatrixUtils.getMaxIndex(layers.get(0).getOutput(scaleInput(input)));
    }

    public double[] getOutputArray(double[] input) {
        return layers.get(0).getOutput(scaleInput(input));
    }

    public int train(double[] train, int ans) {
        double[] output = layers.get(0).getOutput(scaleInput(train));

        if (trainError == null || trainError.length != output.length) {
            trainError = new double[output.length];
        }

        // Softmax cross-entropy gradient: softmax(output) - one_hot(target).
        // Softmax normalizes raw outputs into probabilities [0,1] summing to 1,
        // giving bounded, well-targeted gradients for multi-class classification.
        // Without softmax, raw outputs grow unbounded and gradients explode.
        double maxVal = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxVal) maxVal = output[i];
        }
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            trainError[i] = Math.exp(output[i] - maxVal);
            sum += trainError[i];
        }
        double invSum = 1.0 / sum;
        for (int i = 0; i < output.length; i++) {
            trainError[i] *= invSum;
        }
        trainError[ans] -= 1.0;

        layers.get(layers.size() - 1).backPropagate(trainError);
        return MatrixUtils.getMaxIndex(output);
    }

    /**
     * Multiplies all ConnectedLayer learning rates by the given factor.
     * Preserves the ratio between layers (e.g., hidden=0.1, output=0.05).
     */
    public void scaleLearningRate(double factor) {
        for (Layer layer : layers) {
            if (layer instanceof ConnectedLayer cl) {
                cl.setLearningRate(cl.getLearningRate() * factor);
            }
        }
    }

    /**
     * Enables the generation cache on the WordVectorGraphLayer (if present).
     * Call before autoregressive generation to avoid redundant embedding lookups.
     */
    public void enableGenerationCache() {
        if (!layers.isEmpty() && layers.get(0) instanceof WordVectorGraphLayer wvg) {
            wvg.enableGenerationCache();
        }
    }

    /**
     * Disables the generation cache. Call after generation is complete.
     */
    public void disableGenerationCache() {
        if (!layers.isEmpty() && layers.get(0) instanceof WordVectorGraphLayer wvg) {
            wvg.disableGenerationCache();
        }
    }

    public void saveWeights() {
        for (Layer layer : layers) {
            layer.saveWeights();
        }
    }

    public void restoreWeights() {
        for (Layer layer : layers) {
            layer.restoreWeights();
        }
    }
}