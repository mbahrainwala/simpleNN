package network;

import layer.ConnectedLayer;
import layer.ConvolutionLayer;
import layer.Layer;
import layer.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.sqrt;

public class NetworkBuilder {
    private final List<Layer> layers = new ArrayList<>();

    private final int numInputs;
    private final double scaleFactor;

    private final int rows;
    private final int cols;

    public NetworkBuilder(int numInputs, double scaleFactor){
        this.numInputs = numInputs;
        this.rows = (int) sqrt(numInputs);
        this.cols = (int) sqrt(numInputs);
        this.scaleFactor = scaleFactor;
    }

    public NetworkBuilder(int rows, int cols,  double scaleFactor){
        this.numInputs = rows*cols;
        this.rows = rows;
        this.cols = cols;
        this.scaleFactor = scaleFactor;
    }

    public void addConnectedLayer(int numOutputs){
        if(layers.isEmpty()){
            layers.add(new ConnectedLayer(numInputs, numOutputs));
        } else {
            layers.add(new ConnectedLayer(layers.get(layers.size()-1).getNumberOutput(), numOutputs));
        }
    }

    public void addPoolLayer(int windowSize, int stepSize){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize, windowSize, rows, cols));
        } else {
            layers.add(new MaxPoolLayer(stepSize, windowSize
                    , layers.get(layers.size()-1).getOutputRows()
                    , layers.get(layers.size()-1).getOutputCols()));
        }
    }

    public void addConvolutionLayer(int filterSize, int stepSize){
        if(filterSize > rows || filterSize > cols)
            throw new IllegalArgumentException("Filters cannot be larger than the image.");

        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(rows, cols, filterSize, stepSize, 123));
        } else {
            throw new IllegalArgumentException("Convolution must be the first layer.");
        }
    }

    public NeuralNetwork build(){
        return new NeuralNetwork(layers, scaleFactor);
    }
}