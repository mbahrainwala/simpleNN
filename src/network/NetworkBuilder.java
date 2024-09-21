package network;

import layer.ConnectedLayer;
import layer.Layer;
import layer.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    private final List<Layer> layers = new ArrayList<>();

    private final int numInputs;
    private final double scaleFactor;

    public NetworkBuilder(int numInputs, double scaleFactor){
        this.numInputs = numInputs;
        this.scaleFactor = scaleFactor;
    }

    public void addConnectedLayer(int numOutputs){
        if(layers.isEmpty()){
            layers.add(new ConnectedLayer(numInputs, numOutputs));
        } else {
            layers.add(new ConnectedLayer(layers.get(layers.size()-1).getNumberOutput(), numOutputs));
        }
    }

    public void addPoolLayer(int windowSize, int stepSize, int rows, int cols){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize, windowSize, rows, cols));
        } else {
            throw new IllegalArgumentException("Cannot add a pooling layer after a connected layer");
        }
    }

    public NeuralNetwork build(){
        return new NeuralNetwork(layers, scaleFactor);
    }
}