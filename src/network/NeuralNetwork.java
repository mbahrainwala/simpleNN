package network;

import layer.Layer;
import utils.MatrixUtils;

import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers;
    private final double scaleFactor;

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
    }

    public int getOutput(double[] input) {
        return MatrixUtils.getMaxIndex(layers.get(0).getOutput(MatrixUtils.multiplyScalar(input, 1/scaleFactor)));
    }

    public void train(double[] train, int ans) {
        double[] output = layers.get(0).getOutput(MatrixUtils.multiplyScalar(train, 1/scaleFactor));
        double[] correctResp = new double[output.length];
        correctResp[ans] = -1;
        double[] error = MatrixUtils.addArrays(output, correctResp);

        layers.get(layers.size() - 1).backPropagate(error);
    }
}