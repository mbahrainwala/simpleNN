package ca.behrainwala.mustafa.network;

import ca.behrainwala.mustafa.layer.ConnectedLayer;
import ca.behrainwala.mustafa.layer.ConvolutionLayer;
import ca.behrainwala.mustafa.layer.Layer;
import ca.behrainwala.mustafa.layer.MaxPoolLayer;
import ca.behrainwala.mustafa.layer.WordVectorGraphLayer;

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

    public void addOutputLayer(int numOutputs){
        if(layers.isEmpty()){
            layers.add(new ConnectedLayer(numInputs, numOutputs, false));
        } else {
            layers.add(new ConnectedLayer(layers.get(layers.size()-1).getNumberOutput(), numOutputs, false));
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

    /**
     * Adds a WordVectorGraph as the first layer in the network.
     * Must be called before any other layer is added.
     *
     * The WordVectorGraph layer converts raw token IDs into embedding vectors,
     * so the network handles the full pipeline: token IDs -> embeddings -> hidden -> output.
     *
     * @param tokenIds     Array of token IDs representing the full text corpus
     * @param vocabSize    Total vocabulary size
     * @param embeddingDim Number of dimensions for each word vector
     * @param contextSize  Number of tokens in the context window
     * @return The created WordVectorGraph instance (for calling printWordClusters, etc.)
     */
    public WordVectorGraphLayer addWordVectorLayer(int[] tokenIds, int vocabSize,
                                                   int embeddingDim, int contextSize) {
        if (!layers.isEmpty()) {
            throw new IllegalArgumentException("WordVectorGraph must be the first layer.");
        }
        WordVectorGraphLayer wvg = new WordVectorGraphLayer(tokenIds, vocabSize, embeddingDim,
                contextSize, scaleFactor);
        layers.add(wvg);
        return wvg;
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
        // When WordVectorGraph is the first layer, it handles scaling internally,
        // so pass scaleFactor=1 to NeuralNetwork (making its scaling a no-op).
        double networkScaleFactor = (!layers.isEmpty() && layers.get(0) instanceof WordVectorGraphLayer)
                ? 1.0 : scaleFactor;
        return new NeuralNetwork(layers, networkScaleFactor);
    }
}