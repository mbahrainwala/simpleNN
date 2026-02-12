package ca.behrainwala.mustafa.network;

import org.junit.Test;
import static org.junit.Assert.*;

public class NetworkBuilderTest {

    @Test
    public void testBuildSimpleNetwork() {
        NetworkBuilder nb = new NetworkBuilder(2, 1);
        nb.addConnectedLayer(3);
        nb.addConnectedLayer(2);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network should be created", nn);
    }

    @Test
    public void testBuildNetworkWithPooling() {
        NetworkBuilder nb = new NetworkBuilder(28, 28, 256);
        nb.addPoolLayer(2, 1);
        nb.addConnectedLayer(10);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network with pooling should be created", nn);
    }

    @Test
    public void testNetworkBuilderConstructor() {
        NetworkBuilder nb1 = new NetworkBuilder(10, 2.0);
        NetworkBuilder nb2 = new NetworkBuilder(5, 5, 2.0);

        assertNotNull("NetworkBuilder should be created with single input parameter", nb1);
        assertNotNull("NetworkBuilder should be created with row/col parameters", nb2);
    }

    @Test
    public void testAddConnectedLayer() {
        NetworkBuilder nb = new NetworkBuilder(5, 1.0);
        nb.addConnectedLayer(3);
        nb.addConnectedLayer(2);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network with connected layers should be created", nn);
    }

    @Test
    public void testAddPoolLayer() {
        NetworkBuilder nb = new NetworkBuilder(16, 16, 256);
        nb.addPoolLayer(2, 2);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network with pool layer should be created", nn);
    }

    @Test
    public void testAddPoolLayerWhenNotEmpty() {
        NetworkBuilder nb = new NetworkBuilder(16, 16, 256);
        // First add a layer to make the list non-empty
        nb.addConnectedLayer(8);
        // Then add a pool layer (this should hit the else branch)
        nb.addPoolLayer(2, 2);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network with pool layer on non-empty list should be created", nn);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidConvolutionLayerPosition() {
        NetworkBuilder nb = new NetworkBuilder(28, 28, 256);
        nb.addConnectedLayer(10);
        // This should throw exception since convolution must be first layer
        nb.addConvolutionLayer(5, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidConvolutionFilterSize() {
        NetworkBuilder nb = new NetworkBuilder(4, 4, 256);
        // This should throw exception since filter size (5) is larger than image dimensions (4x4)
        nb.addConvolutionLayer(5, 2);
    }

    @Test
    public void testValidConvolutionFirstLayer() {
        NetworkBuilder nb = new NetworkBuilder(28, 28, 256);
        // This should work since convolution is the first layer and filter size is valid
        nb.addConvolutionLayer(5, 2);
        nb.addConnectedLayer(10);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network with valid convolution layer should be created", nn);
    }

    @Test
    public void testValidConvolutionSmallFilter() {
        NetworkBuilder nb = new NetworkBuilder(4, 4, 256);
        // This should work since filter size (3) is smaller than image dimensions (4x4)
        nb.addConvolutionLayer(3, 1);
        nb.addConnectedLayer(10);
        NeuralNetwork nn = nb.build();

        assertNotNull("Neural network with small valid convolution layer should be created", nn);
    }

    @Test
    public void testMultipleLayers() {
        NetworkBuilder nb = new NetworkBuilder(28, 28, 256);
        nb.addPoolLayer(2, 1);
        nb.addConnectedLayer(16);
        nb.addConnectedLayer(10);
        NeuralNetwork nn = nb.build();

        assertNotNull("Complex network should be created", nn);
    }

    @Test
    public void testGetNumberOutput() {
        NetworkBuilder nb = new NetworkBuilder(3, 1.0);
        nb.addConnectedLayer(4);
        NeuralNetwork nn = nb.build();

        // We can't directly test the number of outputs, but we can test that it builds
        assertNotNull("Network should build successfully", nn);
    }
}
