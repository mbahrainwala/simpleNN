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

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidConvolutionLayerPosition() {
        NetworkBuilder nb = new NetworkBuilder(28, 28, 256);
        nb.addConnectedLayer(10);
        // This should throw exception since convolution must be first layer
        nb.addConvolutionLayer(5, 2);
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
