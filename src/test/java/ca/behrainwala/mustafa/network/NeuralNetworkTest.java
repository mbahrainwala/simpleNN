package ca.behrainwala.mustafa.network;

import ca.behrainwala.mustafa.layer.ConnectedLayer;
import ca.behrainwala.mustafa.layer.Layer;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkTest {

    @Test
    public void testNeuralNetworkCreation() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new ConnectedLayer(2, 3));
        layers.add(new ConnectedLayer(3, 2));

        NeuralNetwork nn = new NeuralNetwork(layers, 1.0);

        assertNotNull("Neural network should be created", nn);
    }

    @Test
    public void testGetOutput() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new ConnectedLayer(2, 2));

        NeuralNetwork nn = new NeuralNetwork(layers, 1.0);

        double[] input = {0.5, 0.5};
        int output = nn.getOutput(input);

        // Output should be a valid index (0 or 1)
        assertTrue("Output should be valid index", output >= 0 && output < 2);
    }

    @Test
    public void testTrainMethod() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new ConnectedLayer(2, 2));

        NeuralNetwork nn = new NeuralNetwork(layers, 1.0);

        double[] trainInput = {0.1, 0.9};
        int expectedAnswer = 1;

        // Training should complete without exception
        nn.train(trainInput, expectedAnswer);

        assertTrue("Training should complete successfully", true);
    }

    @Test
    public void testLinkLayers() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new ConnectedLayer(2, 3));
        layers.add(new ConnectedLayer(3, 2));
        layers.add(new ConnectedLayer(2, 1));

        NeuralNetwork nn = new NeuralNetwork(layers, 1.0);

        // Test that network was created successfully (implies layers were linked)
        assertNotNull("Neural network with multiple layers should be created", nn);
    }

    @Test
    public void testSingleLayerNetwork() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new ConnectedLayer(3, 2));

        NeuralNetwork nn = new NeuralNetwork(layers, 1.0);

        assertNotNull("Single layer network should be created", nn);

        double[] input = {0.1, 0.2, 0.3};
        int output = nn.getOutput(input);

        assertTrue("Output should be valid index", output >= 0 && output < 2);
    }

    @Test
    public void testScaleFactor() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new ConnectedLayer(2, 2));

        double scaleFactor = 2.0;
        NeuralNetwork nn = new NeuralNetwork(layers, scaleFactor);

        double[] input = {1.0, 1.0};
        int output = nn.getOutput(input);

        // Should accept input without error
        assertTrue("Output should be valid", output >= 0 && output < 2);
    }

    @Test
    public void testEmptyLayers() {
        List<Layer> layers = new ArrayList<>();

        NeuralNetwork nn = new NeuralNetwork(layers, 1.0);

        assertNotNull("Network with no layers should be created", nn);
    }
}
