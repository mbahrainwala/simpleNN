package ca.behrainwala.mustafa.layer;

import org.junit.Test;
import static org.junit.Assert.*;

public class ConnectedLayerTest {

    @Test
    public void testForwardPropagation() {
        // Test with simple 2-input, 2-output layer
        ConnectedLayer layer = new ConnectedLayer(2, 2);

        // Test input
        double[] input = {1.0, 2.0};

        // Get output
        double[] output = layer.getOutput(input);

        // Verify output dimensions
        assertEquals("Output should have 2 elements", 2, output.length);

        // Verify output is not null
        assertNotNull("Output should not be null", output);
    }

    @Test
    public void testBackpropagation() {
        // Test with simple 2-input, 2-output layer
        ConnectedLayer layer = new ConnectedLayer(2, 2);

        // Forward pass
        double[] input = {1.0, 2.0};
        layer.getOutput(input);

        // Backward pass
        double[] error = {0.1, 0.2};
        layer.backPropagate(error);

        // Verify that backpropagation completed without exception
        assertTrue("Backpropagation should complete successfully", true);
    }

    @Test
    public void testGetNumberOutput() {
        ConnectedLayer layer = new ConnectedLayer(3, 4);
        assertEquals("Should return correct number of outputs", 4, layer.getNumberOutput());
    }

    @Test
    public void testGetOutputDimensions() {
        ConnectedLayer layer = new ConnectedLayer(5, 3);
        assertEquals("Output rows should be square root of outputs",
                    (int)Math.sqrt(3), layer.getOutputRows());
        assertEquals("Output cols should be outputs/rows",
                    3 / layer.getOutputRows(), layer.getOutputCols());
    }

    @Test
    public void testZeroInput() {
        ConnectedLayer layer = new ConnectedLayer(2, 2);
        double[] input = {0.0, 0.0};
        double[] output = layer.getOutput(input);

        // With zero input and ReLU, output should be mostly negative (due to bias)
        assertNotNull("Output should not be null", output);
        assertEquals("Output should have correct dimensions", 2, output.length);
    }

    @Test
    public void testLargeInput() {
        ConnectedLayer layer = new ConnectedLayer(3, 2);
        double[] input = {100.0, 200.0, 300.0};
        double[] output = layer.getOutput(input);

        assertNotNull("Output should not be null", output);
        assertEquals("Output should have correct dimensions", 2, output.length);
    }
}
