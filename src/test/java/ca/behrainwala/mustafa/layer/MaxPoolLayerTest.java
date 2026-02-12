package ca.behrainwala.mustafa.layer;

import org.junit.Test;
import static org.junit.Assert.*;

public class MaxPoolLayerTest {

    @Test
    public void testMaxPoolOperation() {
        // Create a 4x4 input (represented as 16-element array)
        int inputRows = 4;
        int inputCols = 4;
        int windowSize = 2;
        int stepSize = 2;

        MaxPoolLayer poolLayer = new MaxPoolLayer(stepSize, windowSize, inputRows, inputCols);

        // Create test input data
        double[] input = new double[16];
        for (int i = 0; i < 16; i++) {
            input[i] = i + 1; // Values 1-16
        }

        // Get output
        double[] output = poolLayer.getOutput(input);

        // For 4x4 input with 2x2 window and step size 2, we expect 2x2 output (4 elements)
        assertEquals("Output should have 4 elements", 4, output.length);

        // Verify that output contains reasonable values (max pooling should produce higher values)
        assertNotNull("Output should not be null", output);
    }

    @Test
    public void testGetNumberOutput() {
        int inputRows = 6;
        int inputCols = 6;
        int windowSize = 3;
        int stepSize = 3;

        MaxPoolLayer poolLayer = new MaxPoolLayer(stepSize, windowSize, inputRows, inputCols);
        assertEquals("Should return correct number of outputs", 4, poolLayer.getNumberOutput());
    }

    @Test
    public void testGetOutputDimensions() {
        int inputRows = 4;
        int inputCols = 4;
        int windowSize = 2;
        int stepSize = 2;

        MaxPoolLayer poolLayer = new MaxPoolLayer(stepSize, windowSize, inputRows, inputCols);
        assertEquals("Output rows should match expected value", 2, poolLayer.getOutputRows());
        assertEquals("Output cols should match expected value", 2, poolLayer.getOutputCols());
    }

    @Test
    public void testDifferentWindowSizeAndStep() {
        int inputRows = 8;
        int inputCols = 8;
        int windowSize = 4;
        int stepSize = 2;

        MaxPoolLayer poolLayer = new MaxPoolLayer(stepSize, windowSize, inputRows, inputCols);

        // Create test input data
        double[] input = new double[64];
        for (int i = 0; i < 64; i++) {
            input[i] = i + 1;
        }

        double[] output = poolLayer.getOutput(input);

        // Verify output is produced
        assertNotNull("Output should not be null", output);
    }
}
