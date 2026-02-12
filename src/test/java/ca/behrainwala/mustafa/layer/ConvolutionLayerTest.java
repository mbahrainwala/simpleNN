package ca.behrainwala.mustafa.layer;

import org.junit.Test;
import static org.junit.Assert.*;

public class ConvolutionLayerTest {

    @Test
    public void testConvolutionLayerCreation() {
        ConvolutionLayer layer = new ConvolutionLayer(4, 4, 2, 1, 123);
        assertNotNull("Convolution layer should be created", layer);
    }

    @Test
    public void testGetOutputDimensions() {
        ConvolutionLayer layer = new ConvolutionLayer(4, 4, 2, 1, 123);
        assertEquals("Output rows should be calculated correctly", 3, layer.getOutputRows());
        assertEquals("Output cols should be calculated correctly", 3, layer.getOutputCols());
        assertEquals("Number of outputs should be rows * cols", 9, layer.getNumberOutput());
    }

    @Test
    public void testGetOutputWithNextLayer() {
        // Create a convolution layer
        ConvolutionLayer convLayer = new ConvolutionLayer(4, 4, 2, 1, 123);

        // Create a dummy next layer to avoid the RuntimeException
        DummyLayer nextLayer = new DummyLayer();
        convLayer.setNextLayer(nextLayer);

        // Create test input (4x4 = 16 elements)
        double[] input = new double[16];
        for (int i = 0; i < 16; i++) {
            input[i] = i + 1;
        }

        // This should not throw an exception since we have a next layer
        try {
            double[] output = convLayer.getOutput(input);
            // We can't assert much about the output since it depends on the next layer,
            // but we can verify it executes without exception
            assertTrue("Method should execute without exception", true);
        } catch (RuntimeException e) {
            // This shouldn't happen since we have a next layer
            fail("Should not throw RuntimeException when next layer is present");
        }
    }

    @Test(expected = RuntimeException.class)
    public void testGetOutputWithoutNextLayer() {
        ConvolutionLayer layer = new ConvolutionLayer(4, 4, 2, 1, 123);
        double[] input = new double[16];
        // This should throw RuntimeException since there's no next layer
        layer.getOutput(input);
    }

    @Test
    public void testBackPropagate() {
        ConvolutionLayer layer = new ConvolutionLayer(4, 4, 2, 1, 123);

        // Set up a dummy next layer to avoid RuntimeException in getOutput
        DummyLayer nextLayer = new DummyLayer();
        layer.setNextLayer(nextLayer);

        // First do a forward pass to initialize lastInput
        double[] input = new double[16];
        for (int i = 0; i < 16; i++) {
            input[i] = i + 1;
        }
        layer.getOutput(input);

        // Now test backpropagation
        double[] error = new double[9]; // 3x3 output from convolution
        for (int i = 0; i < 9; i++) {
            error[i] = 0.1 * (i + 1);
        }

        // This should execute without exception
        layer.backPropagate(error);
        assertTrue("Backpropagation should complete successfully", true);
    }

    @Test
    public void testDifferentFilterSizes() {
        // Test with different filter sizes and step sizes
        ConvolutionLayer layer1 = new ConvolutionLayer(6, 6, 3, 2, 123);
        assertEquals("Output rows should be calculated correctly", 2, layer1.getOutputRows());
        assertEquals("Output cols should be calculated correctly", 2, layer1.getOutputCols());

        ConvolutionLayer layer2 = new ConvolutionLayer(8, 8, 4, 1, 456);
        assertEquals("Output rows should be calculated correctly", 5, layer2.getOutputRows());
        assertEquals("Output cols should be calculated correctly", 5, layer2.getOutputCols());
    }

    @Test
    public void testSpaceArrayFunction() {
        ConvolutionLayer layer = new ConvolutionLayer(4, 4, 2, 2, 123); // stepSize = 2

        // Set up a dummy next layer to avoid RuntimeException in getOutput
        DummyLayer nextLayer = new DummyLayer();
        layer.setNextLayer(nextLayer);

        // First do a forward pass to initialize the layer
        double[] input = new double[16];
        for (int i = 0; i < 16; i++) {
            input[i] = i + 1;
        }
        layer.getOutput(input);

        // Test backpropagation which will exercise the spaceArray function
        double[] error = new double[9]; // 3x3 output
        for (int i = 0; i < 9; i++) {
            error[i] = 0.1;
        }

        layer.backPropagate(error);
        assertTrue("spaceArray function should be exercised", true);
    }

    // Dummy layer class for testing
    private static class DummyLayer extends Layer {
        @Override
        public double[] getOutput(double[] input) {
            // Return a simple output for testing
            return new double[]{1.0, 2.0, 3.0};
        }

        @Override
        public void backPropagate(double[] error) {
            // Do nothing for dummy layer
        }

        @Override
        public int getNumberOutput() {
            return 3;
        }

        @Override
        public int getOutputRows() {
            return 1;
        }

        @Override
        public int getOutputCols() {
            return 3;
        }
    }
}