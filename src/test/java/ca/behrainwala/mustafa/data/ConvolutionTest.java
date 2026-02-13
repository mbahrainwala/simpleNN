package ca.behrainwala.mustafa.data;

import org.junit.Test;

import static org.junit.Assert.*;

public class ConvolutionTest {

    @Test
    public void testSinglePixelConvolution() {
        double[][] image = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        double[][] kernel = {
                {0, 1},
                {1, 0}
        };
        // Position (0,0)
        double result = Convolution.singlePixelConvolution(image, 0, 0, kernel, 2, 2);
        // expected = 1*0 + 2*1 + 4*1 + 5*0 = 6
        assertEquals(6.0, result, 1e-9);
    }

    @Test
    public void testConvolution2D() {
        double[][] image = {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9,10,11,12},
                {13,14,15,16}
        };
        double[][] kernel = {
                {1,0},
                {0,1}
        };
        double[][] out = Convolution.convolution2D(image, 4, 4, kernel, 2, 2);
        double[][] expected = {
                {7, 9, 11},
                {15,17, 19},
                {23, 25, 27}
        };
        assertArrayEquals(expected, out);
    }

    @Test
    public void testConvolution2DPadded() {
        double[][] image = {
                {1,2,3},
                {4,5,6},
                {7,8,9}
        };
        double[][] kernel = {{1}}; // identity kernel
        double[][] out = Convolution.convolution2DPadded(image, 3, 3, kernel, 1, 1);
        assertArrayEquals(image, out);
    }
}
