package ca.behrainwala.mustafa.utils;

import org.junit.Test;
import static org.junit.Assert.*;

public class MatrixUtilsTest {

    @Test
    public void testInitializeWeights() {
        // Test with valid dimensions
        double[][] weights = MatrixUtils.initializeWeights(2, 3);

        // Check dimensions
        assertEquals("Rows should match", 2, weights.length);
        assertEquals("Columns should match", 3, weights[0].length);

        // Check that weights are initialized (not all zeros)
        boolean allZeros = true;
        for (double[] weight : weights) {
            for (int j = 0; j < weights[0].length; j++) {
                if (weight[j] != 0.0) {
                    allZeros = false;
                    break;
                }
            }
            if (!allZeros) break;
        }
        assertFalse("Weights should not all be zero", allZeros);

        // Test with single row/column
        double[][] singleWeights = MatrixUtils.initializeWeights(1, 1);
        assertEquals("Single row", 1, singleWeights.length);
        assertEquals("Single column", 1, singleWeights[0].length);
    }

    @Test
    public void testGetMaxIndex() {
        // Test with positive values
        double[] input1 = {0.1, 0.5, 0.3, 0.2};
        assertEquals("Max index should be 1", 1, MatrixUtils.getMaxIndex(input1));

        // Test with negative values
        double[] input2 = {-0.5, -0.1, -0.3};
        assertEquals("Max index should be 1", 1, MatrixUtils.getMaxIndex(input2));

        // Test with all equal values
        double[] input3 = {0.5, 0.5, 0.5};
        assertEquals("Max index should be 0 (first occurrence)", 0, MatrixUtils.getMaxIndex(input3));

        // Test with single element
        double[] input4 = {0.7};
        assertEquals("Max index should be 0", 0, MatrixUtils.getMaxIndex(input4));

        // Test with zero values
        double[] input5 = {0.0, 0.0, 0.0};
        assertEquals("Max index should be 0", 0, MatrixUtils.getMaxIndex(input5));
    }

    @Test
    public void testMultiplyScalarDoubleArray() {
        double[] input = {1.0, 2.0, 3.0};
        double scalar = 2.0;

        double[] result = MatrixUtils.multiplyScalar(input, scalar);

        // Check dimensions
        assertEquals("Result length should match input", input.length, result.length);

        // Check values
        assertArrayEquals("Values should be multiplied",
                         new double[]{2.0, 4.0, 6.0}, result, 0.001);

        // Test with negative scalar
        double[] result2 = MatrixUtils.multiplyScalar(input, -1.0);
        assertArrayEquals("Negative multiplication should work",
                         new double[]{-1.0, -2.0, -3.0}, result2, 0.001);

        // Test with zero scalar
        double[] result3 = MatrixUtils.multiplyScalar(input, 0.0);
        assertArrayEquals("Zero multiplication should work",
                         new double[]{0.0, 0.0, 0.0}, result3, 0.001);

        // Test with single element
        double[] single = {5.0};
        double[] singleResult = MatrixUtils.multiplyScalar(single, 2.0);
        assertArrayEquals("Single element should work",
                         new double[]{10.0}, singleResult, 0.001);
    }

    @Test
    public void testMultiplyScalarDoubleArrayArray() {
        double[][] input = {{1.0, 2.0}, {3.0, 4.0}};
        double scalar = 2.0;

        double[][] result = MatrixUtils.multiplyScalar(input, scalar);

        // Check dimensions
        assertEquals("Rows should match", input.length, result.length);
        assertEquals("Columns should match", input[0].length, result[0].length);

        // Check values
        assertEquals("Value at [0][0] should be correct", 2.0, result[0][0], 0.001);
        assertEquals("Value at [0][1] should be correct", 4.0, result[0][1], 0.001);
        assertEquals("Value at [1][0] should be correct", 6.0, result[1][0], 0.001);
        assertEquals("Value at [1][1] should be correct", 8.0, result[1][1], 0.001);

        // Test with negative scalar
        double[][] result2 = MatrixUtils.multiplyScalar(input, -1.0);
        assertEquals("Negative multiplication should work", -1.0, result2[0][0], 0.001);
        assertEquals("Negative multiplication should work", -2.0, result2[0][1], 0.001);
        assertEquals("Negative multiplication should work", -3.0, result2[1][0], 0.001);
        assertEquals("Negative multiplication should work", -4.0, result2[1][1], 0.001);

        // Test with zero scalar
        double[][] result3 = MatrixUtils.multiplyScalar(input, 0.0);
        assertEquals("Zero multiplication should work", 0.0, result3[0][0], 0.001);
        assertEquals("Zero multiplication should work", 0.0, result3[0][1], 0.001);
        assertEquals("Zero multiplication should work", 0.0, result3[1][0], 0.001);
        assertEquals("Zero multiplication should work", 0.0, result3[1][1], 0.001);
    }

    @Test
    public void testAddArrays() {
        double[] arr1 = {1.0, 2.0, 3.0};
        double[] arr2 = {4.0, 5.0, 6.0};

        double[] result = MatrixUtils.addArrays(arr1, arr2);

        // Check dimensions
        assertEquals("Result length should match inputs", arr1.length, result.length);
        assertEquals("Result length should match inputs", arr2.length, result.length);

        // Check values
        assertArrayEquals("Arrays should be added correctly",
                         new double[]{5.0, 7.0, 9.0}, result, 0.001);

        // Test with negative values
        double[] negArr1 = {-1.0, -2.0};
        double[] negArr2 = {3.0, -4.0};
        double[] negResult = MatrixUtils.addArrays(negArr1, negArr2);
        assertArrayEquals("Negative values should work",
                         new double[]{2.0, -6.0}, negResult, 0.001);

        // Test with zeros
        double[] zeroArr1 = {0.0, 0.0};
        double[] zeroArr2 = {0.0, 0.0};
        double[] zeroResult = MatrixUtils.addArrays(zeroArr1, zeroArr2);
        assertArrayEquals("Zeros should work",
                         new double[]{0.0, 0.0}, zeroResult, 0.001);
    }

    @Test
    public void testVectorToMatrix() {
        double[] input = {1.0, 2.0, 3.0, 4.0};
        int rows = 2;
        int cols = 2;

        double[][] result = MatrixUtils.vectorToMatrix(input, rows, cols);

        // Check dimensions
        assertEquals("Rows should match", rows, result.length);
        assertEquals("Columns should match", cols, result[0].length);

        // Check values
        assertEquals("Value at [0][0] should be correct", 1.0, result[0][0], 0.001);
        assertEquals("Value at [0][1] should be correct", 2.0, result[0][1], 0.001);
        assertEquals("Value at [1][0] should be correct", 3.0, result[1][0], 0.001);
        assertEquals("Value at [1][1] should be correct", 4.0, result[1][1], 0.001);

        // Test with different dimensions
        double[] input2 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[][] result2 = MatrixUtils.vectorToMatrix(input2, 3, 2);
        assertEquals("3x2 matrix should work", 3, result2.length);
        assertEquals("3x2 matrix should work", 2, result2[0].length);
        assertEquals("Value at [2][1] should be correct", 6.0, result2[2][1], 0.001);
    }

    @Test
    public void testMatrixToVector() {
        double[][] input = {{1.0, 2.0}, {3.0, 4.0}};

        double[] result = MatrixUtils.matrixToVector(input);

        // Check dimensions
        assertEquals("Vector length should match matrix elements", 4, result.length);

        // Check values (row-major order)
        assertArrayEquals("Matrix should be converted to vector correctly",
                         new double[]{1.0, 2.0, 3.0, 4.0}, result, 0.001);

        // Test with different dimensions
        double[][] input2 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        double[] result2 = MatrixUtils.matrixToVector(input2);
        assertArrayEquals("Different dimensions should work",
                         new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, result2, 0.001);
    }

    @Test
    public void testAddMatrices() {
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] b = {{5.0, 6.0}, {7.0, 8.0}};

        double[][] result = MatrixUtils.add(a, b);

        // Check dimensions
        assertEquals("Rows should match", a.length, result.length);
        assertEquals("Columns should match", a[0].length, result[0].length);

        // Check values
        assertEquals("Value at [0][0] should be correct", 6.0, result[0][0], 0.001);
        assertEquals("Value at [0][1] should be correct", 8.0, result[0][1], 0.001);
        assertEquals("Value at [1][0] should be correct", 10.0, result[1][0], 0.001);
        assertEquals("Value at [1][1] should be correct", 12.0, result[1][1], 0.001);

        // Test with negative values
        double[][] negA = {{-1.0, -2.0}, {-3.0, -4.0}};
        double[][] negB = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] negResult = MatrixUtils.add(negA, negB);
        assertEquals("Negative addition should work", 0.0, negResult[0][0], 0.001);
        assertEquals("Negative addition should work", 0.0, negResult[0][1], 0.001);
        assertEquals("Negative addition should work", 0.0, negResult[1][0], 0.001);
        assertEquals("Negative addition should work", 0.0, negResult[1][1], 0.001);

        // Test with zeros
        double[][] zeroA = {{0.0, 0.0}, {0.0, 0.0}};
        double[][] zeroB = {{0.0, 0.0}, {0.0, 0.0}};
        double[][] zeroResult = MatrixUtils.add(zeroA, zeroB);
        assertEquals("Zero addition should work", 0.0, zeroResult[0][0], 0.001);
        assertEquals("Zero addition should work", 0.0, zeroResult[0][1], 0.001);
        assertEquals("Zero addition should work", 0.0, zeroResult[1][0], 0.001);
        assertEquals("Zero addition should work", 0.0, zeroResult[1][1], 0.001);
    }
}