package ca.behrainwala.mustafa.data;

import org.junit.Test;
import static org.junit.Assert.*;

public class ImageTest {
    @Test
    public void testImageCreation(){
        // Create test data
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };

        int label = 5;

        Image image = new Image(testData, label);

        assertNotNull("Image should be created", image);
        assertEquals("Label should match", label, image.label());
        assertArrayEquals("Data should match", testData, image.data());
    }

    @Test
    public void testImageToString() {
        // Create test data
        double[][] testData = {
            {0.0, 200.0, 0.0},
            {150.0, 0.0, 100.0},
            {0.0, 50.0, 0.0}
        };

        Image image = new Image(testData, 1);
        String result = image.toString();

        assertNotNull("toString should return non-null string", result);
        assertFalse("toString should return non-empty string", result.isEmpty());
        assertTrue("toString should contain label", result.contains("label->1"));
    }

    @Test
    public void testImageWithZeroData() {
        // Create test data with all zeros
        double[][] testData = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}
        };

        Image image = new Image(testData, 0);

        assertNotNull("Image should be created with zero data", image);
        assertEquals("Label should match", 0, image.label());
        assertArrayEquals("Data should match", testData, image.data());
    }

    @Test
    public void testImageWithLargeValues() {
        // Create test data with large values
        double[][] testData = {
            {255.0, 200.0, 150.0},
            {100.0, 50.0, 0.0}
        };

        Image image = new Image(testData, 9);

        assertNotNull("Image should be created with large values", image);
        assertEquals("Label should match", 9, image.label());
        assertArrayEquals("Data should match", testData, image.data());
    }
}
