package ca.behrainwala.mustafa.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class ImageConverterTest {

    @TempDir
    File tempDir;

    @Test
    void testGetImageGrayscaleConversion() throws IOException {
        // 2×2 image with distinct colours
        BufferedImage testImage = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        testImage.setRGB(0, 0, new Color(255, 0, 0).getRGB()); // red
        testImage.setRGB(1, 0, new Color(0, 255, 0).getRGB()); // green
        testImage.setRGB(0, 1, new Color(0, 0, 255).getRGB()); // blue
        testImage.setRGB(1, 1, new Color(255, 255, 255).getRGB()); // white

        File imageFile = new File(tempDir, "test.png");
        ImageIO.write(testImage, "png", imageFile);

        // Converter: size equals original so no resizing takes place
        ImageConverter converter = new ImageConverter(2, null);

        int label = 42;
        Image image = converter.getImage(imageFile.getAbsolutePath(), label);

        // Verify the label
        assertEquals(label, image.label(), "Label should match the supplied value");

        // Expected grayscale averages
        double[][] expected = new double[2][2];
        expected[0][0] = (255 + 0 + 0) / 3.0;       // red
        expected[1][0] = (0 + 255 + 0) / 3.0;       // green
        expected[0][1] = (0 + 0 + 255) / 3.0;       // blue
        expected[1][1] = (255 + 255 + 255) / 3.0;   // white

        double[][] actual = image.data();

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                assertEquals(expected[x][y], actual[x][y], 0.0001,
                        String.format("Pixel (%d,%d) grayscale value mismatch", x, y));
            }
        }
    }
}
