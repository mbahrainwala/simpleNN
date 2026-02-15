package ca.behrainwala.mustafa.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Unit tests for {@link EdgeDetection}.
 *
 * <p>The tests focus on the behaviour of {@link EdgeDetection#detectEdges(BufferedImage, EdgeFilter)}
 * for the special cases of {@code null} and {@code EdgeFilter.NONE}, as well as a typical
 * convolution scenario.  The image size is intentionally tiny (3×3) so that the convolution
 * is easy to reason about while still exercising the full code path.</p>
 */
public class EdgeDetectionTest {

    /** Helper that creates a 3×3 {@link BufferedImage} filled with the specified RGB value. */
    private static BufferedImage createSolidImage(int rgb) {
        BufferedImage img = new BufferedImage(3, 3, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                img.setRGB(x, y, rgb);
            }
        }
        return img;
    }

    /** Helper that creates a 3×3 gradient image (top‑left 0, bottom‑right 255). */
    private static BufferedImage createGradientImage() {
        BufferedImage img = new BufferedImage(3, 3, BufferedImage.TYPE_INT_RGB);
        int[][] values = {
                {0, 64, 128},
                {64, 128, 192},
                {128, 192, 255}
        };
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                int gray = values[y][x];
                img.setRGB(x, y, new Color(gray, gray, gray).getRGB());
            }
        }
        return img;
    }

    @Test
    public void detectEdges_returnsSameInstance_whenFilterIsNone() {
        EdgeDetection detector = new EdgeDetection();
        BufferedImage original = createSolidImage(0xFFFFFFFF); // white

        BufferedImage result = detector.detectEdges(original, EdgeFilter.NONE);

        // The implementation explicitly returns the original image instance
        assertSame(original, result, "Result should be the same instance when filter is NONE");
    }

    @Test
    public void detectEdges_returnsSameInstance_whenFilterIsNull() {
        EdgeDetection detector = new EdgeDetection();
        BufferedImage original = createSolidImage(0xFF000000); // black

        BufferedImage result = detector.detectEdges(original, null);

        // The implementation explicitly returns the original image instance
        assertSame(original, result, "Result should be the same instance when filter is null");
    }

    @Test
    public void detectEdges_appliesFilter_andChangesPixelValues() {
        EdgeDetection detector = new EdgeDetection();
        BufferedImage original = createSolidImage(0xFFFFFFFF); // white image

        BufferedImage result = detector.detectEdges(original, EdgeFilter.VERTICAL_FILTER);

        // The result must be a new image instance
        assertNotSame(original, result, "Result should not be the same instance when filter is applied");

        // Convolving a uniform white image with a zero‑sum filter should produce all zeroes,
        // which are mapped to 0 by fixOutOfRangeRGBValues.  Hence the resulting image
        // should be completely black.
        for (int y = 0; y < result.getHeight(); y++) {
            for (int x = 0; x < result.getWidth(); x++) {
                int rgb = result.getRGB(x, y);
                // Expect 0xFF000000 (fully opaque black)
                assertEquals(0xFF000000, rgb, "Pixel at (%d,%d) should be black".formatted(x, y));
            }
        }
    }

    @Test
    public void detectEdges_onGradientChangesPixels() {
        EdgeDetection detector = new EdgeDetection();
        BufferedImage gradient = createGradientImage();

        BufferedImage result = detector.detectEdges(gradient, EdgeFilter.HORIZONTAL_FILTER);

        // The result must not be identical to the original
        assertNotSame(gradient, result, "Result should be a new image after filtering");

        // Verify that at least one pixel has changed (edge detection should create contrasts)
        boolean different = false;
        outer:
        for (int y = 0; y < result.getHeight(); y++) {
            for (int x = 0; x < result.getWidth(); x++) {
                if (result.getRGB(x, y) != gradient.getRGB(x, y)) {
                    different = true;
                    break outer;
                }
            }
        }
        assertTrue(different, "At least one pixel should differ after applying a filter");
    }
}
