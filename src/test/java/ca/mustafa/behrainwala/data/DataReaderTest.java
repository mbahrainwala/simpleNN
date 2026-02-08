package ca.mustafa.behrainwala.data;

import ca.behrainwala.mustafa.data.DataReader;
import ca.behrainwala.mustafa.data.Image;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class DataReaderTest {
    @Test
    public void testReadData() throws IOException {
        /* ── Arrange ──────────────────────────────────────────────────────── */
        // Create a temporary file that will be deleted automatically
        Path tempFile = Files.createTempFile("mnist", ".csv");

        // Build a minimal CSV that follows the format expected by DataReader
        StringBuilder sb = new StringBuilder();
        // first image – label 1, all pixels 0
        sb.append("1");
        sb.append(",0".repeat(28 * 28));
        sb.append(System.lineSeparator());
        // second image – label 2, all pixels 1
        sb.append("2");
        sb.append(",1".repeat(28 * 28));
        sb.append(System.lineSeparator());

        // Write the CSV into the temp file
        try (BufferedWriter writer = Files.newBufferedWriter(tempFile)) {
            writer.write(sb.toString());
        }

        /* ── Act ───────────────────────────────────────────────────────────── */
        List<Image> images = DataReader.readData(tempFile.toString());

        /* ── Assert ──────────────────────────────────────────────────────── */
        assertEquals("Number of images", 2, images.size());

        Image first = images.get(0);
        assertEquals("First image label", 1, first.label());
        assertEquals("First image pixel[0][0]", 0.0, first.data()[0][0], 1e-9);

        Image second = images.get(1);
        assertEquals("Second image label", 2, second.label());
        assertEquals("Second image pixel[0][0]", 1.0, second.data()[0][0], 1e-9);

        /* ── Cleanup ─────────────────────────────────────────────────────── */
        Files.deleteIfExists(tempFile);
    }
}
