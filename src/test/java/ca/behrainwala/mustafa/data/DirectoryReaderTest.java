package ca.behrainwala.mustafa.data;

import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.*;

public class DirectoryReaderTest {

    @Test
    public void testListFilesForFolder() throws IOException {
        // Arrange: create a temporary directory with two files
        Path tempDir = Files.createTempDirectory("dirreader-test");
        Path file1 = Files.createTempFile(tempDir, "file1", ".txt");
        Path file2 = Files.createTempFile(tempDir, "file2", ".txt");

        try {
            // Act
            List<String> fileList = DirectoryReader.listFilesForFolder(tempDir.toString());

            // Assert
            assertNotNull("File list should not be null", fileList);
            assertEquals("Should contain two files", 2, fileList.size());
            assertTrue("Should contain file1", fileList.contains(file1.toString()));
            assertTrue("Should contain file2", fileList.contains(file2.toString()));
        } finally {
            // Cleanup
            Files.deleteIfExists(file1);
            Files.deleteIfExists(file2);
            Files.deleteIfExists(tempDir);
        }
    }

    @Test
    public void testListFilesForNonDirectory() throws IOException {
        // Arrange: create a temporary file (not a directory)
        Path tempFile = Files.createTempFile("dirreader-nondir", ".tmp");

        try {
            // Act
            List<String> result = DirectoryReader.listFilesForFolder(tempFile.toString());

            // Assert
            assertNull("Result should be null for non-directory", result);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
}
