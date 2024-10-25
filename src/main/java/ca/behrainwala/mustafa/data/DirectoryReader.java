package ca.behrainwala.mustafa.data;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class DirectoryReader {
    public static List<String> listFilesForFolder(String folderName) throws IOException {
        File folder = new File(folderName);
        if(!folder.isDirectory())
            return null;

        List<String> retFilePaths = new ArrayList<>();

        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isFile())
                retFilePaths.add(fileEntry.getPath());
        }

        return retFilePaths;
    }
}
