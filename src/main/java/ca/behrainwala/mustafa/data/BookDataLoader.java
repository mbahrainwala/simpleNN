package ca.behrainwala.mustafa.data;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Static utility class for loading and preprocessing book text data.
 * Handles file loading, vocabulary construction, tokenization, and
 * identification of valid training positions for the SimpleLLM.
 */
public class BookDataLoader {

    /**
     * Holds the results of vocabulary building: the tokenizer, the set of
     * vocabulary words, and the total vocabulary size (including unknown token 0).
     */
    public static class VocabularyResult {
        public final TextToken tokenizer;
        public final Set<String> vocabSet;
        public final int vocabSize;

        public VocabularyResult(TextToken tokenizer, Set<String> vocabSet, int vocabSize) {
            this.tokenizer = tokenizer;
            this.vocabSet = vocabSet;
            this.vocabSize = vocabSize;
        }
    }

    /**
     * Loads all .txt files from the specified directory and concatenates them.
     * Each file is expected to be a plain text book file.
     *
     * @param dir Path to the directory containing .txt book files
     * @return Concatenated text from all book files
     * @throws Exception if directory doesn't exist or contains no .txt files
     */
    public static String loadBooks(String dir) throws Exception {
        StringBuilder sb = new StringBuilder();
        File folder = new File(dir);

        if (!folder.exists() || !folder.isDirectory()) {
            throw new FileNotFoundException("Books directory not found: " + dir);
        }

        File[] files = folder.listFiles((d, name) -> name.endsWith(".txt"));
        if (files == null || files.length == 0) {
            throw new FileNotFoundException("No .txt files found in: " + dir);
        }

        Arrays.sort(files); // deterministic load order
        for (File f : files) {
            System.out.println("  Loading: " + f.getName()
                    + " (" + f.length() / 1024 + " KB)");
            sb.append(new String(Files.readAllBytes(f.toPath())));
            sb.append(" ");
        }

        return sb.toString();
    }

    /**
     * Cleans a word by removing all non-letter characters and lowercasing.
     * This normalizes text so that "Hello!", "hello", and "HELLO" all
     * become the same token "hello".
     *
     * @param word Raw word from the text
     * @return Cleaned, lowercase, letters-only version of the word
     */
    public static String cleanWord(String word) {
        return word.replaceAll("[^a-zA-Z]", "").toLowerCase();
    }

    /**
     * Counts word frequencies from the given words array, keeps the top
     * {@code maxVocab} most frequent words, and builds a {@link TextToken}
     * tokenizer with integer IDs assigned to each vocabulary word.
     *
     * @param words    Array of raw words (will be cleaned internally)
     * @param maxVocab Maximum number of vocabulary words to keep
     * @return A {@link VocabularyResult} containing the tokenizer, vocab set, and vocab size
     */
    public static VocabularyResult buildVocabulary(String[] words, int maxVocab) {
        Map<String, Integer> freq = new LinkedHashMap<>();
        for (String w : words) {
            w = cleanWord(w);
            if (!w.isEmpty()) {
                freq.merge(w, 1, Integer::sum);
            }
        }

        List<String> topWords = freq.entrySet().stream()
                .sorted((a, b) -> b.getValue() - a.getValue())
                .limit(maxVocab)
                .map(Map.Entry::getKey)
                .toList();

        Set<String> vocabSet = new HashSet<>(topWords);

        TextToken tokenizer = new TextToken();
        for (String w : topWords) {
            tokenizer.addTextToToken(w);
        }

        int vocabSize = tokenizer.getTokenSize() + 1; // +1 for unknown token (0)

        System.out.println("Vocabulary size: " + vocabSize + " words");
        System.out.println("Top 30 words: " + topWords.subList(0, Math.min(30, topWords.size())));

        return new VocabularyResult(tokenizer, vocabSet, vocabSize);
    }

    /**
     * Converts an array of raw words into an array of integer token IDs.
     * Unknown words (not in vocabSet) are mapped to token 0.
     *
     * @param words    Array of raw words (will be cleaned internally)
     * @param tokenizer The tokenizer for word-to-ID conversion
     * @param vocabSet  Set of known vocabulary words
     * @return Array of integer token IDs (trimmed to actual size)
     */
    public static int[] tokenize(String[] words, TextToken tokenizer, Set<String> vocabSet) {
        int[] tokenIds = new int[words.length];
        int tokenCount = 0;
        for (String w : words) {
            w = cleanWord(w);
            if (!w.isEmpty()) {
                tokenIds[tokenCount++] = vocabSet.contains(w) ? tokenizer.getTextToken(w) : 0;
            }
        }
        return Arrays.copyOf(tokenIds, tokenCount);
    }

    /**
     * Finds valid training positions where the target token (the token after
     * the context window) is a known vocabulary word (not token 0).
     *
     * @param tokenIds    Array of token IDs for the full text
     * @param contextSize Size of the context window
     * @return List of valid starting positions for training windows
     */
    public static List<Integer> findValidPositions(int[] tokenIds, int contextSize) {
        List<Integer> validPositions = new ArrayList<>();
        for (int i = 0; i <= tokenIds.length - contextSize - 1; i++) {
            if (tokenIds[i + contextSize] != 0) {
                validPositions.add(i);
            }
        }
        return validPositions;
    }
}
