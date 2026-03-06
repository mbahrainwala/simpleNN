package ca.behrainwala.mustafa.layer;

import ca.behrainwala.mustafa.data.TextToken;

/**
 * ==================================================================================
 * WordVectorGraph - In-Memory Word Vector Representation
 * ==================================================================================
 *
 * WHAT IS A WORD VECTOR?
 * -----------------------
 * A word vector (or "word embedding") is a list of numbers that represents a word.
 * Words with similar meanings or that appear in similar contexts get similar vectors.
 *
 * Example (simplified 3D vectors):
 *   "king"  = [0.9, 0.1, 0.8]
 *   "queen" = [0.9, 0.1, 0.7]    <- similar to "king"!
 *   "cat"   = [0.1, 0.8, 0.2]    <- very different from "king"
 *
 * This is the foundation of how modern AI understands language.
 * Models like Word2Vec, GloVe, and the embeddings inside GPT/BERT all use this idea.
 *
 *
 * HOW THIS CLASS BUILDS WORD VECTORS:
 * -------------------------------------
 * We use the "distributional hypothesis": words that appear in similar contexts
 * have similar meanings. ("You shall know a word by the company it keeps" - Firth, 1957)
 *
 * Algorithm (Co-occurrence based):
 * 1. Scan through all the training text
 * 2. For each word, count how often each other word appears nearby (within a window)
 * 3. These co-occurrence counts become the word's vector
 * 4. We use the top-K most frequent words as "anchor dimensions" to keep vectors small
 * 5. Normalize all vectors to unit length for fair comparison
 *
 * Example: If "king" often appears near "the", "his", "was", "and", "of":
 *   king_vector = [count_with_the, count_with_and, count_with_of, count_with_to, ...]
 *   (normalized to unit length)
 *
 *
 * THE WORD GRAPH:
 * ----------------
 * Once we have vectors, we can measure "distance" between words using cosine similarity:
 *   similarity(A, B) = (A dot B) / (|A| * |B|)
 *
 * Values range from -1 (opposite) to 1 (identical).
 * This creates a graph where each word is a node and edges connect similar words.
 *
 * The graph reveals interesting patterns:
 *   - "man" is near "woman", "boy", "young"
 *   - "said" is near "replied", "asked", "told"
 *   - "house" is near "room", "door", "window"
 *
 *
 * HOW THIS IMPROVES THE LLM:
 * ----------------------------
 * Without embeddings: each word is just an arbitrary number (token ID).
 *   "cat"=42, "dog"=67 - the network sees no relationship between them.
 *
 * With embeddings: each word is a meaningful vector.
 *   "cat"=[0.1, 0.8, 0.2, ...], "dog"=[0.1, 0.7, 0.3, ...] - very similar inputs!
 *   The network can generalize: if it learns "the cat sat" it can also predict
 *   reasonable continuations for "the dog sat" because the inputs are similar.
 *
 * This is called "transfer learning through shared representations" - one of the
 * most important ideas in modern deep learning.
 *
 *
 * USAGE:
 * -------
 *   // Build vectors from tokenized text
 *   WordVectorGraph graph = new WordVectorGraph(tokenIds, vocabSize, 8);
 *
 *   // Get a word's vector
 *   double[] vec = graph.getVector(tokenId);
 *
 *   // Find similar words
 *   int[] similar = graph.findSimilar(tokenId, 5);
 *
 *   // Print word clusters
 *   graph.printWordClusters(tokenizer);
 *
 *   // Build input for neural network (replaces raw token IDs with embeddings)
 *   double[] input = graph.buildInput(tokenIds, startPos, contextSize);
 *
 * @see ca.behrainwala.mustafa.SimpleLLM - Uses this class for word representations
 */
public class WordVectorGraphLayer extends Layer {

    /** Word vectors: vectors[tokenId] = embedding for that token */
    private final double[][] vectors;

    /** Number of words in the vocabulary (including token 0 = unknown) */
    private final int vocabSize;

    /** Dimensionality of each word vector */
    private final int embeddingDim;

    /** Number of tokens in the context window (0 if used standalone) */
    private final int contextSize;

    /** Scale factor applied to embedding output (1.0 if used standalone) */
    private final double scaleFactor;

    /** Co-occurrence window size (how many words left/right to look) */
    private static final int WINDOW_SIZE = 5;

    /**
     * Builds word vectors from a tokenized text corpus.
     *
     * The algorithm:
     * 1. For each token in the text, look at its neighbors within WINDOW_SIZE
     * 2. Count co-occurrences with the top embeddingDim "anchor" words
     *    (tokens 1, 2, ..., embeddingDim are the most frequent words)
     * 3. Apply log scaling to dampen very frequent co-occurrences
     * 4. Normalize each vector to unit length
     *
     * The anchor words are the most frequent words in the vocabulary
     * (typically: the, and, of, to, a, in, i, he, ...).
     * Each dimension of the word vector captures how strongly a word
     * co-occurs with that particular anchor word.
     *
     * @param tokenIds   Array of token IDs representing the full text
     * @param vocabSize  Total vocabulary size (including unknown token 0)
     * @param embeddingDim Number of dimensions for each word vector.
     *                     Must be less than vocabSize. Recommended: 8-32.
     */
    public WordVectorGraphLayer(int[] tokenIds, int vocabSize, int embeddingDim) {
        this(tokenIds, vocabSize, embeddingDim, 0, 1.0);
    }

    /**
     * Builds word vectors for use as a Layer in the neural network pipeline.
     *
     * When used as a layer, getOutput() accepts an array of contextSize token IDs
     * (as doubles), looks up each token's embedding vector, concatenates them,
     * divides by scaleFactor, and passes the result to the next layer.
     *
     * @param tokenIds     Array of token IDs representing the full text
     * @param vocabSize    Total vocabulary size (including unknown token 0)
     * @param embeddingDim Number of dimensions for each word vector
     * @param contextSize  Number of tokens in the context window
     * @param scaleFactor  Scale factor to apply to embedding output
     */
    public WordVectorGraphLayer(int[] tokenIds, int vocabSize, int embeddingDim,
                                int contextSize, double scaleFactor) {
        this.vocabSize = vocabSize;
        this.embeddingDim = Math.min(embeddingDim, vocabSize - 1);
        this.vectors = new double[vocabSize][this.embeddingDim];
        this.contextSize = contextSize;
        this.scaleFactor = scaleFactor;

        buildVectors(tokenIds);
    }

    /**
     * Scans the text and builds co-occurrence vectors for all words.
     *
     * For each position in the text:
     *   - Look at the token at that position (the "center" word)
     *   - Look at all tokens within WINDOW_SIZE positions left and right
     *   - If a neighbor token is one of the anchor words (tokens 1..embeddingDim),
     *     increment the center word's co-occurrence count for that anchor dimension
     *
     * After counting, apply log(1 + count) scaling and normalize to unit length.
     */
    private void buildVectors(int[] tokenIds) {
        // Step 1: Count co-occurrences with anchor words
        // Anchor words are tokens 1 through embeddingDim (the most frequent words)
        for (int i = 0; i < tokenIds.length; i++) {
            int center = tokenIds[i];
            if (center <= 0 || center >= vocabSize) continue;

            // Look at neighbors within the window
            int start = Math.max(0, i - WINDOW_SIZE);
            int end = Math.min(tokenIds.length - 1, i + WINDOW_SIZE);

            for (int j = start; j <= end; j++) {
                if (j == i) continue; // skip self
                int neighbor = tokenIds[j];

                // Only count co-occurrence with anchor words (tokens 1..embeddingDim)
                if (neighbor > 0 && neighbor <= embeddingDim) {
                    vectors[center][neighbor - 1] += 1.0;
                }
            }
        }

        // Step 2: Apply log scaling to dampen very frequent co-occurrences
        // Without this, "the" would dominate every vector since it's so common.
        // log(1 + count) compresses the range: 0->0, 1->0.69, 10->2.4, 100->4.6
        for (int i = 0; i < vocabSize; i++) {
            for (int d = 0; d < embeddingDim; d++) {
                vectors[i][d] = Math.log(1.0 + vectors[i][d]);
            }
        }

        // Step 3: Normalize each vector to unit length (L2 norm = 1)
        // This ensures cosine similarity is just the dot product,
        // and makes all vectors comparable regardless of word frequency.
        for (int i = 0; i < vocabSize; i++) {
            double norm = 0;
            for (int d = 0; d < embeddingDim; d++) {
                norm += vectors[i][d] * vectors[i][d];
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int d = 0; d < embeddingDim; d++) {
                    vectors[i][d] /= norm;
                }
            }
        }
    }

    /**
     * Returns the embedding vector for a given token.
     *
     * @param tokenId The token's integer ID
     * @return The word's embedding vector (length = embeddingDim)
     */
    public double[] getVector(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabSize) {
            return new double[embeddingDim]; // zero vector for invalid tokens
        }
        return vectors[tokenId];
    }

    /**
     * Returns the dimensionality of the word vectors.
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }

    /**
     * Builds a neural network input array by replacing each token ID with its
     * embedding vector. This creates a richer input than raw token IDs.
     *
     * Raw token IDs:  [42, 15, 3, 8]    -> input length = 4
     * With embeddings: [vec(42), vec(15), vec(3), vec(8)] -> input length = 4 * embeddingDim
     *
     * Each position in the context window contributes embeddingDim features
     * instead of just 1 feature. This gives the network much more information
     * about each word's meaning and relationships.
     *
     * @param tokenIds    The full token sequence
     * @param startPos    Starting position in the token sequence
     * @param contextSize Number of tokens in the context window
     * @return Input array of length contextSize * embeddingDim
     */
    public double[] buildInput(int[] tokenIds, int startPos, int contextSize) {
        double[] input = new double[contextSize * embeddingDim];
        for (int j = 0; j < contextSize; j++) {
            int token = tokenIds[startPos + j];
            double[] vec = getVector(token);
            System.arraycopy(vec, 0, input, j * embeddingDim, embeddingDim);
        }
        return input;
    }

    /**
     * Computes cosine similarity between two word vectors.
     *
     * Cosine similarity measures the angle between two vectors:
     *   similarity = (A . B) / (|A| * |B|)
     *
     * Since our vectors are already normalized to unit length,
     * this simplifies to just the dot product: similarity = A . B
     *
     * Values range from -1 (opposite) to 1 (identical).
     * Typical thresholds:
     *   > 0.8  = very similar (likely related words)
     *   > 0.5  = somewhat similar
     *   < 0.3  = unrelated
     *
     * @param token1 First word's token ID
     * @param token2 Second word's token ID
     * @return Cosine similarity in range [-1, 1]
     */
    public double cosineSimilarity(int token1, int token2) {
        double[] v1 = getVector(token1);
        double[] v2 = getVector(token2);

        double dot = 0;
        for (int d = 0; d < embeddingDim; d++) {
            dot += v1[d] * v2[d];
        }
        return dot;
    }

    /**
     * Finds the K most similar words to a given word, based on cosine similarity.
     *
     * This is the core operation of the "word graph": given a word, which other
     * words are most similar? The results reveal semantic relationships that the
     * model has learned from the text.
     *
     * @param tokenId The query word's token ID
     * @param k       Number of similar words to return
     * @return Array of token IDs for the K most similar words (excluding the query word)
     */
    public int[] findSimilar(int tokenId, int k) {
        // Compute similarity with all other words
        double[] similarities = new double[vocabSize];
        for (int i = 1; i < vocabSize; i++) { // skip token 0 (unknown)
            if (i == tokenId) continue;
            similarities[i] = cosineSimilarity(tokenId, i);
        }

        // Find top K by repeatedly selecting the maximum
        int[] topK = new int[k];
        boolean[] used = new boolean[vocabSize];
        used[0] = true;       // exclude unknown token
        used[tokenId] = true;  // exclude self

        for (int n = 0; n < k; n++) {
            int bestIdx = -1;
            double bestSim = -2;
            for (int i = 1; i < vocabSize; i++) {
                if (!used[i] && similarities[i] > bestSim) {
                    bestSim = similarities[i];
                    bestIdx = i;
                }
            }
            if (bestIdx >= 0) {
                topK[n] = bestIdx;
                used[bestIdx] = true;
            }
        }

        return topK;
    }

    /**
     * Prints word clusters showing which words are grouped together in vector space.
     *
     * For each query word, displays the most similar words and their similarity scores.
     * This visualizes the "word graph" - the network of semantic relationships
     * that the model has learned from reading the training text.
     *
     * @param tokenizer Token-to-word converter for displaying results
     */
    public void printWordClusters(TextToken tokenizer) {
        System.out.println("=============== Word Vector Graph ===============");
        System.out.println("Words grouped by similarity (cosine similarity):\n");

        // Query words that demonstrate interesting relationships
        String[] queries = {
            "man", "woman", "king", "love", "war", "death",
            "house", "said", "young", "good", "great", "old",
            "heart", "dark", "night", "father", "mother", "child",
            "money", "time", "world", "hand", "eyes", "head",
            "book", "wrote", "author"
        };

        int topK = 8;
        for (String word : queries) {
            int token = tokenizer.getTextToken(word);
            if (token <= 0) continue; // skip words not in vocabulary

            int[] similar = findSimilar(token, topK);
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("  %-10s -> ", word));

            for (int i = 0; i < topK; i++) {
                if (similar[i] <= 0) break;
                String simWord = tokenizer.getTokenText(similar[i]);
                double sim = cosineSimilarity(token, similar[i]);
                sb.append(String.format("%s(%.2f) ", simWord, sim));
            }

            System.out.println(sb.toString());
        }
        System.out.println();
    }

    // ==================== Layer interface ====================

    /**
     * Forward pass: converts token IDs (as doubles) into concatenated embedding vectors.
     *
     * Input is contextSize doubles where each value is a token ID.
     * Output is contextSize * embeddingDim doubles (the concatenated embeddings),
     * divided by scaleFactor. The result is passed to the next layer.
     */
    @Override
    public double[] getOutput(double[] input) {
        double[] output = new double[contextSize * embeddingDim];
        for (int j = 0; j < contextSize; j++) {
            int token = (int) Math.round(input[j]);
            double[] vec = getVector(token);
            for (int d = 0; d < embeddingDim; d++) {
                output[j * embeddingDim + d] = vec[d] / scaleFactor;
            }
        }

        if (getNextLayer() != null)
            return getNextLayer().getOutput(output);
        else
            return output;
    }

    /**
     * Backpropagation: no-op since word vectors are not trained by gradient descent.
     * The embeddings are fixed co-occurrence vectors built from the text corpus.
     */
    @Override
    public void backPropagate(double[] error) {
        // No trainable weights — embeddings are fixed
    }

    @Override
    public int getNumberOutput() {
        return contextSize * embeddingDim;
    }

    @Override
    public int getOutputRows() {
        return (int) Math.sqrt(getNumberOutput());
    }

    @Override
    public int getOutputCols() {
        return getNumberOutput() / getOutputRows();
    }
}
