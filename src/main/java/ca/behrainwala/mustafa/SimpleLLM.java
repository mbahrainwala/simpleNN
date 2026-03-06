package ca.behrainwala.mustafa;

import ca.behrainwala.mustafa.data.BookDataLoader;
import ca.behrainwala.mustafa.data.TextToken;
import ca.behrainwala.mustafa.layer.WordVectorGraphLayer;
import ca.behrainwala.mustafa.network.NetworkBuilder;
import ca.behrainwala.mustafa.network.NeuralNetwork;

import java.util.*;
import java.util.Scanner;

/**
 * ==================================================================================
 * SimpleLLM - A Minimal Language Model (Educational Implementation)
 * ==================================================================================
 *
 * WHAT IS A LANGUAGE MODEL (LLM)?
 * --------------------------------
 * A language model predicts the next word given a sequence of previous words.
 * For example, given "the cat sat on the", a good model predicts "mat" or "floor".
 *
 * This is the core idea behind GPT, ChatGPT, Claude, and similar AI systems.
 * Those systems use billions of parameters and transformer architectures;
 * this educational version uses a simpler neural network to demonstrate
 * the same fundamental principle: NEXT-TOKEN PREDICTION.
 *
 *
 * HOW THIS PROGRAM WORKS (5 Phases):
 * ------------------------------------
 *
 * PHASE 1 - DATA LOADING:
 *   Read raw text from book files (classic literature from Project Gutenberg).
 *   More diverse text = better word patterns for the model to learn.
 *
 * PHASE 2 - TOKENIZATION:
 *   Convert words into numbers (called "tokens"). Each unique word gets a unique ID.
 *   Example: "the"=1, "and"=2, "of"=3, "to"=4, ...
 *   We keep only the most frequent words as our vocabulary (MAX_VOCAB).
 *   Rare words are mapped to token 0 (unknown).
 *
 * PHASE 3 - TRAINING DATA (Sliding Windows):
 *   Create input-output pairs using a sliding window over the text:
 *
 *   Text:    [the, old, man, and, the, sea, was, calm, and, blue]
 *   Window:  [the, old, man, and]  ->  predict: "the"     (window size = 4 here)
 *            [old, man, and, the]  ->  predict: "sea"
 *            [man, and, the, sea]  ->  predict: "was"
 *            ... and so on, sliding one word at a time
 *
 *   Each window of CONTEXT_SIZE tokens becomes the input, and the very next
 *   token becomes the target the network must learn to predict.
 *
 * PHASE 4 - TRAINING:
 *   Feed millions of these windows through the neural network.
 *   The network adjusts its weights (via backpropagation) to get better
 *   at predicting the correct next token. These weights are the model's
 *   "learned knowledge" - stored entirely in memory as weight matrices.
 *
 *   The weight matrices inside the ConnectedLayers ARE the "learning vectors" -
 *   they encode all the word co-occurrence patterns the model has learned.
 *
 * PHASE 5 - TEXT GENERATION (Autoregressive):
 *   To generate text, we feed a seed phrase through the network, get a
 *   prediction for the next word, append it to the input, and repeat.
 *   This is called "autoregressive generation" - each prediction becomes
 *   part of the input for the next prediction.
 *
 *   Seed:      "it was"
 *   Step 1:    input=[..., it, was]    -> predict "a"      -> "it was a"
 *   Step 2:    input=[..., was, a]     -> predict "very"   -> "it was a very"
 *   Step 3:    input=[..., a, very]    -> predict "good"   -> "it was a very good"
 *
 *
 * KEY CONCEPTS FOR STUDENTS:
 * ---------------------------
 * - VOCABULARY: The set of words the model knows (most frequent words from training text)
 * - TOKEN: A word converted to a number (integer ID)
 * - CONTEXT WINDOW: How many previous tokens the model looks at (CONTEXT_SIZE)
 * - EPOCH: One complete pass through the training data
 * - ACCURACY: % of times the model correctly predicts the exact next word
 * - TEMPERATURE: Controls randomness in generation (low=predictable, high=creative)
 * - SCALE FACTOR: Normalizes input values to prevent numerical instability
 * - SOFTMAX: Converts raw network outputs into a probability distribution
 *
 *
 * ARCHITECTURE:
 * -------------
 * Input (16 words x 32-dim embeddings = 512 features) -> Hidden Layer (512 neurons, ReLU) -> Output (801 neurons, linear)
 *
 * Each word in the context window is represented by a 32-dimensional embedding
 * vector from the WordVectorGraph. This gives the network 512 input features
 * (16 words * 32 dimensions) instead of just 16 raw token IDs.
 * Words with similar meanings have similar embedding vectors, so the network
 * can generalize: learning about "king" helps it understand "queen" too.
 *
 * The hidden layer with 512 neurons uses ReLU activation. The output layer
 * has one neuron per vocabulary word and uses linear activation for full
 * gradient flow. The neuron with the highest activation indicates the
 * model's prediction for the next word.
 *
 * Total parameters: 512*512 + 512*801 = 262,144 + 410,112 = ~672K weights
 * (Compare to GPT-3's 175 BILLION parameters!)
 *
 *
 * WHY SCALE_FACTOR MATTERS:
 * --------------------------
 * The existing ConnectedLayer uses:
 *   - Weight initialization: random Gaussian N(0,1)
 *   - Learning rate: 0.1 (fixed)
 *
 * Without scaling, token IDs (0-500) fed directly into the network would cause
 * enormous activations, leading to "gradient explosion" (weights change too wildly).
 * The SCALE_FACTOR shrinks inputs to small values (0 to ~0.1), keeping
 * activations and gradients in a stable range. This is similar to how
 * real models use "learning rate scheduling" and "weight initialization schemes"
 * (like Xavier/He initialization) to achieve the same stability.
 *
 *
 * RUNNING THIS PROGRAM:
 * ----------------------
 *   mvn exec:java -Dexec.mainClass="ca.behrainwala.mustafa.SimpleLLM"
 *
 * Expected behavior:
 *   - Accuracy increases from ~1% to ~7-10% over training (predicting exact next
 *     word from 500 options is hard! Random chance = 0.2%)
 *   - Generated text shows learned English word patterns but won't be grammatical
 *   - Training takes a few minutes on a modern CPU
 *
 * @see ca.behrainwala.mustafa.data.TextToken - Handles word-to-token conversion
 * @see ca.behrainwala.mustafa.network.NetworkBuilder - Constructs the neural network
 * @see ca.behrainwala.mustafa.network.NeuralNetwork - Manages training and prediction
 */
public class SimpleLLM {

    // ==================== HYPERPARAMETERS ====================
    // These control the model's capacity and training behavior.
    // Tuning these is a key part of machine learning ("hyperparameter search").

    /**
     * Number of previous tokens the model sees when predicting the next token.
     * Larger = more context but more parameters and slower training.
     * Real LLMs use 2048-128000+ tokens of context.
     */
    private static final int CONTEXT_SIZE = 128;

    /**
     * Maximum vocabulary size - the number of unique words the model knows.
     * Words outside this set are mapped to token 0 (unknown/ignored).
     * We keep only the most frequent words since they cover most of the text.
     * The top 500 English words typically cover ~75-80% of all word occurrences.
     * Real LLMs use 30,000-100,000+ tokens (including subword pieces).
     */
    private static final int MAX_VOCAB = 800;

    /**
     * Dimensionality of each word's embedding vector.
     * Each word is represented as a vector of this many numbers.
     * The vectors are built from co-occurrence patterns in the text using
     * the WordVectorGraph class. More dimensions = richer representation
     * but more network parameters (input size = CONTEXT_SIZE * EMBEDDING_DIM).
     *
     * With EMBEDDING_DIM=32 and CONTEXT_SIZE=16, input size = 512 features.
     * This replaces raw token IDs with meaningful vectors where similar words
     * have similar inputs, helping the network generalize.
     */
    private static final int EMBEDDING_DIM = 32;

    /**
     * Number of hidden neurons in the network's hidden layer.
     * More neurons = more capacity to learn patterns, but slower training.
     * This layer learns to represent word relationships and patterns.
     */
    private static final int HIDDEN_SIZE = 512;

    /**
     * Number of training epochs (full passes through the data).
     * More epochs = model sees data more times = better learning (up to a point).
     * Too many epochs can lead to "overfitting" (memorizing instead of generalizing).
     */
    private static final int EPOCHS = 200;

    /**
     * Number of training examples shown per epoch.
     * With ~4+ million available windows, we use a large subset each epoch.
     * Each "sample" is one sliding window that trains the network once.
     */
    private static final int SAMPLES_PER_EPOCH = 300000;

    /** Number of words to generate after each seed phrase. */
    private static final int GENERATE_LENGTH = 80;

    /**
     * Temperature for text generation (controls randomness).
     *   0.1 = very deterministic (always picks most likely word)
     *   0.5 = somewhat random (usually picks likely words)
     *   0.8 = moderately random (good balance of quality and variety)
     *   1.0 = fully random according to model's probabilities
     *   2.0 = very random (picks unlikely words often)
     *
     * Mathematically: probability[i] = exp(output[i] / temperature) / sum
     * Lower temperature sharpens the distribution; higher flattens it.
     */
    private static final double TEMPERATURE = 0.7;

    /**
     * Scale factor for input normalization.
     * The NeuralNetwork divides all inputs by this value before processing.
     *
     * WHY SO LARGE? (Important concept!)
     * Token IDs range from 0 to ~500. Without scaling, these large values
     * would cause the first layer's activations to be enormous (sum of
     * 16 inputs * N(0,1) weights * ~250 average value = huge numbers).
     * Large activations lead to large gradients, which with lr=0.1 cause
     * catastrophic weight updates ("gradient explosion").
     *
     * With embedding-based input (values 0 to ~1 after normalization),
     * SCALE_FACTOR=30 shrinks them to ~0.03, keeping activations in a
     * reasonable range and gradient updates small (~2% per step).
     * This value is tuned for EMBEDDING_DIM=8, CONTEXT_SIZE=16, HIDDEN=256.
     */
    private static final double SCALE_FACTOR = 30.0;

    // ==================== MAIN ENTRY POINT ====================

    public static void main(String[] args) throws Exception {
        System.out.println("SimpleLLM - Minimal Language Model");
        System.out.println("===================================");
        System.out.println("Training a neural network to predict the next word in a sequence.\n");

        // ============================================================
        // PHASE 1: Load training text from book files
        // ============================================================
        // We read classic literature (public domain books from Project Gutenberg).
        // More text = more word patterns for the model to learn from.
        String text = BookDataLoader.loadBooks("data/books");
        System.out.println("Loaded " + text.length() + " characters ("
                + (text.length() / 1024 / 1024) + " MB)\n");

        // Split into individual words (tokens are word-level in this model)
        String[] words = text.toLowerCase().split("\\s+");
        System.out.println("Total words: " + words.length);

        // ============================================================
        // PHASE 2: Build vocabulary (word -> integer token mapping)
        // ============================================================
        BookDataLoader.VocabularyResult vocab = BookDataLoader.buildVocabulary(words, MAX_VOCAB);
        TextToken tokenizer = vocab.tokenizer;
        Set<String> vocabSet = vocab.vocabSet;
        int vocabSize = vocab.vocabSize;

        // ============================================================
        // PHASE 3: Convert text to token sequence & find valid training positions
        // ============================================================
        int[] tokenIds = BookDataLoader.tokenize(words, tokenizer, vocabSet);
        System.out.println("Token sequence length: " + tokenIds.length);

        List<Integer> validPositions = BookDataLoader.findValidPositions(tokenIds, CONTEXT_SIZE);
        System.out.println("Valid training positions: " + validPositions.size());

        double coveragePercent = (double) validPositions.size() /
                Math.max(1, tokenIds.length - CONTEXT_SIZE) * 100;
        System.out.printf("Vocabulary coverage: %.1f%% of text\n\n", coveragePercent);

        // ============================================================
        // PHASE 4: Build the neural network (with embedded word vectors)
        // ============================================================
        // Architecture: WordVectorGraph(16 tokens -> 512 embeddings) -> Hidden(512, ReLU) -> Output(vocabSize, linear)
        //
        // The WordVectorGraph is the first layer in the network. It converts
        // raw token IDs into embedding vectors using co-occurrence patterns.
        // Words that appear in similar contexts get similar vectors, helping
        // the network generalize. The network handles the full pipeline:
        //   token IDs[16] ---(WordVectorGraph)--->  embeddings[512]
        //   embeddings[512] ---(W1: 512x512)---> hidden[512] ---(W2: 512x801)---> output[801]
        //
        // W1 learns to combine word embeddings into context features.
        // W2 learns to map those features to next-word predictions.
        System.out.println("Building word vector graph (embedding dim = " + EMBEDDING_DIM + ")...");
        NetworkBuilder nb = new NetworkBuilder(CONTEXT_SIZE, SCALE_FACTOR);
        WordVectorGraphLayer wordVectors = nb.addWordVectorLayer(tokenIds, vocabSize, EMBEDDING_DIM, CONTEXT_SIZE);

        // Display the word graph: which words are grouped together
        wordVectors.printWordClusters(tokenizer);

        nb.addConnectedLayer(HIDDEN_SIZE);
        nb.addOutputLayer(vocabSize);
        NeuralNetwork nn = nb.build();

        int inputSize = CONTEXT_SIZE * EMBEDDING_DIM;
        int totalParams = inputSize * HIDDEN_SIZE + HIDDEN_SIZE * vocabSize;
        System.out.println("Network: " + inputSize + " (16x" + EMBEDDING_DIM + ") -> "
                + HIDDEN_SIZE + " -> " + vocabSize);
        System.out.println("Total parameters (weights): " + String.format("%,d", totalParams));
        System.out.println("Scale factor: " + SCALE_FACTOR);
        System.out.println("Training: " + EPOCHS + " epochs x " +
                String.format("%,d", SAMPLES_PER_EPOCH) + " samples\n");

        // ============================================================
        // PHASE 5: Train the model on next-token prediction
        // ============================================================
        // For each training sample:
        //   1. Pick a random position in the text
        //   2. Extract a window of CONTEXT_SIZE token IDs as input
        //   3. The token right after the window is the target (correct answer)
        //   4. Feed the input through the network (forward pass)
        //   5. Compare output to the target (compute error)
        //   6. Adjust weights to reduce error (backpropagation)
        //
        // This is called STOCHASTIC GRADIENT DESCENT (SGD):
        //   - "Stochastic" = we use random samples, not the whole dataset
        //   - "Gradient" = we compute the direction of steepest error reduction
        //   - "Descent" = we move weights in that direction to reduce error
        Random rand = new Random(42);
        int totalPositions = validPositions.size();

        System.out.println("Training started...");
        System.out.println("(Random chance accuracy = " +
                String.format("%.2f", 100.0 / vocabSize) + "% for " + vocabSize + " words)\n");

        double bestAccuracy = 0;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Save weights before each epoch so we can roll back if accuracy drops
            nn.saveWeights();

            int correct = 0;
            long startTime = System.currentTimeMillis();

            // Shuffle training positions for each epoch.
            // This prevents the model from memorizing the order of examples
            // and helps it generalize to new contexts.
            Collections.shuffle(validPositions, rand);

            int numSamples = Math.min(SAMPLES_PER_EPOCH, totalPositions);
            for (int s = 0; s < numSamples; s++) {
                // Create the input window on-the-fly (memory efficient)
                // The input is raw token IDs — the WordVectorGraph layer inside the
                // network converts them to embedding vectors automatically.
                int pos = validPositions.get(s);
                double[] input = new double[CONTEXT_SIZE];
                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    input[j] = tokenIds[pos + j];
                }
                int target = tokenIds[pos + CONTEXT_SIZE];

                // Train: forward pass + backpropagation
                // The NeuralNetwork internally:
                //   1. Scales input by 1/SCALE_FACTOR
                //   2. Forward propagates through layers
                //   3. Computes error = output - one_hot(target)
                //   4. Backpropagates error, updating all weights
                int predicted = nn.train(input, target);

                // Track accuracy: did the network's top prediction match the target?
                if (predicted == target) correct++;

                if ((s + 1) % 50000 == 0) {
                    double acc = (double) correct / (s + 1) * 100;
                    System.out.printf("  Epoch %2d - Step %6d/%d - Accuracy: %.2f%%\n",
                            epoch + 1, s + 1, numSamples, acc);
                }
            }

            long elapsed = System.currentTimeMillis() - startTime;
            double acc = (double) correct / numSamples * 100;

            // If accuracy dropped compared to the best, discard this epoch's changes
            if (acc < bestAccuracy) {
                nn.restoreWeights();
                System.out.printf("Epoch %2d discarded - Accuracy: %.2f%% < best %.2f%% - rolled back - Time: %.1fs\n\n",
                        epoch + 1, acc, bestAccuracy, elapsed / 1000.0);
            } else {
                bestAccuracy = acc;
                System.out.printf("Epoch %2d complete - Accuracy: %.2f%% - Time: %.1fs\n\n",
                        epoch + 1, acc, elapsed / 1000.0);
            }

            // Stop early if we've reached 60% accuracy
            if (bestAccuracy >= 60.0) {
                System.out.println("Reached 60% accuracy — stopping training early.\n");
                break;
            }
        }

        // ============================================================
        // PHASE 6: Generate text using the trained model
        // ============================================================
        // Autoregressive generation:
        //   1. Start with a seed phrase (converted to token IDs)
        //   2. Pad/truncate to CONTEXT_SIZE tokens
        //   3. Feed through network to get output probabilities
        //   4. Sample a next token using temperature-scaled softmax
        //   5. Append the predicted token to the context
        //   6. Slide the window forward by 1 and repeat
        //
        // The temperature parameter controls how "creative" vs "safe" the
        // generation is. Lower temperature = more deterministic (picks the
        // most likely word), higher = more random (explores unlikely words).
        System.out.println("=============== Text Generation ===============\n");
        System.out.println("Temperature: " + TEMPERATURE + " (lower=more deterministic, higher=more creative)\n");

        String[] seeds = {
            "who wrote pride and prejudice",
            "describe moby dick",
            "who is the author of frankenstein",
            "what is war and peace about",
            "tell me about sherlock holmes",
            "the old man had",
            "she said to him",
            "it was a dark"
        };
        for (String seed : seeds) {
            String generated = generate(nn, tokenizer, seed, vocabSize, vocabSet, rand);
            System.out.println("Seed: \"" + seed + "\"");
            System.out.println("  -> " + generated);
            System.out.println();
        }

        // ============================================================
        // PHASE 7: Interactive Chat Mode
        // ============================================================
        // After training, the user can type prompts and the model will
        // generate continuations. This simulates a basic "chat" interface,
        // though the model is only doing next-word prediction, not true
        // conversation. It completes your text, like early autocomplete.
        //
        // Special commands:
        //   "quit" or "exit"  - End the session
        //   "temp 0.5"        - Change the temperature (generation randomness)
        //   "len 100"         - Change the generation length
        //
        // NOTE: This is a word-level model with a 500-word vocabulary.
        // It will not understand questions or produce meaningful answers.
        // It predicts likely word sequences based on patterns from classic literature.
        interactiveChat(nn, tokenizer, vocabSize, vocabSet, rand);
    }

    // ==================== INTERACTIVE CHAT ====================

    /**
     * Runs an interactive REPL (Read-Eval-Print Loop) where the user types
     * prompts and the model generates text continuations.
     *
     * This demonstrates how LLMs are used in practice: the user provides
     * a prompt (seed text), and the model generates a continuation by
     * repeatedly predicting the next most likely word.
     *
     * Supports commands:
     *   - "quit" / "exit": End the session
     *   - "temp <value>":  Change generation temperature (e.g., "temp 0.5")
     *   - "len <value>":   Change generation length (e.g., "len 100")
     *
     * @param nn          The trained neural network
     * @param tokenizer   Token converter
     * @param vocabSize   Vocabulary size
     * @param vocabSet    Set of known vocabulary words
     * @param rand        Random number generator
     */
    private static void interactiveChat(NeuralNetwork nn, TextToken tokenizer,
                                        int vocabSize, Set<String> vocabSet, Random rand) {
        Scanner scanner = new Scanner(System.in);
        double currentTemp = TEMPERATURE;
        int currentLen = GENERATE_LENGTH;

        System.out.println("=============== Interactive Mode ===============");
        System.out.println("Type any text and the model will continue it.");
        System.out.println("Commands: 'quit' to exit, 'temp 0.5' to change temperature,");
        System.out.println("          'len 100' to change generation length.");
        System.out.println("================================================\n");

        while (true) {
            System.out.print("You> ");
            System.out.flush();

            if (!scanner.hasNextLine()) break;
            String userInput = scanner.nextLine().trim();

            if (userInput.isEmpty()) continue;

            // Handle special commands
            if (userInput.equalsIgnoreCase("quit") || userInput.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            if (userInput.toLowerCase().startsWith("temp ")) {
                try {
                    currentTemp = Double.parseDouble(userInput.substring(5).trim());
                    System.out.println("Temperature set to " + currentTemp + "\n");
                } catch (NumberFormatException e) {
                    System.out.println("Invalid temperature. Usage: temp 0.5\n");
                }
                continue;
            }

            if (userInput.toLowerCase().startsWith("len ")) {
                try {
                    currentLen = Integer.parseInt(userInput.substring(4).trim());
                    System.out.println("Generation length set to " + currentLen + " words\n");
                } catch (NumberFormatException e) {
                    System.out.println("Invalid length. Usage: len 100\n");
                }
                continue;
            }

            // Generate text continuation from the user's prompt
            String generated = generateWithParams(nn, tokenizer, userInput,
                    vocabSize, vocabSet, rand, currentTemp, currentLen);
            System.out.println("\nLLM> " + generated);
            System.out.println();
        }

        scanner.close();
    }

    // ==================== TEXT GENERATION ====================

    /**
     * Generates text autoregressively using the trained neural network.
     *
     * Algorithm:
     *   1. Convert seed words to token IDs
     *   2. Pad the context to CONTEXT_SIZE from the left with 0s (unknown)
     *   3. Loop GENERATE_LENGTH times:
     *      a. Feed context through the network to get output scores
     *      b. Apply temperature-scaled softmax to get probabilities
     *      c. Sample a token from the probability distribution
     *      d. Append the token's word to the result
     *      e. Slide the context window: remove oldest token, add new one
     *
     * @param nn        The trained neural network
     * @param tokenizer Token-to-word and word-to-token converter
     * @param seed      Initial text to start generation from
     * @param vocabSize Total vocabulary size (including unknown token 0)
     * @param vocabSet  Set of known vocabulary words
     * @param rand      Random number generator for sampling
     * @return Generated text string starting with the seed
     */
    private static String generate(NeuralNetwork nn, TextToken tokenizer,
                                   String seed, int vocabSize, Set<String> vocabSet,
                                   Random rand) {
        return generateWithParams(nn, tokenizer, seed, vocabSize, vocabSet, rand,
                TEMPERATURE, GENERATE_LENGTH);
    }

    /**
     * Generates text with custom temperature and length parameters.
     * Used by both the demo generation phase and the interactive chat mode.
     *
     * @param nn          The trained neural network
     * @param tokenizer   Token-to-word and word-to-token converter
     * @param seed        Initial text to start generation from
     * @param vocabSize   Total vocabulary size (including unknown token 0)
     * @param vocabSet    Set of known vocabulary words
     * @param rand        Random number generator for sampling
     * @param temperature Controls randomness (0.1=focused, 2.0=very random)
     * @param maxLength   Maximum number of words to generate
     * @return Generated text string starting with the seed
     */
    private static String generateWithParams(NeuralNetwork nn, TextToken tokenizer,
                                             String seed, int vocabSize, Set<String> vocabSet,
                                             Random rand, double temperature, int maxLength) {
        // Convert seed words to token IDs
        String[] seedWords = seed.toLowerCase().split("\\s+");
        List<Integer> context = new ArrayList<>();

        for (String w : seedWords) {
            w = BookDataLoader.cleanWord(w);
            if (!w.isEmpty() && vocabSet.contains(w)) {
                context.add(tokenizer.getTextToken(w));
            } else {
                context.add(0); // unknown words get token 0
            }
        }

        // Pad context to CONTEXT_SIZE from the left with zeros
        // Example (CONTEXT_SIZE=4): seed="he said" -> [0, 0, 42, 15]
        while (context.size() < CONTEXT_SIZE) {
            context.add(0, 0);
        }
        if (context.size() > CONTEXT_SIZE) {
            context = new ArrayList<>(context.subList(context.size() - CONTEXT_SIZE, context.size()));
        }

        StringBuilder result = new StringBuilder(seed);

        for (int i = 0; i < maxLength; i++) {
            // Build input array from current context as raw token IDs.
            // The WordVectorGraph layer inside the network converts these to
            // embedding vectors automatically during the forward pass.
            double[] input = new double[CONTEXT_SIZE];
            for (int j = 0; j < CONTEXT_SIZE; j++) {
                input[j] = context.get(j);
            }

            // Get raw output scores from the network (before softmax)
            // These are NOT probabilities yet - just activation values from the last ReLU layer
            double[] output = nn.getOutputArray(input);

            // Sample using temperature-scaled softmax
            int predicted = sampleWithTemperature(output, temperature, rand);

            // Validate the prediction
            if (predicted <= 0 || predicted >= vocabSize) {
                predicted = 1; // fallback to most common word
            }

            // Convert token ID back to word
            String word = tokenizer.getTokenText(predicted);
            if (word.equals(TextToken.END)) break;

            result.append(" ").append(word);

            // Slide context window: remove oldest token, add new prediction
            // This is the "autoregressive" part - the model's output becomes its input
            context.remove(0);
            context.add(predicted);
        }

        return result.toString();
    }

    // ==================== TEMPERATURE SAMPLING ====================

    /**
     * Samples a token from the output distribution using temperature-scaled softmax.
     *
     * THE MATH:
     * 1. Subtract max value for numerical stability (prevents overflow in exp())
     * 2. Apply temperature: scaled[i] = (output[i] - max) / temperature
     * 3. Softmax: prob[i] = exp(scaled[i]) / sum(exp(scaled[j]))
     * 4. Sample from the resulting probability distribution
     *
     * Temperature effects:
     *   T -> 0: Always picks the highest-scoring token (argmax/greedy)
     *   T = 1: Samples proportional to the model's actual confidence
     *   T -> inf: Uniform random (all tokens equally likely)
     *
     * WHY SOFTMAX?
     * The network's raw outputs are arbitrary positive numbers (after ReLU).
     * Softmax converts them into a valid probability distribution:
     *   - All values between 0 and 1
     *   - All values sum to exactly 1
     *   - Higher raw values get higher probabilities (exponentially)
     *
     * @param output      Raw output values from the neural network
     * @param temperature Controls randomness (0.1=focused, 1.0=normal, 2.0=random)
     * @param rand        Random number generator
     * @return Index of the sampled token
     */
    private static int sampleWithTemperature(double[] output, double temperature, Random rand) {
        // Step 1: Find max value (for numerical stability in exp())
        // Without this, exp(large_number) could overflow to Infinity
        double maxVal = output[0];
        for (double v : output) {
            if (v > maxVal) maxVal = v;
        }

        // Step 2 & 3: Temperature-scaled softmax
        // prob[i] = exp((output[i] - max) / T) / sum(exp((output[j] - max) / T))
        double[] probs = new double[output.length];
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            probs[i] = Math.exp((output[i] - maxVal) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        // Step 4: Sample from the probability distribution
        // Generate a random number [0, 1) and walk through the cumulative
        // distribution until we pass it. This is called "roulette wheel" sampling.
        double r = rand.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.length - 1; // fallback (rounding edge case)
    }

}
