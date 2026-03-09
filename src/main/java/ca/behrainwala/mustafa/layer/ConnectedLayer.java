package ca.behrainwala.mustafa.layer;

import ca.behrainwala.mustafa.utils.MatrixUtils;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ConnectedLayer extends Layer{
    private final double[][] weights;

    private final int numberInps;
    private final int numberOuts;
    private final boolean useRelu;

    /** Shared thread pool for all ConnectedLayer instances. Daemon threads so they don't block JVM shutdown. */
    private static final int THREAD_COUNT = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final ExecutorService POOL = Executors.newFixedThreadPool(THREAD_COUNT, r -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    });

    /** Minimum numberInps to justify thread overhead. Small layers run single-threaded. */
    private static final int PARALLEL_THRESHOLD = 256;

    /** Whether this layer should use multi-threaded computation */
    private final boolean useParallel;

    /** Per-thread partial output accumulators for forward pass (only allocated if useParallel) */
    private final double[][] threadPartialOutputs;

    /** Per-thread partial backError accumulators (only allocated if useParallel and needed) */
    private final double[][] threadPartialBackErrors;

    public ConnectedLayer(int numberInps, int numberOuts) {
        this(numberInps, numberOuts, true);
    }

    public ConnectedLayer(int numberInps, int numberOuts, boolean useRelu) {
        this(numberInps, numberOuts, useRelu, 123, 0.1);
    }

    public ConnectedLayer(int numberInps, int numberOuts, boolean useRelu, long seed, double learningRate) {
        this.numberInps = numberInps;
        this.numberOuts = numberOuts;
        this.useRelu = useRelu;
        this.learningRate = learningRate;

        weights = MatrixUtils.initializeWeights(numberInps, numberOuts, seed);

        // Pre-allocate arrays reused every forward/backward pass
        prevOutput = new double[numberOuts];
        forwardResult = new double[numberOuts];
        backError = new double[numberInps];
        errorDerv = new double[numberOuts];

        useParallel = numberInps >= PARALLEL_THRESHOLD && THREAD_COUNT > 1;
        if (useParallel) {
            threadPartialOutputs = new double[THREAD_COUNT][numberOuts];
            threadPartialBackErrors = new double[THREAD_COUNT][numberInps];
        } else {
            threadPartialOutputs = null;
            threadPartialBackErrors = null;
        }
    }

    private double[] prevInput;
    private final double[] prevOutput;
    private final double[] forwardResult;
    private final double[] backError;
    /** Pre-allocated: error[col] * derivative, hoisted out of the rows loop */
    private final double[] errorDerv;

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = forwardPass(input);

        if(getNextLayer() != null)
            return getNextLayer().getOutput(forwardPass);
        else
            return forwardPass;
    }

    private static final double bias = 0.01;
    public double[] forwardPass(double[] input) {
        prevInput = input;

        if (useParallel) {
            forwardPassParallel(input);
        } else {
            forwardPassSingle(input);
        }

        for(int j = 0; j < numberOuts; j++){
            forwardResult[j] = useRelu ? relU(prevOutput[j])-bias : prevOutput[j];
        }

        return forwardResult;
    }

    private void forwardPassSingle(double[] input) {
        Arrays.fill(prevOutput, 0);
        for(int rows = 0; rows < numberInps; rows++){
            for(int cols = 0; cols < numberOuts; cols++){
                prevOutput[cols] += input[rows]*weights[rows][cols];
            }
        }
    }

    private void forwardPassParallel(double[] input) {
        int rowsPerThread = numberInps / THREAD_COUNT;
        CountDownLatch latch = new CountDownLatch(THREAD_COUNT);

        for (int t = 0; t < THREAD_COUNT; t++) {
            final int threadIdx = t;
            final int startRow = t * rowsPerThread;
            final int endRow = (t == THREAD_COUNT - 1) ? numberInps : startRow + rowsPerThread;
            final double[] partial = threadPartialOutputs[threadIdx];

            POOL.execute(() -> {
                Arrays.fill(partial, 0);
                for (int rows = startRow; rows < endRow; rows++) {
                    double inp = input[rows];
                    double[] weightRow = weights[rows];
                    for (int cols = 0; cols < numberOuts; cols++) {
                        partial[cols] += inp * weightRow[cols];
                    }
                }
                latch.countDown();
            });
        }

        try { latch.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

        // Merge partial results
        System.arraycopy(threadPartialOutputs[0], 0, prevOutput, 0, numberOuts);
        for (int t = 1; t < THREAD_COUNT; t++) {
            double[] partial = threadPartialOutputs[t];
            for (int cols = 0; cols < numberOuts; cols++) {
                prevOutput[cols] += partial[cols];
            }
        }
    }

    double learningRate;

    public void setLearningRate(double lr) { this.learningRate = lr; }
    public double getLearningRate() { return learningRate; }

    @Override
    public void backPropagate(double[] error){
        // Pre-compute error[col] * derivative once per column (was recomputed numberInps times each)
        for(int cols=0; cols<numberOuts; cols++) {
            double derv = useRelu ? dervRelU(prevOutput[cols]) : 1.0;
            errorDerv[cols] = error[cols] * derv;
        }

        boolean propagateBack = getPrevLayer() != null && getPrevLayer().needsBackpropError();

        if (useParallel) {
            backPropagateParallel(propagateBack);
        } else {
            backPropagateSingle(propagateBack);
        }

        if (propagateBack) {
            getPrevLayer().backPropagate(backError);
        }
    }

    private void backPropagateSingle(boolean propagateBack) {
        if (propagateBack) {
            Arrays.fill(backError, 0);
            for(int rows=0; rows<numberInps; rows++) {
                double lrInput = prevInput[rows] * learningRate;
                double[] weightRow = weights[rows];
                for(int cols=0; cols<numberOuts; cols++) {
                    double ed = errorDerv[cols];
                    backError[rows] += ed * weightRow[cols];
                    weightRow[cols] -= ed * lrInput;
                }
            }
        } else {
            for(int rows=0; rows<numberInps; rows++) {
                double lrInput = prevInput[rows] * learningRate;
                double[] weightRow = weights[rows];
                for(int cols=0; cols<numberOuts; cols++) {
                    weightRow[cols] -= errorDerv[cols] * lrInput;
                }
            }
        }
    }

    private void backPropagateParallel(boolean propagateBack) {
        int rowsPerThread = numberInps / THREAD_COUNT;
        CountDownLatch latch = new CountDownLatch(THREAD_COUNT);

        for (int t = 0; t < THREAD_COUNT; t++) {
            final int threadIdx = t;
            final int startRow = t * rowsPerThread;
            final int endRow = (t == THREAD_COUNT - 1) ? numberInps : startRow + rowsPerThread;

            POOL.execute(() -> {
                if (propagateBack) {
                    double[] partialBack = threadPartialBackErrors[threadIdx];
                    for (int i = startRow; i < endRow; i++) partialBack[i] = 0;

                    for (int rows = startRow; rows < endRow; rows++) {
                        double lrInput = prevInput[rows] * learningRate;
                        double[] weightRow = weights[rows];
                        for (int cols = 0; cols < numberOuts; cols++) {
                            double ed = errorDerv[cols];
                            partialBack[rows] += ed * weightRow[cols];
                            weightRow[cols] -= ed * lrInput;
                        }
                    }
                } else {
                    for (int rows = startRow; rows < endRow; rows++) {
                        double lrInput = prevInput[rows] * learningRate;
                        double[] weightRow = weights[rows];
                        for (int cols = 0; cols < numberOuts; cols++) {
                            weightRow[cols] -= errorDerv[cols] * lrInput;
                        }
                    }
                }
                latch.countDown();
            });
        }

        try { latch.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

        if (propagateBack) {
            // Each thread wrote to non-overlapping row ranges, so just copy into backError
            for (int t = 0; t < THREAD_COUNT; t++) {
                int startRow = t * rowsPerThread;
                int endRow = (t == THREAD_COUNT - 1) ? numberInps : startRow + rowsPerThread;
                System.arraycopy(threadPartialBackErrors[t], startRow, backError, startRow, endRow - startRow);
            }
        }
    }

    private double relU(double x) {
        return x > 0 ? x : 0;
    }

    double leak = 0.01;
    private double dervRelU(double x) {
        return x > 0 ? 1 : leak;
    }

    @Override
    public int getNumberOutput() {
        return numberOuts;
    }

    @Override
    public int getOutputRows() { return (int)Math.sqrt(numberOuts);}

    @Override
    public int getOutputCols() { return numberOuts/getOutputRows();}

    private double[][] savedWeights;

    @Override
    public void saveWeights() {
        savedWeights = new double[numberInps][numberOuts];
        for (int i = 0; i < numberInps; i++) {
            System.arraycopy(weights[i], 0, savedWeights[i], 0, numberOuts);
        }
    }

    @Override
    public void restoreWeights() {
        if (savedWeights != null) {
            for (int i = 0; i < numberInps; i++) {
                System.arraycopy(savedWeights[i], 0, weights[i], 0, numberOuts);
            }
        }
    }
}