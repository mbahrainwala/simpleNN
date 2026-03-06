package ca.behrainwala.mustafa.layer;

import ca.behrainwala.mustafa.utils.MatrixUtils;

public class ConnectedLayer extends Layer{
    private final double[][] weights;

    private final int numberInps;
    private final int numberOuts;
    private final boolean useRelu;

    public ConnectedLayer(int numberInps, int numberOuts) {
        this(numberInps, numberOuts, true);
    }

    public ConnectedLayer(int numberInps, int numberOuts, boolean useRelu) {
        this.numberInps = numberInps;
        this.numberOuts = numberOuts;
        this.useRelu = useRelu;

        weights = MatrixUtils.initializeWeights(numberInps, numberOuts);
    }

    private double[] prevInput;
    private double[] prevOutput;

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

        double[] out = new double[numberOuts];
        prevOutput = new double[numberOuts];


        for(int rows = 0; rows < numberInps; rows++){
            for(int cols = 0; cols < numberOuts; cols++){
                prevOutput[cols] += input[rows]*weights[rows][cols];
            }
        }


        for(int j = 0; j < numberOuts; j++){
            out[j] = useRelu ? relU(prevOutput[j])-bias : prevOutput[j];
        }


        return out;
    }

    double learningRate = 0.1;

    @Override
    public void backPropagate(double[] error){
        double [] backError = new double[numberInps];

        for(int rows=0; rows<numberInps; rows++) {
            for(int cols=0; cols<numberOuts; cols++) {
                double derv = useRelu ? dervRelU(prevOutput[cols]) : 1.0;
                double cost = error[cols] * derv * prevInput[rows];

                if(getPrevLayer() != null)
                    backError[rows] += error[cols] * derv * weights[rows][cols];

                weights[rows][cols] = weights[rows][cols] - (cost*learningRate);
            }
        }

        if(getPrevLayer() != null)
            getPrevLayer().backPropagate(backError);
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