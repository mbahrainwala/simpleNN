package layer;

import utils.MatrixUtils;

public class ConnectedLayer extends Layer{
    private final double[][] weights;

    private final int numberInps;
    private final int numberOuts;

    public ConnectedLayer(int numberInps, int numberOuts) {
        this.numberInps = numberInps;
        this.numberOuts = numberOuts;

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
            out[j] = relU(prevOutput[j]);
        }


        return out;
    }

    double learningRate = 0.1;

    @Override
    public void backPropagate(double[] error){
        double [] backError = new double[numberInps];

        for(int rows=0; rows<numberInps; rows++) {
            for(int cols=0; cols<numberOuts; cols++) {
                double cost = error[cols] * dervRelU(prevOutput[cols])*prevInput[rows];

                if(getPrevLayer() != null)
                    backError[rows] += error[cols] * dervRelU(prevOutput[cols]) * weights[rows][cols];

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
}