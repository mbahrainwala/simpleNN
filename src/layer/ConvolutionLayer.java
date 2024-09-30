package layer;

import utils.MatrixUtils;

import java.util.Random;

public class ConvolutionLayer extends Layer{
    private final int imageX, imageY;
    private final int filterSize, stepSize;
    private final int seed;

    private double[][] filter;

    public ConvolutionLayer(int imageX, int imageY, int filterSize, int stepSize, int seed){
        this.imageX=imageX;
        this.imageY=imageY;
        this.filterSize=filterSize;
        this.stepSize=stepSize;
        this.seed=seed;
        filter = new double[filterSize][filterSize];

        randomizeFilters();
    }

    private void randomizeFilters(){
        Random rand = new Random(seed);

        for(int i=0; i<filterSize; i++){
            for(int j=0; j<filterSize; j++){
                filter[i][j] = rand.nextGaussian();
            }
        }
    }

    @Override
    public double[] getOutput(double[] input) {
        if(getNextLayer()!=null)
            return getNextLayer().getOutput(MatrixUtils.matrixToVector(
                    convolutionForwardPass(MatrixUtils.vectorToMatrix(input, imageX, imageY))
                ));
        else
            throw new RuntimeException("This cannot be the last layer");
    }

    private double[][] lastInput;

    private double[][] convolutionForwardPass(double[][] input){
        lastInput=input;

        return convolve(input, filter, stepSize);
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize){
        int outRows = (input.length - filter.length)/stepSize+1;
        int outCols = (input[0].length - filter[0].length)/stepSize+1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow=0;

        for(int i=0; i<= inRows - fRows; i+=stepSize){
            int outCol=0;
            for(int j=0; j<= inCols - fCols; j+=stepSize) {
                double sum = 0;

                //Apply the filter around this position
                for(int x=0; x<fRows; x++){
                    for(int y=0; y<fCols; y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        double value = filter[x][y]*input[inputRowIndex][inputColIndex];
                        sum+=value;
                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }

    private final double learningRate=0.01;

    @Override
    public void backPropagate(double[] error) {
        double[][] filtersDelta = new double[filterSize][filterSize];

        double[][] errorMatrix = MatrixUtils.vectorToMatrix(error, getOutputRows(), getOutputCols());

        double[][] spacedError = spaceArray(errorMatrix);
        double[][] dLdF= convolve(lastInput, spacedError, 1);
        double[][] delta = MatrixUtils.multiplyScalar(dLdF, learningRate*-1);
        filtersDelta = MatrixUtils.add(filtersDelta, delta);

        filter = MatrixUtils.add(filtersDelta, filter);
    }

    @Override
    public int getNumberOutput() {
        return getOutputRows()*getOutputCols();
    }

    @Override
    public int getOutputRows() {
        return (imageX-filterSize)/stepSize+1;
    }

    @Override
    public int getOutputCols() {
        return (imageY-filterSize)/stepSize+1;
    }

    /**
     * This will create a spaced out matrix based on our input.
     */
    private double[][] spaceArray(double[][] input){
        if(stepSize < 2){
            return input;
        }

        int outRows = (input.length - 1)* stepSize + 1;
        int outCols = (input[0].length -1)*stepSize+1;

        double[][] output = new double[outRows][outCols];

        for(int i = 0; i < input.length; i++){
            for(int j = 0; j < input[0].length; j++){
                output[i*stepSize][j*stepSize] = input[i][j];
            }
        }

        return output;
    }
}
