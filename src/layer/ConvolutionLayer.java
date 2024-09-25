package layer;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionLayer extends Layer{
    private final int imageX, imageY;
    private final int filterSize, stepSize;


    public ConvolutionLayer(int imageX, int imageY, int filterSize, int stepSize){
        this.imageX=imageX;
        this.imageY=imageY;
        this.filterSize=filterSize;
        this.stepSize=stepSize;
    }

    private void generateFilters(int numFilters){

    }

    @Override
    public double[] getOutput(double[] input) {
        return new double[0];
    }

    private List<double[][]> lastInput;

    private List<double[][]> convolutionForwardPass(List<double[][]> input){
        lastInput=input;
        List<double[][]> output = new ArrayList<>();

        return output;
    }

    @Override
    public void backPropagate(double[] error) {

    }

    @Override
    public int getNumberOutput() {
        return 0;
    }
}
