package layer;

import utils.MatrixUtils;

public class MaxPoolLayer extends Layer{
    private final int _stepSize;
    private final int _windowSize;

    private final int _inRows;
    private final int _inCols;

    double[] output;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }
    @Override
    public double[] getOutput(double[] input) {
        double[][] matrixList = MatrixUtils.vectorToMatrix(input, _inRows, _inCols);
        output = MatrixUtils.matrixToVector(pool(matrixList));

        if(getNextLayer() != null)
            return getNextLayer().getOutput(output);
        else
            return output;
    }

    @Override
    public void backPropagate(double[] error) {

    }

    @Override
    public int getNumberOutput() {
        return getOutputRows() * getOutputCols();
    }

    private double[][] pool(double[][] input){
        double[][] output = new double[getOutputRows()][getOutputCols()];

        for(int r = 0; r < getOutputRows(); r+= _stepSize) {
            for (int c = 0; c < getOutputCols(); c += _stepSize) {
                double max = 0.0;
                int maxx = 0;
                int maxy = 0;

                for(int x = 0; x < _windowSize; x++) {
                    for (int y = 0; y < _windowSize; y++) {
                        if(max < input[r+x][c+y]){
                            max=input[r+x][c+y];
                            maxx = r+x;
                            maxy = c+y;
                        }
                    }
                }

                if(maxx > getOutputRows()-_stepSize)
                    maxx = getOutputRows()-_stepSize;
                if(maxy > getOutputCols()-_stepSize)
                    maxy = getOutputCols()-_stepSize;

                output[maxx][maxy] = max;
            }
        }

        return output;
    }

    public int getOutputRows() {return (_inRows-_windowSize)/_stepSize+1;}

    public int getOutputCols() {return (_inCols-_windowSize)/_stepSize+1;}

    private static final char FULL_BLOCK= '█';
    private static final char MEDIUM_SHADE= '▒';
    private static final char LIGHT_SHADE= '░';

    public String toString(){
        if(output == null)
            return "No output";

        StringBuilder sb = new StringBuilder();

        for(int i=0; i<getOutputRows(); i++) {
            for (int j = 0; j < getOutputCols(); j++) {
                if(output[i*getOutputRows()+j] != 0) {
                    if(output[i * getOutputRows() + j] > 196)
                        sb.append(FULL_BLOCK);
                    else if (output[i * getOutputRows() + j] > 100)
                        sb.append(MEDIUM_SHADE);
                    else
                        sb.append(LIGHT_SHADE);
                }
                else
                    sb.append(" "); //only print the rows which are not zero
            }
                sb.append("\n");
        }

        return sb.toString();
    }
}