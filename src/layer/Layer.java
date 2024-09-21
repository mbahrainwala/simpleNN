package layer;

public abstract class Layer {
    private Layer prevLayer;
    private Layer nextLayer;

    public void setPrevLayer(Layer prevLayer) {
        this.prevLayer = prevLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public Layer getPrevLayer() {
        return prevLayer;
    }

    public abstract double[] getOutput(double[] input);

    public abstract void backPropagate(double[] error);

    public abstract int getNumberOutput();
}
