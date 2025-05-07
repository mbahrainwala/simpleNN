package ca.behrainwala.mustafa.data;

public record Image(double[][] data, int label) {
    private static final char FULL_BLOCK= '█';
    private static final char MEDIUM_SHADE= '▒';
    private static final char LIGHT_SHADE= '░';

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("label->").append(label);
        sb.append("\n");

        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; j++) {
                if (datum[j] != 0) {
                    if(datum[j] > 196)
                        sb.append(FULL_BLOCK);
                    else if(datum[j] > 100)
                        sb.append(MEDIUM_SHADE);
                    else
                        sb.append(LIGHT_SHADE);
                } else
                    sb.append(" ");
            }
            sb.append("\n");
        }

        return sb.toString();
    }
}