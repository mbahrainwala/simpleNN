package ca.behrainwala.mustafa;

import ca.behrainwala.mustafa.layer.ConnectedLayer;
import ca.behrainwala.mustafa.layer.Layer;
import ca.behrainwala.mustafa.network.NetworkBuilder;
import ca.behrainwala.mustafa.network.NeuralNetwork;
import ca.behrainwala.mustafa.utils.MatrixUtils;

public class SimpleNN {
    public static void main(String[] args) {
        System.out.println("Simple NN training.");

        SimpleNN.testOR();
        SimpleNN.testAdultORChild();
        SimpleNN.testAdultORChildGender();
        SimpleNN.testXOR();
    }

    private static final double ZERO=.1;
    private static final double ONE=.9;

    private static void testOR() {
        System.out.println("Testing OR");

        double[][] train = {{ZERO,ONE}, {ONE,ONE}, {ONE,ZERO}, {ZERO,ZERO}};
        int[] out = {1, 1, 1, 0};

        double[][] test = {{ZERO, ZERO}, {ZERO, ONE}, {ONE, ZERO}, {ONE, ONE}};

        NetworkBuilder nb = new NetworkBuilder(2, 1);
        nb.addConnectedLayer(3);
        nb.addConnectedLayer(2);
        NeuralNetwork nn = nb.build();

        System.out.println("Before training");
        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+"-->"+nn.getOutput(doubles));

        for(int epoch=0; epoch<100000; epoch++) {
            for (int i = 0; i < train.length; i++) {
                nn.train(train[i], out[i]);
            }
        }

        System.out.println("After training");

        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+"-->"+nn.getOutput(doubles));
    }

    private static void testAdultORChild() {
        System.out.println("\n\nTesting Adult[1] OR Child[0] Height/Weight");

        double[][] train = {
                {.170,.70}, //a
                {.165,.65}, //a
                {.130,.40}, //c
                {.120,.30}, //c
                {.080,.20}, //c
                {.180,.90} //a
        };
        double[][] out = {
                {0,1}, {0,1}, {1,0}, {1,0}, {1,0}, {0,1}
        };

        double[][] test = {
                {.176, .82},
                {.095, .30},
                {.185, .90}
        };

        Layer nn = new ConnectedLayer(2, 2);

        System.out.println("Before training");
        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+"-->"+MatrixUtils.getMaxIndex(nn.getOutput(doubles)));

        for(int epoch=0; epoch<100000; epoch++) {
            for (int i = 0; i < train.length; i++) {
                double[] output = nn.getOutput(train[i]);
                double[] error = MatrixUtils.addArrays(output, MatrixUtils.multiplyScalar(out[i], -1));
                nn.backPropagate(error);
            }
        }

        System.out.println("After training");

        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+"-->"+MatrixUtils.getMaxIndex(nn.getOutput(doubles)));
    }

    private static void testAdultORChildGender() {
        System.out.println("\n\nTesting Adult[1] OR Child[0] with Gender - Height/Weight/Sex M[0.9] F[0.0]");

        double[][] train = {
                {.170,.70, .9}, //a
                {.165,.65, .9}, //a
                {.150,.65, 0}, //a
                {.140,.60, 0}, //a
                {.150,.65, .9}, //c
                {.140,.60, .9}, //c
                {.130,.40, 0}, //c
                {.130,.40, .9}, //c
                {.120,.30, 0}, //c
                {.120,.30, .9}, //c
                {.080,.20, .9}, //c
                {.080,.20, 0}, //c
                {.180,.90, 0}, //a
                {.180,.90, .9} //a
        };
        double[][] out = {
                {0,1}, {0,1},{0,1}, {0,1},
                {1,0}, {1,0}, {1,0}, {1,1},
                {1,0}, {1,0}, {1,0}, {1,1},
                {0,1}, {0,1}
        };

        double[][] test = {
                {.176, .82, .9},
                {.095, .30, .9},
                {.185, .90, .9},
                {.148, .62, 0}
        };

        Layer nn = new ConnectedLayer(3, 2);

        System.out.println("Before training");
        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+", "+doubles[2]+"-->"+MatrixUtils.getMaxIndex(nn.getOutput(doubles)));

        for(int epoch=0; epoch<100000; epoch++) {
            for (int i = 0; i < train.length; i++) {
                double[] output = nn.getOutput(train[i]);
                double[] error = MatrixUtils.addArrays(output, MatrixUtils.multiplyScalar(out[i], -1));
                nn.backPropagate(error);
            }
        }

        System.out.println("After training");

        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+", "+doubles[2]+"-->"+MatrixUtils.getMaxIndex(nn.getOutput(doubles)));
    }

    private static void testXOR() {
        System.out.println("\n\nTesting XOR");

        double[][] train = {{ZERO,ONE}, {ONE,ONE}, {ONE,ZERO}, {ZERO,ZERO}};
        int[] out = {1, 0, 1, 0};

        double[][] test = {{ZERO, ZERO}, {ZERO, ONE}, {ONE, ZERO}, {ONE, ONE}};

        NetworkBuilder nb = new NetworkBuilder(2, 1);
        nb.addConnectedLayer(3);
        nb.addConnectedLayer(2);
        NeuralNetwork nn = nb.build();

        System.out.println("Before training");
        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+"-->"+nn.getOutput(doubles));

        for(int epoch=0; epoch<100000; epoch++) {
            for (int i = 0; i < train.length; i++) {
                nn.train(train[i], out[i]);
            }
        }

        System.out.println("After training");

        for (double[] doubles : test) System.out.println(doubles[0]+", "+doubles[1]+"-->"+nn.getOutput(doubles));
    }
}