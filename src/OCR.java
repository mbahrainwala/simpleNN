import data.DataReader;
import data.Image;
import layer.MaxPoolLayer;
import network.NetworkBuilder;
import network.NeuralNetwork;
import utils.MatrixUtils;

import java.util.Collections;
import java.util.List;

public class OCR {
    public static void main(String[] args) {
        System.out.println("***OCR***");
        System.out.println("\n\n***Starting***\n...loading data...\n");

        List<Image> imagesTest = DataReader.readData("data/mnist_test.csv");
        System.out.println("***Test Data Loaded***");
        System.out.println("Images in test data: "+imagesTest.size());

        List<Image> imagesTrain = DataReader.readData("data/mnist_train.csv");
        System.out.println("***Train Data Loaded***");
        System.out.println("Images in train data: "+imagesTrain.size());

        int testImg=16;
        System.out.println("\n\n***Image***\n\n"+imagesTest.get(testImg)+"\n\n");
        MaxPoolLayer player = new MaxPoolLayer(1, 2, 28, 28);
        player.getOutput(MatrixUtils.matrixToVector(imagesTest.get(testImg).data()));
        System.out.println("\n\n***MaxPoolLayer***\n\n"+player+"\n\n");

        NetworkBuilder nb = new NetworkBuilder(28, 28, 25600);
        nb.addConvolutionLayer(3,1);
        nb.addPoolLayer(2, 1);
        nb.addConnectedLayer(10);
        NeuralNetwork nn = nb.build();

        int countCorrect=0;
        for(Image image: imagesTest){
            int ans = nn.getOutput(MatrixUtils.matrixToVector(image.data()));
            if(ans == image.label()){
                countCorrect++;
            }
        }
        System.out.println("\n\n***Accuracy*** -> "+(countCorrect*100.0/imagesTest.size())+"%");

        for(int epoc=0; epoc<10000; epoc++){
            Collections.shuffle(imagesTrain);
            for(Image image: imagesTrain){
                nn.train(MatrixUtils.matrixToVector(image.data()), image.label());
            }

            countCorrect=0;
            for(Image image: imagesTest){
                int ans = nn.getOutput(MatrixUtils.matrixToVector(image.data()));
                if(ans == image.label()){
                    countCorrect++;
                }
            }
            System.out.println("***Accuracy after epoc "+(epoc+1)+"*** -> "+(countCorrect*100.0/imagesTest.size())+"%");
            if(countCorrect*100.0/imagesTest.size()>91)
                break; //do not train the model to have no flexibility.
        }
    }
}