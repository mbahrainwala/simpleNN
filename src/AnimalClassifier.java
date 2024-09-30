import data.Image;
import data.ImageConverter;
import network.NetworkBuilder;
import network.NeuralNetwork;
import utils.MatrixUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class AnimalClassifier {
    public static void main(String[] args) throws IOException {
        System.out.println("***Animal Classifier***");


        ImageConverter ic = new ImageConverter(128);
//        Image imageTest = ic.getImage("data/pets/cat/cat (1).jpeg", 0);
//        System.out.println(imageTest);

        NetworkBuilder nb = new NetworkBuilder(128, 128, 25600);
        nb.addConvolutionLayer(8, 5);
        nb.addPoolLayer(2, 1);
        nb.addConnectedLayer(1638); //
        nb.addConnectedLayer(163);
        nb.addConnectedLayer(16);
        nb.addConnectedLayer(2);
        NeuralNetwork nn = nb.build();

        int numImages = 20;

        List<Image> imagesTrain = new ArrayList<>(40);

        for(int i=1; i<=numImages; i++){
            Image image = ic.getImage("data/pets/cat/cat ("+i+").jpeg", 0);
            imagesTrain.add(image);
        }

        for(int i=1; i<=numImages; i++){
            Image image = ic.getImage("data/pets/dog/dog ("+i+").jpeg", 1);
            imagesTrain.add(image);
        }

        System.out.println("***Images Loaded***");
        System.out.println("***Commencing Training***");
        int epochs = 100;
        for (int i = 0; i < epochs; i++) {
            Collections.shuffle(imagesTrain);
            for(Image trainingImg: imagesTrain) {
                nn.train(MatrixUtils.matrixToVector(trainingImg.data()), trainingImg.label());
            }
        }
        System.out.println("***Training Complete***");

        while(true){
            System.out.print("\n\nPlease enter the image:");
            Scanner scan= new Scanner(System.in);
            String imageName= scan.nextLine();
            Image image = ic.getImage("data/pets/"+imageName, 0);
            if(nn.getOutput(MatrixUtils.matrixToVector(image.data()))==0)
                System.out.println("Its a cat");
            else
                System.out.println("Its a dog");
        }
    }
}
