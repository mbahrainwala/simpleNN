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

        NetworkBuilder nb = new NetworkBuilder(128, 128, 25600);
        nb.addConvolutionLayer(16, 8);
        nb.addPoolLayer(2, 1);
        nb.addConnectedLayer(160);
        nb.addConnectedLayer(2);
        NeuralNetwork nn = nb.build();

        int numImages = 21;

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
        int epochs = 10000;

        double rate=0.0d;
        for (int i = 0; i < epochs; i++) {
            Collections.shuffle(imagesTrain);
            for(Image trainingImg: imagesTrain) {
                nn.train(MatrixUtils.matrixToVector(trainingImg.data()), trainingImg.label());
            }

            int correctCtr=0;
            for(Image trainingImg: imagesTrain) {
                int out = nn.getOutput(MatrixUtils.matrixToVector(trainingImg.data()));
                if (out == trainingImg.label())
                    correctCtr++;

                rate = (double) correctCtr / imagesTrain.size();
            }

            System.out.println("Rate after epoc "+i+" = "+rate);
            if(rate >= .975)
                break;//do not train the model to have no flexibility.
        }
        System.out.println("***Training Complete***");


        while(true){
            System.out.print("\n\nPlease enter the image name in relation to data\\pets.\ne.g. cat\\catT1 or dog\\dogT1\n'exit' to leave: ");
            Scanner scan= new Scanner(System.in);
            String imageName= scan.nextLine();
            if("exit".equalsIgnoreCase(imageName))
                break;

            try {
                Image image = ic.getImage("data/pets/" + imageName + ".jpeg", 0);
                if (nn.getOutput(MatrixUtils.matrixToVector(image.data())) == 0)
                    System.out.println("Its a cat");
                else
                    System.out.println("Its a dog");
                System.out.println("Is this correct y/n");
                scan= new Scanner(System.in);
                if("n".equalsIgnoreCase(scan.nextLine())) {
                    int corAns = nn.getOutput(MatrixUtils.matrixToVector(image.data())) == 0 ? 1 : 0;
                    while(nn.getOutput(MatrixUtils.matrixToVector(image.data()))!=corAns){
                        nn.train(MatrixUtils.matrixToVector(image.data()), corAns);
                        }
                }
            }catch(Exception e){
                System.out.println("unable to load file. Please reenter...");
            }
        }
    }
}
