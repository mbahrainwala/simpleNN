import data.Image;
import data.ImageConverter;

import java.io.IOException;

public class AnimalClassifier {
    public static void main(String[] args) throws IOException {
        System.out.println("***Animal Classifier***");

        ImageConverter ic = new ImageConverter(128);
        Image image = ic.getImage("data/pets/cat/cat (1).jpeg", 0);
        System.out.println(image);
    }
}
