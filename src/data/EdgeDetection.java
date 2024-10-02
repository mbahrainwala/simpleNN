package data;

/*
 * Code copied from
 * https://github.com/PacktPublishing/Java-Machine-Learning-for-Computer-Vision/tree/master/EdgeDetection
 */

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;

public class EdgeDetection {
    private static final double[][] FILTER_VERTICAL = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
    private static final double[][] FILTER_HORIZONTAL = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};

    private static final double[][] FILTER_SOBEL_V = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    private static final double[][] FILTER_SOBEL_H = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    private static final double[][] FILTER_SCHARR_V = {{3, 0, -3}, {10, 0, -10}, {3, 0, -3}};
    private static final double[][] FILTER_SCHARR_H = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};

    private final HashMap<String, double[][]> filterMap;

    public EdgeDetection() {
        filterMap = buildFilterMap();
    }

    public BufferedImage detectEdges(BufferedImage bufferedImage, EdgeFilter selectedFilter) {
        if(selectedFilter==null || EdgeFilter.NONE.equals(selectedFilter))
            return bufferedImage;

        double[][][] image = transformImageToArray(bufferedImage);
        double[][] filter = filterMap.get(selectedFilter.toString());
        double[][] convolvedPixels = applyConvolution(bufferedImage.getWidth(),
                bufferedImage.getHeight(), image, filter);
        return createImageFromConvolutionMatrix(bufferedImage, convolvedPixels);
    }

    private double[][][] transformImageToArray(BufferedImage bufferedImage) {
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();

        double[][][] image = new double[3][height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Color color = new Color(bufferedImage.getRGB(j, i));
                image[0][i][j] = color.getRed();
                image[1][i][j] = color.getGreen();
                image[2][i][j] = color.getBlue();
            }
        }
        return image;
    }

    private double[][] applyConvolution(int width, int height, double[][][] image, double[][] filter) {
        Convolution convolution = new Convolution();
        double[][] redConv = convolution.convolutionType2(image[0], width, height, filter, 3, 3, 1);
        double[][] greenConv = convolution.convolutionType2(image[1], width, height, filter, 3, 3, 1);
        double[][] blueConv = convolution.convolutionType2(image[2], width, height, filter, 3, 3, 1);
        double[][] finalConv = new double[redConv.length][redConv[0].length];
        for (int i = 0; i < redConv.length; i++) {
            for (int j = 0; j < redConv[i].length; j++) {
                finalConv[i][j] = redConv[i][j] + greenConv[i][j] + blueConv[i][j];
            }
        }
        return finalConv;
    }

    private BufferedImage createImageFromConvolutionMatrix(BufferedImage originalImage, double[][] imageRGB) {
        BufferedImage writeBackImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < imageRGB.length; i++) {
            for (int j = 0; j < imageRGB[i].length; j++) {
                Color color = new Color(fixOutOfRangeRGBValues(imageRGB[i][j]),
                        fixOutOfRangeRGBValues(imageRGB[i][j]),
                        fixOutOfRangeRGBValues(imageRGB[i][j]));
                writeBackImage.setRGB(j, i, color.getRGB());
            }
        }
        return writeBackImage;
    }

    private int fixOutOfRangeRGBValues(double value) {
        if (value < 0.0) {
            value = -value;
        }
        if (value > 255) {
            return 255;
        } else {
            return (int) value;
        }
    }

    private HashMap<String, double[][]> buildFilterMap() {
        HashMap<String, double[][]> filterMap;
        filterMap = new HashMap<>();
        filterMap.put(EdgeFilter.VERTICAL_FILTER.toString(), FILTER_VERTICAL);
        filterMap.put(EdgeFilter.HORIZONTAL_FILTER.toString(), FILTER_HORIZONTAL);

        filterMap.put(EdgeFilter.SOBEL_FILTER_VERTICAL.toString(), FILTER_SOBEL_V);
        filterMap.put(EdgeFilter.SOBEL_FILTER_HORIZONTAL.toString(), FILTER_SOBEL_H);

        filterMap.put(EdgeFilter.SCHARR_FILTER_VETICAL.toString(), FILTER_SCHARR_V);
        filterMap.put(EdgeFilter.SCHARR_FILTER_HORIZONTAL.toString(), FILTER_SCHARR_H);
        return filterMap;
    }
}
