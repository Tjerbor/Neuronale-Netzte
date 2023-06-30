package utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageReader {

    public static double[] ImageToArray(String filepath) {
        try {
            return Array_utils.flatten(convertRGBImageToGrayscaleArray(ImageIO.read(new File(filepath))));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static double[][] convertRGBImageToGrayscaleArray(BufferedImage image) {
        double[][] grayscale = new double[image.getWidth()][image.getHeight()];

        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                int rgb = image.getRGB(x, y);
                grayscale[y][x] = 0.299 * (double) ((rgb & 0x00ff0000) >> 16) + //Red
                        0.587 * (double) ((rgb & 0x0000ff00) >> 8) + //Green
                        0.114 * (double) (rgb & 0x000000ff); //Blue

                grayscale[y][x] /= 255;
            }
        }
        return grayscale;
    }


}
