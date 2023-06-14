package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class DataLoader {


    static int lastPos;
    static int PIXELS = 784;
    static int Classes = 47;

    public static float[][][] read(String path, int start, int size) throws IOException {
        float[][] pixels = new float[size][PIXELS];
        float[][] digits = new float[size][Classes];

        int i = 0;

        try (BufferedReader in = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = in.readLine()) != null) {


                if (i < start) {
                    i++;
                    continue;
                }
                String[] data = line.split("\t");

                String[] x = data[0].split(";");
                String[] y = data[1].split(";");

                if (x.length != PIXELS || y.length != Classes) {
                    throw new IllegalArgumentException("The file " + path + " does not conform to the EMNIST format.");
                }

                pixels[i] = new float[PIXELS];
                digits[i] = new float[Classes];

                for (int j = 0; j < x.length; j++) {
                    pixels[i][j] = Float.parseFloat(x[j]);
                }

                for (int j = 0; j < y.length; j++) {
                    digits[i][j] = Float.parseFloat(y[j]);
                }

                if (++i == size) {
                    lastPos += size;
                    break;
                }
            }
        }

        return new float[][][]{pixels, digits};
    }

    public float[][][] getData(String filepath, int size, int maxSize) throws IOException {
        return read(filepath, lastPos, maxSize);


    }


}
