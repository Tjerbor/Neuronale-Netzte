package Train;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class LoadOwn {

    static int PIXELS = 784;
    static int DIGITS = 10;

    public static void main(String[] args) {


        String dirFpath = "/home/dblade/Documents/Neuronale-Netzte/src/Train/OwnData";
        double[][][] data = getTestData(dirFpath);


    }

    public static double[][][] getTestData(String dirFpath) {


        double[] pixels;
        double[] digits;
        List<String> files = load(dirFpath);
        System.out.println("files: " + files);

        double[][] X = new double[files.size()][];
        double[][] Y = new double[files.size()][];

        for (int f = 0; f < files.size(); f++) {

            pixels = new double[PIXELS];
            digits = new double[DIGITS];

            try (BufferedReader in = new BufferedReader(new FileReader(files.get(f)))) {
                String line;

                while ((line = in.readLine()) != null) {
                    String[] data = line.split(";");

                    if (data.length != PIXELS) {
                        throw new IllegalArgumentException("The file " + files.get(f) + " does not conform to the MNIST format.");
                    }


                    for (int j = 0; j < data.length; j++) {
                        pixels[j] = Double.parseDouble(data[j]);
                    }


                    String filename = files.get(f);
                    String[] p = filename.split("_");
                    filename = p[1].replace(".txt", "");

                    digits[Integer.parseInt(filename)] = 1;

                }
                X[f] = pixels;
                Y[f] = digits;

            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        return new double[][][]{X, Y};

    }

    public static List load(String dirFpath) {
        File folder = new File(dirFpath);
        File[] listOfFiles = folder.listFiles();


        List l = new ArrayList();


        for (File file : listOfFiles) {
            if (file.isFile()) {
                l.add(file.toString());

            }
        }

        return l;
    }

    public static double[][] loadSingle(String fpath) {

        double[][] out;
        double[] pixels = new double[PIXELS];
        double[] digits = new double[DIGITS];

        try (BufferedReader in = new BufferedReader(new FileReader(fpath))) {
            String line;

            while ((line = in.readLine()) != null) {
                String[] data = line.split(";");

                if (data.length != PIXELS) {
                    throw new IllegalArgumentException("The file " + fpath + " does not conform to the MNIST format.");
                }


                for (int j = 0; j < data.length; j++) {
                    pixels[j] = Double.parseDouble(data[j]);
                }


                String filename = fpath;
                String[] p = filename.split("_");
                filename = p[1].replace(".txt", "");

                digits[Integer.parseInt(filename)] = 1;

            }

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        out = new double[][]{pixels, digits};
        return out;
    }

}
