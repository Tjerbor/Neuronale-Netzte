package main;

import java.io.BufferedReader;
import java.io.FileReader;

public class Mnist_reader {


    public static int limit = 50000;

    public static double[][] getTrainData_x(String fpath) throws Exception {

        int expectedSize = 784; // num of pixels 28x28 = 784

        double[][] x_train = new double[limit][expectedSize];
        double[] tmp = new double[expectedSize];
        int count = 0;
        try (BufferedReader in = new BufferedReader(new FileReader(fpath))) {
            String line;

            while ((line = in.readLine()) != null) {
                String[] values = line.split("\t");
                values = values[0].split(";");


                if (values.length != expectedSize) {
                    throw new IllegalArgumentException("got wrong x values from fpath: " + fpath);
                }


                tmp = new double[expectedSize];
                for (int i = 0; i < values.length; i++) {
                    tmp[i] = Double.parseDouble(values[i]);


                }
                x_train[count] = tmp;
                count += 1;
                if (count == limit) {
                    return x_train;
                }

            }

        } catch (Exception e) {
            System.out.println(e);
            return null;
        }

        return x_train;
    }

    public static double[][] getTrainData_y(String fpath) throws Exception {

        int expectedSize = 10; // num_classes from 0-9

        double[][] y_train = new double[limit][expectedSize];
        double[] tmp = new double[expectedSize];
        int count = 0;
        try {
            String line;


            BufferedReader br = new BufferedReader(new FileReader(fpath));


            // Condition holds true till
            // there is character in a string
            while ((line = br.readLine()) != null) {
                String[] values = line.split("\t");
                values = values[1].split(";");


                if (values.length != expectedSize) {
                    throw new IllegalArgumentException("got wrong y values from fpath: " + fpath);
                }
                tmp = new double[expectedSize];
                for (int i = 0; i < values.length; i++) {
                    tmp[i] = Double.parseDouble(values[i]);


                }
                y_train[count] = tmp;
                count += 1;
                if (count == limit - 1) {
                    return y_train;
                }


            }
        } catch (Exception e) {
            return null;
        }

        return y_train;
    }


    public static double[][][] x_train_2_batch(double[][] x_train, int batch_size) throws Exception {

        int data_size = x_train.length;

        if (data_size % batch_size != 0) {
            throw new IllegalArgumentException("Batch size musst be equally dividable");
        }


        int firstShape = data_size / batch_size;
        double[][][] x_train_out = new double[firstShape][batch_size][x_train[0].length];
        int count = 0;
        double[][] tmp = new double[batch_size][x_train[0].length];
        for (int i = 0; i < data_size; i += batch_size) {
            for (int j = 0; j < batch_size; j++) {
                x_train_out[count][j] = x_train[i + j];

            }

            count += 1;
        }


        return x_train_out;
    }

    public static double[][][] y_train_2_batch(double[][] y_train, int batch_size) throws Exception {

        int data_size = y_train.length;

        if (data_size % batch_size != 0) {
            throw new IllegalArgumentException("Batch size musst be equally dividable");
        }


        int firstShape = data_size / batch_size;
        double[][][] y_train_out = new double[firstShape][batch_size][y_train[0].length];
        int count = 0;
        double[][] tmp = new double[batch_size][y_train[0].length];
        for (int i = 0; i < data_size; i += batch_size) {
            for (int j = 0; j < batch_size; j++) {
                y_train_out[count][j] = y_train[i + j];

            }

            count += 1;
        }


        return y_train_out;
    }

}
