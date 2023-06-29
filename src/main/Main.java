package main;

import Train.LoadOwn;
import load.LoadModel;
import utils.ImageReader;
import utils.Utils;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
//        NeuralNetwork neuralNetwork = LoadModel.loadModel("mnist.csv");
//
//
//        String dirFpath2 = "data/selfmade_data";
//        double[][][] testData = LoadOwn.getImages(dirFpath2);
//
//        double[][] y_testOwn = testData[1];
//        double[][] x_testOwn = testData[0];
//
//
//        int count = 0;
//        for (int j = 1; j < 4; j++) {
//            for (int i = 0; i < 10; i++) {
//                double[] result = neuralNetwork.compute(ImageReader.ImageToArray("data/selfmade_data/sample0" + j + "_" + i + ".png"));
//
//                System.out.println("Soll: " + i + " predicted: " + Utils.argmax(result)
//                        + " confidence: " + result[Utils.argmax(result)]);
//
//
//            }
//        }
//        System.out.println("right: ");
//
//        neuralNetwork.printTestStats(x_testOwn, y_testOwn);

    }

    public static void mnistTrain_mnistTest(){
        //TODO
    }

    public static void ImageOnTheFLyTest() throws IOException {
        //TODO
        NeuralNetwork neuralNetwork = LoadModel.loadModel(null);

        String path = "uwu.png";
        double[] testData = ImageReader.ImageToArray(path);

        double[] probabilities = neuralNetwork.compute(testData);

        System.out.println(Arrays.toString(probabilities));
    }


}
