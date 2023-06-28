package main;

import load.LoadModel;
import utils.ImageReader;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        NeuralNetwork neuralNetwork = LoadModel.loadModel("nn_weights.txt");

        double[] result = neuralNetwork.compute(ImageReader.ImageToArray("data/selfmade_data/sample01_0.png"));

        System.out.println(Arrays.toString(result));
    }
}
