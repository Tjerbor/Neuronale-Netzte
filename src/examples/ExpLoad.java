package examples;

import load.LoadModel;
import main.NeuralNetwork;

import java.io.IOException;

public class ExpLoad {

    public static void main(String[] args) throws IOException {


        NeuralNetwork nn = LoadModel.loadModel("fcl_emnist_letter_weights2.csv");
        nn.printSummary();

    }
}
