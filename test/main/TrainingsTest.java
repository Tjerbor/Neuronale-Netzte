package main;

import org.junit.jupiter.api.Test;
import utils.Reader;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TrainingsTest {

    private NeuralNetwork neuralNetwork;


    @Test
    void readTrainData() throws IOException {


        double[][] x_train = Reader.getTrainDataInputs("data/training/trafficLight.csv", 3);

        double[][] y_train = Reader.getTrainDataOutputs("data/training/trafficLight.csv", 4);


        double[][] y_train_soll = new double[][]{
                {1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 1.0},
                {0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 1.0}};


        double[][] x_train_soll = new double[][]{
                {1.0, 0.0, 0.0}, {0.8, 0.0, 0.1},
                {0.99, 0.1, 0.0}, {1.1, 0.0, 0.01},
                {1.0, 1.0, 0.0}, {0.99, 1.1, 0.0},
                {1.1, 0.9, 0.0}, {1.0, 1.0, 0.1},
                {0.0, 0.0, 1.0}, {0.1, 0.1, 1.0},
                {0.0, 0.1, 1.1}, {0.1, 0.0, 1.0},
                {0.0, 1.0, 0.0}, {0.1, 1.1, 0.0},
                {0.01, 1.1, 0.1}, {0.0, 0.99, -0.01}};

        assertAll(
                () -> assertArrayEquals(y_train_soll, y_train),
                () -> assertArrayEquals(x_train_soll, x_train)
        );


    }


}
