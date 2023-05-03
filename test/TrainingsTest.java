import org.junit.jupiter.api.Test;
import utils.Reader;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TrainingsTest {

    private NeuralNetwork neuralNetwork;


    @Test
    void readTrainData() throws IOException {


        double[][] x_train = Reader.getTrainDataInputs("csv/trainData/KW16_traindata_trafficlights_classification(1).csv", 3);

        double[][] y_train = Reader.getTrainDataOutputs("csv/trainData/KW16_traindata_trafficlights_classification(1).csv", 4);


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
                () -> assertEquals(y_train_soll, y_train),
                () -> assertEquals(x_train_soll, x_train)
        );


    }


}
