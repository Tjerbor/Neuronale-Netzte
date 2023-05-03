import layers.Layer;
import layers.MSE;
import utils.Reader;

import java.util.Arrays;

public class test2 {

    public static void main(String[] args) throws Exception {

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.create(new int[]{3, 3, 4}, "tanh");

        double[][] x_train = Reader.getTrainDataInputs("csv/trainData/KW16_traindata_trafficlights_classification(1).csv",
                3);

        double[][] y_train = Reader.getTrainDataOutputs("csv/trainData/KW16_traindata_trafficlights_classification(1).csv",
                4);

        neuralNetwork.loss = new MSE();
        neuralNetwork.train_single(20, x_train, y_train, 0.15);
        for (Layer l : neuralNetwork.structur) {
            System.out.println(Arrays.deepToString(l.getWeights()));

        }

    }
}
