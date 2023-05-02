import layers.CustomActivation;
import layers.Layer;
import utils.read_utils;

import java.io.IOException;

public class extra_test {
    public static void main(String[] args) throws IOException {

        NeuralNetwork NN = new NeuralNetwork();

        NN.create(read_utils.correkt_read_weights("csv/topology/trafficLight.csv"));

        Layer[] l = NN.getModel();

        Layer cF = new CustomActivation(l[l.length - 2].biases.length, 1.5);

        l[-1] = cF;
        NN.create(l);

        //Hier kommt der rest hin


    }
}
