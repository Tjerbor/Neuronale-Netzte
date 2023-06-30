package Train;

import builder.NetworkBuilder;
import loss.MSE;
import main.NeuralNetwork;

public class TestTrain {


    public static void main(String[] args) {


        String dirFpath2 = "data/selfmade_data";
        double[][][] testData = LoadOwn.getImages(dirFpath2);

        double[][] y_train = testData[1];
        double[][] x_train = testData[0];

        NeuralNetwork nn = new NeuralNetwork();


        NetworkBuilder builder = new NetworkBuilder(784);

        builder.addFastLayer(40);
        builder.addDropout(0.3);
        builder.addFastLayer(10);

        nn = builder.getModel();

        nn.setLoss(new MSE());
        nn.train(3, x_train, y_train, 0.1);
        nn.test(x_train, y_train);
        nn.writeModel("weights_selfdata.csv");


    }
}
