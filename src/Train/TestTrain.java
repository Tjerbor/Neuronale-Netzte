package Train;

import builder.NetworkBuilder;
import layer.FastLinearLayer;
import layer.Layer;
import loss.MSE;
import main.NeuralNetwork;

public class TestTrain {


    public static void main(String[] args) {


        String dirFpath2 = "data/selfmade_data";
        double[][][] testData = LoadOwn.getImages(dirFpath2);

        double[][] y_train = testData[1];
        double[][] x_train = testData[0];

        NeuralNetwork nn = new NeuralNetwork();

        FastLinearLayer f = new FastLinearLayer(784, 40);
        FastLinearLayer f_out = new FastLinearLayer(40, 10);

        f.setUseBiases(true);
        f.setLearningRate(0.4);
        f_out.setUseBiases(true);
        f.setLearningRate(0.4);

        String s = "";
        Layer[] ls = new Layer[]{f, f_out};

        NetworkBuilder builder = new NetworkBuilder(784);

        builder.addFastLayer(40);
        builder.addDropout(0.3);
        builder.addFastLayer(10);

        nn = builder.getModel();

        nn.setLoss(new MSE());
        nn.train(4, x_train, y_train, 0.1);


    }
}
