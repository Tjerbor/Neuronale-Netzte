package examples;

import builder.NetworkBuilder;
import function.TanH;
import load.LoadModel;
import main.NeuralNetwork;

import java.io.IOException;

public class ExpCreateAndLoad {

    public static void main(String[] args) throws IOException {

        int numFilter = 8;
        int numClasses = 10;
        int[] inputShape = new int[]{28, 28, 1};
        int kernelSize = 5; //5x5 Kernel/Maske
        int strides = 1; //Schrittweite.

        NetworkBuilder builder = new NetworkBuilder(inputShape);

        builder.addConv2D_Last(numFilter, kernelSize, strides);
        builder.addDropout(0.1);
        builder.addActivationLayer(new TanH());

        builder.addMaxPooling2D_Last(2, 2);
        builder.addFlatten();

        builder.addFCL(numClasses);
        builder.addActivationLayer(new TanH());

        NeuralNetwork nn = builder.getModel();
        nn.writeModel("modelTest.csv");

        nn = LoadModel.loadModel("modelTest.csv");
        nn.printSummary();

    }
}
