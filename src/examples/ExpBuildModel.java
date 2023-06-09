package examples;


import builder.NetworkBuilder;
import main.NeuralNetwork;

public class ExpBuildModel {
    public static void main(String[] args) {

        int numFilter = 16;
        int numClasses = 10;
        int[] inputShape = new int[]{28, 28, 1};
        int kernelSize = 5;
        int strides = 2; //stepSize.
        NetworkBuilder builder = new NetworkBuilder(inputShape);
        builder.addConv2D_Last(numFilter, kernelSize, strides);
        builder.addDropout(0.5);
        builder.addMaxPooling2D_Last(); //uses for standard strides2 and poolSize2
        builder.addBatchNorm();
        builder.addFlatten();
        builder.addFastLayer(numClasses);

        NeuralNetwork nn = builder.getModel();
        nn.printSummary();


    }

}
