package optimizer;

import layer.Layer;

public interface Optimizer {


    default void pre_epoch() {
    }

    default void past_epoch() {
    }

    default void update(Layer l) {
    }

    default void setIterationAt(int t) {
    }
    

    default void updateParameter(double[][][][] weights, double[][][][] deltaWeights) {
    }

    default void updateParameter(double[][][] weights, double[][][] deltaWeights) {
    }

    default void updateParameter(double[][] weights, double[][] deltaWeights) {
    }

    default void updateParameter(double[] biases, double[] deltaBiases) {
    }

    default void setLearningRate(double learningRate) {
    }

    default void updateParameter(double[][][][] weights, double[][][][] deltaWeights, int iteration) {
    }

    default void updateParameter(double[][][] weights, double[][][] deltaWeights, int iteration) {
    }

    default void updateParameter(double[][] weights, double[][] deltaWeights, int iteration) {
    }

    default void updateParameter(double[] biases, double[] deltaBiases, int iteration) {
    }

    default String export() {
        return "null;";
    }

    default void setEpochAt(int iteration) {
    }


}
