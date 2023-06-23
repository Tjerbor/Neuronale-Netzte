package main;

import layer.Activation;
import optimizer.Optimizer;

abstract public class LayerNew {

    abstract public <E> getWeights();

    abstract public void setTraining(boolean training);

    abstract public LayerNew getNextLayer();

    abstract public void setNextLayer(LayerNew l);

    abstract public LayerNew getPreviousLayer();

    abstract public void setPreviousLayer(LayerNew l);

    abstract public double[] forward(double[] input);

    abstract public double[][] forward(double[][] inputs);

    abstract public void backward(double[] input, double learningRate);

    abstract public void backward(double[][] inputs, double learningRate);

    abstract public void backward(double[] input);

    //abstract public double[] getWeights();

    //abstract public double[][] getWeights();

    abstract public void backward(double[][] inputs);

    abstract public void setWeights(double[][] weights);

    abstract public void setOptimizer(Optimizer optimizer);

    abstract public int parameters();

    abstract public int[] getInputShape();

    abstract public int[] getOutputShape();

    abstract public String export();

    abstract public void setEpochAt(int epochAt);

    abstract public void setDropout(double rate);

    abstract public void setDropout(int size);

    abstract public void setActivation(Activation act);

    abstract public void setUseBiases(boolean useBiases);

}
