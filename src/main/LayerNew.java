package main;

import optimizer.Optimizer;

abstract public class LayerNew {

    abstract public LayerNew getNextLayer();

    abstract public void setNextLayer(LayerNew l);

    abstract public LayerNew getPreviousLayer();

    abstract public void setPreviousLayer(LayerNew l);

    abstract public double[] forward(double[] input);

    abstract public double[][] forward(double[][] inputs);

    abstract public double[] backward(double[] input, double learningRate);

    abstract public double[][] backward(double[][] inputs, double learningRate);

    abstract public double[] backward(double[] input);

    abstract public double[][] backward(double[][] inputs);

    abstract public double[][] getWeights();

    abstract public void setWeights(double[][] weights);

    abstract public void setOptimizer(Optimizer optimizer);

    abstract public int parameters();


}
