package main;

import layer.Activation;
import layer.Dropout;
import optimizer.Optimizer;
import utils.Matrix;

abstract public class LayerNew {

    protected Matrix output;

    protected Optimizer optimizer;
    protected LayerNew previousLayer;
    protected LayerNew nextLayer;

    protected boolean training = false;
    protected boolean useBiases = false;
    protected int iterationAt;

    protected double learningRate = 0.4;

    protected Dropout dropout;

    protected Activation act;

    protected int[] inputShape;
    protected int[] outputShape;


    public void setTraining(boolean training) {
        this.training = training;
    }

    public LayerNew getNextLayer() {
        return this.nextLayer;
    }

    public void setNextLayer(LayerNew l) {
        this.nextLayer = l;

    }

    public LayerNew getPreviousLayer() {
        return this.previousLayer;
    }

    public void setPreviousLayer(LayerNew l) {
        this.previousLayer = l;
    }

    abstract public void forward(Matrix m);

    abstract public void backward(Matrix m);

    abstract public void backward(Matrix m, double learningRate);

    abstract public Matrix getWeights();

    abstract public void setWeights(Matrix m);
    
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public int parameters() {
        return 0;
    }

    public int[] getInputShape() {
        return this.inputShape;
    }

    public void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }

    public int[] getOutputShape() {
        return this.outputShape;
    }

    public void setOutputShape(int[] outputShape) {
        this.outputShape = outputShape;
    }

    abstract public String export(); //needs to implemented.

    public void setDropout(double rate) {
        this.dropout = new Dropout(rate);
    }

    public void setDropout(int size) {
        this.dropout = new Dropout(size);
    }

    public void setActivation(Activation act) {
        this.act = act;
    }

    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }


    public void setIterationAt(int iterationAt) {
        this.iterationAt = iterationAt;
    }

    public Matrix getOutput() {
        return output;
    }

    abstract public boolean isEqual(LayerNew other);

    abstract public String summary();

}
