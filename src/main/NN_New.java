package main;

public class NN_New {

    private LayerNew fristLayer;
    private LayerNew lastLayer;
    private int size;

    public double[] compute(double[] input) {
        return fristLayer.forward(input);
    }

    public double[][] compute(double[][] inputs) {
        return fristLayer.forward(inputs);
    }

    public double[] computeBackward(double[] input) {
        return lastLayer.backward(input);
    }

    public double[] computeBackward(double[] input, double learningRate) {
        return lastLayer.backward(input);
    }

    public void computeBackward(double[][] inputs) {
        lastLayer.backward(inputs);
    }

    public void computeBackward(double[][] inputs, double learningRate) {
        lastLayer.backward(inputs, learningRate);
    }

    public int size() {
        return size;
    }

    public void add(LayerNew l) {

        if (this.fristLayer == null) {
            fristLayer = l;

        } else {
            LayerNew tmp = fristLayer;
            while (tmp.getNextLayer() != null) {
                tmp = tmp.getNextLayer();


            }
            LayerNew before = tmp;
            before.setNextLayer(l);
            l.setPreviousLayer(before);
        }
        size++;
    }


}
