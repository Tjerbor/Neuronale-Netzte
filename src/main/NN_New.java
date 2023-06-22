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


    private LayerNew[] layers2Array() {
        LayerNew[] layers = new LayerNew[size];

        layers[0] = fristLayer;

        LayerNew tmp = fristLayer;
        for (int i = 1; i < layers.length; i++) {
            tmp = tmp.getNextLayer();
            layers[i] = tmp;

        }

        return layers;
    }

    public LayerNew[] getLayers() {
        return layers2Array();
    }
}
