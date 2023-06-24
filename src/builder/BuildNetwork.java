package builder;

import extraLayer.BatchNorm;
import extraLayer.Conv2D;
import extraLayer.Conv2D_Last;
import layer.Activation;
import main.FullyConnectedLayerNew;
import main.LayerNew;

import java.util.ArrayList;
import java.util.List;

public class BuildNetwork {


    List<LayerNew> layers = new ArrayList<>();
    int inputSize;

    int[] inputShape;


    public BuildNetwork(int inputSize) {
        this.inputSize = inputSize;

    }

    public BuildNetwork(int[] inputShape) {
        this.inputShape = inputShape;

    }

    public BuildNetwork() {

    }

    public void addFullyConnectedLayer(int inputSize, int outputSize, double rate) {
        this.layers.add(new FullyConnectedLayerNew(inputSize, outputSize));
        int size = layers.size() - 1;

        layers.get(size).setDropout(rate);
    }

    public void addFullyConnectedLayer(int NeuronSize, boolean useBiases) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayerNew(sizeBefore, NeuronSize));

            layers.get(position).setUseBiases(useBiases);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayerNew(inputSize, NeuronSize));
            int position = layers.size() - 1;

            layers.get(position).setUseBiases(useBiases);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFullyConnectedLayer(int NeuronSize, double rate, Activation act, boolean useBiases) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayerNew(sizeBefore, NeuronSize));
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);
            layers.get(position).setUseBiases(useBiases);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayerNew(inputSize, NeuronSize));
            int position = layers.size() - 1;
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);
            layers.get(position).setUseBiases(useBiases);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFullyConnectedLayer(int NeuronSize, double rate, Activation act) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayerNew(sizeBefore, NeuronSize));
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayerNew(inputSize, NeuronSize));
            int position = layers.size() - 1;
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public int getInputShape(int[] shape) {

        int sum = shape[0];
        for (int i = 1; i < shape.length; i++) {
            sum *= shape[i];
        }

        return sum;
    }

    public void addFullyConnectedLayer(int NeuronSize, double rate) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayerNew(sizeBefore, NeuronSize));
            layers.get(position).setDropout(rate);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayerNew(inputSize, NeuronSize));
            int position = layers.size() - 1;
            layers.get(position).setDropout(rate);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFullyConnectedLayer(int NeuronSize) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayerNew(sizeBefore, NeuronSize));
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayerNew(inputSize, NeuronSize));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addConv2D(int numFilter) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position - 1).getOutputShape());
            this.layers.add(new Conv2D(shapeBefore, numFilter));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D(inputShape, numFilter));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addConv2D(int[] inputShape, int numFilter) {

        if (layers.size() != 0) {
            this.layers.add(new Conv2D(inputShape, numFilter));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D(inputShape, numFilter));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }
    }

    public void addConv2D(int numFilter, int kernelSize, int strideSize) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position - 1).getOutputShape());
            this.layers.add(new Conv2D(shapeBefore, numFilter, kernelSize, strideSize));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D(inputShape, numFilter, kernelSize, strideSize));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addConv2D_Last(int numFilter) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position - 1).getOutputShape());
            this.layers.add(new Conv2D_Last(numFilter, shapeBefore));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D_Last(numFilter, inputShape));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addConv2D_Last(int[] inputShape, int numFilter) {

        if (layers.size() != 0) {
            this.layers.add(new Conv2D_Last(numFilter, inputShape));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D_Last(numFilter, inputShape));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }
    }

    public void addConv2D_Last(int numFilter, int kernelSize, int strideSize) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position).getOutputShape());
            this.layers.add(new Conv2D_Last(numFilter, shapeBefore, kernelSize, strideSize));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D_Last(numFilter, inputShape, kernelSize, strideSize));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addBatchNorm() {
        if (layers.size() == 0) {
            throw new IllegalArgumentException("BatchNorm can not be set as first Layer without a InputShape.");
        } else {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position).getOutputShape());
            this.layers.add(new BatchNorm(shapeBefore[0]));

        }
    }


    public void addBatchNorm(int inputSize, int[] shape) {
        this.layers.add(new BatchNorm(inputSize));
        this.layers.get(layers.size() - 1).setInputShape(shape);
    }

    public LayerNew[] getLayers() {
        return (LayerNew[]) layers.toArray();
    }
}
