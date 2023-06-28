package builder;

import function.Activation;
import layer.*;
import main.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static utils.Array_utils.getFlattenInputShape;


/**
 * Makes it easier for Users to create a Model, because
 * the input and Output Shape are set by the Model.
 * this Builder can return the Layer, which then can be used to create the model.
 * also has a Method to return a finished Model.
 */

public class NetworkBuilder {


    int inputSize;
    int[] inputShape;
    private List<Layer> layers = new ArrayList<>();


    public NetworkBuilder(int inputSize) {
        this.inputSize = inputSize;

    }

    public NetworkBuilder(int[] inputShape) {
        this.inputShape = inputShape;

    }

    public NetworkBuilder() {
    }

    public Layer[] getLayers() {

        Layer[] layers = new Layer[this.layers.size()];

        for (int i = 0; i < layers.length; i++) {
            layers[i] = this.layers.get(i);
        }

        return layers;
    }

    public NeuralNetwork getModel() {
        NeuralNetwork nn = new NeuralNetwork();
        nn.create(this.getLayers());
        return nn;

    }

    public void addFullyConnectedLayer(int inputSize, int outputSize, double rate) {
        this.layers.add(new FullyConnectedLayer(inputSize, outputSize));
        int size = layers.size() - 1;

        layers.get(size).setDropout(rate);
    }

    public void addFullyConnectedLayer(int NeuronSize, boolean useBiases) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getFlattenInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayer(sizeBefore, NeuronSize));

            layers.get(position).setUseBiases(useBiases);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayer(inputSize, NeuronSize));
            int position = layers.size() - 1;

            layers.get(position).setUseBiases(useBiases);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFullyConnectedLayer(int NeuronSize, double rate, Activation act, boolean useBiases) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getFlattenInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayer(sizeBefore, NeuronSize));
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);
            layers.get(position).setUseBiases(useBiases);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayer(inputSize, NeuronSize));
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
            int sizeBefore = getFlattenInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayer(sizeBefore, NeuronSize));
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayer(inputSize, NeuronSize));
            int position = layers.size() - 1;
            layers.get(position).setDropout(rate);
            layers.get(position).setActivation(act);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }


    public void addFullyConnectedLayer(int NeuronSize, double rate) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getFlattenInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayer(sizeBefore, NeuronSize));
            layers.get(position).setDropout(rate);
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayer(inputSize, NeuronSize));
            int position = layers.size() - 1;
            layers.get(position).setDropout(rate);

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFullyConnectedLayer(int NeuronSize) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getFlattenInputShape(layers.get(position - 1).getOutputShape());
            this.layers.add(new FullyConnectedLayer(sizeBefore, NeuronSize));
        } else if (this.inputSize > 0) {
            this.layers.add(new FullyConnectedLayer(inputSize, NeuronSize));
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
            this.layers.add(new Conv2D_Last(shapeBefore, numFilter));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D_Last(inputShape, numFilter));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addConv2D_Last(int[] inputShape, int numFilter) {

        if (layers.size() != 0) {
            this.layers.add(new Conv2D_Last(inputShape, numFilter));
        } else if (this.inputShape != null) {
            this.layers.add(new Conv2D_Last(inputShape, numFilter));
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

    public void addDropout(double rate) {
        if (layers.size() == 0) {
            throw new IllegalArgumentException("Dropout is meant to be used on Layer");
        } else if (rate < 1.0) {
            int position = layers.size() - 1;
            int[] shape = layers.get(position).getOutputShape();
            DropoutLayer d = new DropoutLayer(rate);
            d.setInputShape(shape);
            d.setOutputShape(shape);
            layers.add(d);
        } else {
            throw new IllegalArgumentException("Dropout rate was greater than 1, only accepts 0-0.99 values.");

        }

    }


    public void addBatchNorm() {
        if (layers.size() == 0) {
            throw new IllegalArgumentException("BatchNorm can not be set as first Layer without a InputShape.");
        } else {
            int position = layers.size() - 1;
            int[] shapeBefore = layers.get(position).getOutputShape();
            this.layers.add(new BatchNorm(shapeBefore[0]));
            this.layers.get(layers.size() - 1).setInputShape(shapeBefore);
            this.layers.get(layers.size() - 1).setOutputShape(shapeBefore);


        }
    }


    public void addBatchNorm(int[] shape, int inputSize) {
        this.layers.add(new BatchNorm(inputSize));
        this.layers.get(layers.size() - 1).setInputShape(shape);
        this.layers.get(layers.size() - 1).setOutputShape(shape);
    }


    public void addFastLayer(int NeuronSize) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getFlattenInputShape(layers.get(position).getOutputShape());
            this.layers.add(new FastLinearLayer(sizeBefore, NeuronSize));
        } else if (this.inputSize > 0) {
            this.layers.add(new FastLinearLayer(inputSize, NeuronSize));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFastLayer(int NeuronSize, Activation act) {

        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int sizeBefore = getFlattenInputShape(layers.get(position).getOutputShape());
            this.layers.add(new FastLinearLayer(sizeBefore, NeuronSize, act));
        } else if (this.inputSize > 0) {
            this.layers.add(new FastLinearLayer(inputSize, NeuronSize, act));
        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }


    }

    public void addFastLinearLayer(int inputSize, int NeuronSize) {
        this.layers.add(new FastLinearLayer(inputSize, NeuronSize));

    }

    public void addFastLinearLayer(int inputSize, int NeuronSize, Activation act) {
        this.layers.add(new FastLinearLayer(inputSize, NeuronSize, act));


    }

    public void addMaxPooling2D_Last() {
        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position).getOutputShape());
            layers.add(new MaxPooling2D_Last(shapeBefore));

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }

    }

    public void addMaxPooling2D_Last(int strides, int poolSize) {
        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position).getOutputShape());
            layers.add(new MaxPooling2D_Last(shapeBefore));

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }

    }

    public void addMaxPooling2D_Last(int[] inputShape) {
        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = (layers.get(position).getOutputShape());
            layers.add(new MaxPooling2D_Last(shapeBefore));

        } else {
            throw new IllegalArgumentException("No previous Layers are set.");
        }

    }

    public void addFlatten() {
        if (layers.size() != 0) {
            int position = layers.size() - 1;
            int[] shapeBefore = layers.get(position).getOutputShape();
            layers.add(new Flatten(shapeBefore));

        } else if (inputShape != null) {
            layers.add(new Flatten(inputShape));
        } else {
            throw new IllegalArgumentException("No previous Layers are set for Flatten.");
        }


    }

    public void addOnlyFastLayer(int[] topologie) {
        this.addFastLinearLayer(topologie[0], topologie[1]);
        for (int i = 2; i < topologie.length; i++) {
            this.addFastLayer(topologie[i]);
        }

    }

    public void addOnlyFastLayer(int[] topologie, Activation act) {
        this.addFastLinearLayer(topologie[0], topologie[1], act);
        for (int i = 2; i < topologie.length; i++) {
            this.addFastLayer(topologie[i], act);
        }

    }

    public void addOnlyFastLayer(int[] topologie, Activation[] activations) {

        if (topologie.length < 2 || activations.length == 0 || (activations.length != 2 && activations.length != topologie.length - 1)) {
            throw new IllegalArgumentException("got topologie: " + Arrays.toString(topologie) + " and Activation-functions: " + Arrays.toString(activations));
        }


        if (activations.length == topologie.length - 1) {
            this.addFastLinearLayer(topologie[0], topologie[1], activations[0]);
            for (int i = 2; i < topologie.length; i++) {
                this.addFastLayer(topologie[i], activations[i - 2]);
            }

        } else if (activations.length == 2) {
            this.addFastLinearLayer(topologie[0], topologie[1], activations[0]);
            for (int i = 2; i < topologie.length; i++) {
                if (i == topologie.length - 1) {
                    this.addFastLayer(topologie[i], activations[1]);
                } else {
                    this.addFastLayer(topologie[i], activations[0]);
                }

            }
        } else {
            throw new IllegalArgumentException("got topologie: " + Arrays.toString(topologie) + " and Activation-functions: " + Arrays.toString(activations));
        }


    }

    public void addOnlyFCL(int[] topologie) {
        this.addFullyConnectedLayer(topologie[0], topologie[1]);
        for (int i = 1; i < topologie.length; i++) {
            this.addFullyConnectedLayer(topologie[i]);
        }

    }


    public void addFlatten(int[] inputShape) {
        layers.add(new Flatten(inputShape));
    }


}
