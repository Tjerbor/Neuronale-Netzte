package main;

import layer.Activation;
import layer.FullyConnectedLayer;
import loss.Loss;
import optimizer.Optimizer;
import utils.Array_utils;
import utils.Reader;
import utils.Utils;

import java.io.IOException;

public class NN_New {

    private LayerNew fristLayer;
    private LayerNew lastLayer;

    private Loss loss;

    private Optimizer optimizer;
    private int size;


    /**
     * This method initializes the neural network with the given topology and activation function.
     * A {@link FullyConnectedLayer} is created for each edge layer.
     */
    public void create(int[] topology, Activation function) {

        size = topology.length - 1;
        for (int i = 0; i < size; i++) {
            this.add(new FullyConnectedLayerNew(topology[i], topology[i + 1], function));
        }
    }

    public void create(int[] topology) {

        size = topology.length - 1;
        for (int i = 0; i < size; i++) {
            this.add(new FullyConnectedLayerNew(topology[i], topology[i + 1]));
        }
    }

    public void create(int[] topology, int weightGenType) {

        size = topology.length - 1;
        for (int i = 0; i < size; i++) {
            FullyConnectedLayerNew f = new FullyConnectedLayerNew(topology[i], topology[i + 1]);
            f.genWeights(weightGenType);
            this.add(f);
        }
    }

    public void create(String fpath) throws IOException {
        LayerNew[] layers = Reader.createNew(fpath);
        this.setLayers(layers);

    }

    public double[] compute(double[] input) {
        return fristLayer.forward(input);
    }

    public double[][] computeAll(double[][] inputs) {
        return fristLayer.forward(inputs);
    }

    public void computeBackward(double[] input, int epochAt) {
        if (this.lastLayer == null) {
            this.build();
        }

        lastLayer.backward(input, epochAt);
    }

    public void computeBackward(double[] input) {
        if (this.lastLayer == null) {
            this.build();
        }

        lastLayer.backward(input);
    }

    public void computeBackward(double[] input, double learningRate) {
        this.build();
        lastLayer.backward(input);
    }

    public void computeAllBackward(double[][] inputs) {
        lastLayer.backward(inputs);
    }

    public void computeAllBackward(double[][] inputs, double learningRate) {
        lastLayer.backward(inputs, learningRate);
    }

    /**
     * only needed for training.
     */
    public void build() {

        this.setTrainingsStuff();
        this.setLastLayers();


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

    public void setLoss(Loss loss) {
        this.loss = loss;
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
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

    private void setTraining() {
        LayerNew[] layers = new LayerNew[size];

        fristLayer.setTraining(true);
        layers[0] = fristLayer;

        LayerNew tmp = fristLayer;
        for (int i = 1; i < layers.length; i++) {
            tmp = tmp.getNextLayer();
            tmp.setTraining(true);
            layers[i] = tmp;

        }


    }

    /**
     * set Training for all Layers.
     * Also set Optimizer.
     * Maybe set new Strides.
     */
    private void setTrainingsStuff() {


        if (this.optimizer != null) {
            fristLayer.setOptimizer(this.optimizer);

        }
        fristLayer.setTraining(true);

        LayerNew tmp = fristLayer;
        for (int i = 1; i < size; i++) {
            tmp = tmp.getNextLayer();
            tmp.setTraining(true);
            if (this.optimizer != null) {
                tmp.setOptimizer(this.optimizer);
            }


        }


    }

    private void setLastLayers() {

        LayerNew tmp = fristLayer;
        for (int i = 1; i < size; i++) {
            tmp = tmp.getNextLayer();
        }

        lastLayer = tmp;

    }

    public LayerNew[] getLayers() {
        return layers2Array();
    }

    public void setLayers(LayerNew[] layers) {

        fristLayer = layers[0];

        for (int i = 1; i < layers.length; i++) {
            this.add(layers[i]);
        }


    }

    public void checkTraining(double[][] x_train, double[][] y_train) {


        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have different Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (this.lastLayer.getOutputShape()[0] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + lastLayer.getOutputShape()[0]);
        } else if (fristLayer.getInputShape()[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + fristLayer.getInputShape()[0]);
        }


    }


    public void test(double[][] inputDaten, double[][] classes) {

        int step_size = inputDaten.length;

        double acc = 0;
        for (int j = 0; j < step_size; j++) {
            double[] out;
            out = compute(Array_utils.copyArray(inputDaten[j]));

            if (Utils.argmax(out) == Utils.argmax(classes[j])) {
                acc += 1;
            }


        }

        System.out.println("Acc: " + acc / inputDaten.length);


    }


    public void train(int epochs, double[][] inputDaten, double[][] classes) {

        this.build();
        this.checkTraining(inputDaten, classes);


        long st = 0;
        long end;
        int step_size = inputDaten.length;
        double[] stepLosses = new double[step_size];
        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;

            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(inputDaten[j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, classes[j]);
                //calculates backward Loss
                out = loss.backward(out, classes[j]);
                // now does back propagation
                this.computeBackward(out, i);
            }


        }
        end = System.currentTimeMillis();
        System.out.println("Time: " + (end - st / 1000));
        System.out.println("Loss: " + Utils.mean(stepLosses));


    }

    public void train(int epochs, double[][] inputDaten, double[][] classes, double[][] testInput, double[][] testClasses) {
        this.build();
        this.checkTraining(inputDaten, classes);


        long st = 0;
        long end;
        int step_size = inputDaten.length;
        double[] stepLosses = new double[step_size];
        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;

            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(inputDaten[j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, classes[j]);
                //calculates backward Loss
                out = loss.backward(out, classes[j]);
                // now does back propagation
                this.computeBackward(out);
            }


        }
        end = System.currentTimeMillis();
        System.out.println("Time: " + (end - st / 1000));
        System.out.println("Loss: " + Utils.mean(stepLosses));
        test(testInput, testClasses);

    }

}
