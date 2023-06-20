package main;

import layer.Activation;
import layer.FullyConnectedLayer;
import layer.Losses;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

/**
 * This class models a fully connected feed-forward artificial neural network.
 */
public class NeuralNetwork {
    Losses loss = null; //loss function of our NN. // now only MSE is available.
    SGD optimizer = null; // right now not really supported.

    /**
     * This variable contains the layers of the neural network.
     * It does not correspond to the topology used to create the neural network.
     *
     * @see NeuralNetwork#create(int[], Activation)
     */
    private FullyConnectedLayer[] layers;

    /**
     * This method initializes the neural network with the given topology and activation function.
     * A {@link FullyConnectedLayer} is created for each edge layer.
     */
    public void create(int[] topology, Activation function) {
        layers = new FullyConnectedLayer[topology.length - 1];

        for (int i = 0; i < layers.length; i++) {
            layers[i] = new FullyConnectedLayer(topology[i], topology[i + 1], function);
        }
    }

    /**
     * This method initializes the neural network with the given topology and activation functions.
     * A {@link FullyConnectedLayer} is created for each edge layer.
     * The activation functions can be set in two different ways:
     *
     * <ul>
     *     <li>One activation function is given for each layer.</li>
     *     <li>
     *         Two activation functions are given.
     *         In this case, the second one is used for the output layer, and the first one for all other layers.
     *     </li>
     * </ul>
     * <p>
     * The method throws an exception if the number of activation functions is not correct.
     */
    public void create(int[] topology, Activation[] functions) {
        int size = topology.length - 1;

        if (functions.length != size && !(functions.length == 2 && size > 1)) {
            throw new IllegalArgumentException("The number of activation functions is not correct.");
        }

        if (functions.length == 2) {
            create(topology, functions[0]);

            layers[layers.length - 1].setActivation(functions[1]);
        } else {
            layers = new FullyConnectedLayer[topology.length - 1];

            for (int i = 0; i < topology.length - 1; i++) {
                layers[i] = new FullyConnectedLayer(topology[i], topology[i + 1], functions[i]);
            }
        }
    }

    /**
     * This method returns the topology used to create the neural network.
     */
    protected int[] topology() {
        int[] topology = new int[layers.length + 1];

        for (int i = 0; i < topology.length - 1; i++) {
            topology[i] = layers[i].getWeights().length - 1;
            topology[i + 1] = layers[i].getWeights()[0].length;
        }
        return topology;
    }

    /**
     * This method returns the {@link NeuralNetwork#layers}.
     */
    public FullyConnectedLayer[] getLayers() {
        return layers;
    }

    /**
     * This method sets the {@link NeuralNetwork#layers}.
     * It can be used to initialize a neural network with the return value of {@link utils.Reader#create(String)}.
     */
    public void setLayers(FullyConnectedLayer[] layers) {
        this.layers = layers;
    }

    public void setLoss(Losses loss) {
        this.loss = loss;
    }

    /**
     * This method returns the number of layers of the neural network.
     * The returned value does not correspond to the length of the topology used to create the neural network.
     *
     * @see NeuralNetwork#create(int[], Activation)
     */
    protected int size() {
        return layers.length;
    }

    /**
     * This method writes the topology and the weights of the neural network to a file with the given name.
     * The file has the same format as the files that can be read with {@link utils.Reader#create(String)}.
     * The method throws an exception if an I/O error occurs.
     */
    public void exportWeights(String fileName) throws IOException {
        StringBuilder s = new StringBuilder("layers");

        for (int i : topology()) {
            s.append(";").append(i);
        }

        s.append("\n");

        for (int i = 0; i < size(); i++) {
            double[][] weights = layers[i].getWeights();

            if (i != 0) {
                s.append("\n");
            }

            for (double[] weight : weights) {
                s.append(weight[0]);

                for (int j = 1; j < weight.length; j++) {
                    s.append(";").append(weight[j]);
                }

                s.append("\n");
            }
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            writer.write(s.toString());
        }
    }

    /**
     * This method overwrites the weights of the given edge layer.
     * It throws an exception if the index is out of bounds.
     *
     * @see NeuralNetwork#create(int[], Activation)
     */
    public void setWeights(int index, double[][] weights) {
        if (index < 0 || index >= layers.length) {
            throw new IllegalArgumentException();
        }

        layers[index].setWeights(weights);
    }

    /**
     * This method overwrites the {@link Activation} function of the given edge layer.
     * It throws an exception if the index is out of bounds.
     */
    public void setFunction(int index, Activation function) {
        if (index < 0 || index >= layers.length) {
            throw new IllegalArgumentException();
        }

        layers[index].setActivation(function);
    }

    /**
     * This method overwrites the {@link Activation} function of all edge layers.
     */
    public void setFunctions(Activation function) {
        for (FullyConnectedLayer f : layers) {
            f.setActivation(function);
        }
    }

    public void setActivateMomAll() {
        for (FullyConnectedLayer f : layers) {
            f.activateMomentum();
        }
    }

    public void setActivateMomentum(int index) {
        layers[index].activateMomentum();
    }

    /**
     * This method returns the number of parameters of the neural network.
     */
    public int parameters() {
        int parameters = 0;

        for (FullyConnectedLayer layer : layers) {
            parameters += layer.parameters();
        }

        return parameters;
    }

    /**
     * This method calculates a forward pass through the neural network with the given input.
     */
    public double[] compute(double[] input) {
        double[] result = input.clone();

        for (FullyConnectedLayer layer : layers) {
            result = layer.forward(result);
        }

        return result;
    }

    /**
     * This method calculates a forward pass through the neural network with each of the given inputs.
     */
    public double[][] computeAll(double[][] inputs) {
        double[][] result = Arrays.stream(inputs).map(double[]::clone).toArray(double[][]::new);

        for (FullyConnectedLayer layer : layers) {
            result = layer.forward(result);
        }

        return result;
    }

    /**
     * computes the backpropagation
     *
     * @param deltaLoss     delta input of NN.
     * @param learning_rate learning rate of the NN. (adjustment weights rate.)
     * @throws Exception shape Errors
     */
    public void computeBackward(double[] deltaLoss, double learning_rate) {


        double[] doutput = deltaLoss;
        for (int i = 0; i < layers.length; i++) {
            doutput = this.layers[layers.length - 1 - i].backward(doutput);
        }

    }

    /**
     * computes the backpropagation
     *
     * @param dinput delta input of NN.
     *               has noo learning rate because the optimizer updates the parameter.
     * @throws Exception Shape Errors
     */
    public void computeBackward(double[] dinput) {

        double[] doutput = dinput;
        for (int i = 0; i < layers.length; i++) {
            doutput = this.layers[layers.length - 1 - i].backward(doutput);
        }

    }

    /**
     * computes the backpropagation
     *
     * @param dinputs       delta inputs of NN.
     * @param learning_rate learning rate of the NN. (adjustment weights rate.)
     * @throws Exception Shape Errors
     */
    public void computeAllBackward(double[][] dinputs, double learning_rate) throws Exception {


        double[][] doutputs = dinputs;
        for (int i = 0; i < layers.length; i++) {
            doutputs = this.layers[layers.length - 1 - i].backward(doutputs, learning_rate);

        }

    }

    /**
     * computes the backpropagation
     *
     * @param dinputs delta input of NN.
     *                has noo learning rate because the optimizer updates the parameter.
     * @throws Exception Shape Errors
     */
    public void computeAllBackward(double[][] dinputs) throws Exception {

        double[][] doutputs = dinputs;


        for (int i = 0; i < size(); i++) {
            doutputs = this.layers[size() - i].backward(doutputs);

        }

    }

    /**
     * train the model with a batch.
     * has no learning rate because it uses and optimizer.
     *
     * @param epoch   epochs to train for
     * @param x_train input data for the NN.
     * @param y_train the output the NN shall give.
     * @throws Exception Shape Error.
     */
    public void train_with_batch(int epoch, double[][][] x_train, double[][][] y_train) throws Exception {

        if (this.optimizer == null) {
            throw new Exception("Got no Optimizer and no Learning rate");
        } else if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (topology()[topology().length - 1] != y_train[0][0].length) {
            throw new IllegalArgumentException("y has " + y_train[0][0].length + " classes but " +
                    "model output shape is: " + topology()[topology().length - 1]);
        } else if (topology()[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + topology()[0]);
        }


        double loss_per_epoch;
        int step_size = x_train.length;

        double[] step_losses = new double[step_size];

        for (int i = 0; i < epoch; i++) {
            double[][] outs;

            for (int j = 0; j < step_size; j++) {

                outs = computeAll(Array_utils.copyArray(x_train[j]));
                //one epoch is finished.
                //calculates Loss
                step_losses[j] = loss.forward(outs, y_train[j]);
                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does back propagation
                this.computeAllBackward(outs);
            }
            loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }


    }

    /**
     * train the model with a batch.
     *
     * @param epoch         epochs to train for
     * @param x_train       input data for the NN.
     * @param y_train       the output the NN shall give.
     * @param learning_rate learning rate for weights.
     * @throws Exception Shape Error.
     */
    public void train_with_batch(int epoch, double[][][] x_train, double[][][] y_train, double learning_rate) throws Exception {

        System.out.println("Train Data Batch Size: " + x_train[0].length);
        System.out.println("Train Data Length: " + x_train[0][0].length);
        //checks for rxceptions
        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (this.topology()[this.topology().length - 1] != y_train[0][0].length) {
            throw new IllegalArgumentException("y has " + y_train[0][0].length + " classes but " +
                    "model output shape is: " + topology()[topology().length - 1]);
        } else if (topology()[0] != x_train[0][0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + topology()[0]);
        }


        double loss_per_epoch;

        int step_size = x_train.length;

        double[] step_losses = new double[step_size];


        for (int i = 0; i < epoch; i++) {

            for (int j = 0; j < step_size; j++) {
                double[][] outs;
                outs = computeAll(Array_utils.copyArray(x_train[j]));
                //one epoch is finished.
                //calculates Loss
                step_losses[j] = loss.forward(outs, y_train[j]);
                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does back propagation //updates values.
                this.computeAllBackward(outs, learning_rate);
            }
            loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }
    }

    /**
     * train the model with a batch.
     *
     * @param epoch         epochs to train for
     * @param x_train       single data input for the NN.
     * @param y_train       single y NN shall give.
     * @param learning_rate learning rate for weights.
     * @throws Exception More.
     */
    public void train_single(int epoch, double[][] x_train, double[][] y_train, double learning_rate) throws Exception {

        System.out.println("Train Topologie: " + Arrays.toString(this.topology()));
        System.out.println("Train Data Length: " + x_train.length);
        //checks for rxceptions
        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (this.topology()[topology().length - 1] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + this.topology()[this.topology().length - 1]);
        } else if (this.topology()[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + this.topology()[0]);
        }


        double loss_per_epoch;
        double step_loss;
        int step_size = x_train.length;
        for (int i = 0; i < epoch; i++) {
            loss_per_epoch = 0;

            double[] outs;

            for (int j = 0; j < step_size; j++) {
                outs = compute(Array_utils.copyArray(x_train[j]));
                step_loss = loss.forward(outs, y_train[j]);
                loss_per_epoch += step_loss;
                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does back propagation // an updates the weights.
                computeBackward(outs, learning_rate);

            }
            loss_per_epoch = loss_per_epoch / x_train.length;
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }
    }


    public void trainTestsingle(int epoch, double[][] x_train, double[][] y_train, double[][] x_test, double[][] y_test, double learning_rate) throws Exception {

        System.out.println("Train Topologie: " + Arrays.toString(this.topology()));
        System.out.println("Train Data Length: " + x_train.length);
        //checks for rxceptions
        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (this.topology()[topology().length - 1] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + this.topology()[this.topology().length - 1]);
        } else if (this.topology()[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + this.topology()[0]);
        }


        double loss_per_epoch;
        double step_loss;
        int step_size = x_train.length;
        for (int i = 0; i < epoch; i++) {
            loss_per_epoch = 0;

            double[] outs;

            for (int j = 0; j < step_size; j++) {
                outs = compute(Array_utils.copyArray(x_train[j]));
                step_loss = loss.forward(outs, y_train[j]);
                loss_per_epoch += step_loss;
                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does backpropagation // and updates the weights.
                computeBackward(outs, learning_rate);

            }
            loss_per_epoch = loss_per_epoch / x_train.length;
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }

        double lossTest = test_single(x_test, y_test);
        System.out.println("Acc: " + lossTest);

    }

    public double test_single(double[][] x_train, double[][] y_train) {


        System.out.println(Arrays.toString(this.topology()));
        System.out.println("Test Data Length: " + x_train.length);


        double loss_per_epoch = 0;
        int step_size = x_train.length;


        int[] predictions = new int[x_train.length];

        int pred;
        int ist;
        double[] outs;
        for (int j = 0; j < step_size; j++) {
            outs = compute(x_train[j]);
            outs = Utils.clean_input(outs, y_train[0].length);
            pred = Utils.argmax(outs);
            ist = Utils.argmax(y_train[j]);


            if (pred == ist) {
                loss_per_epoch += 1;
            }
            predictions[j] = pred;


        }
        loss_per_epoch = loss_per_epoch / x_train.length;
        System.out.println("Acc ing epoch: " + loss_per_epoch);

        /**
         for (int i = 0; i < 20; i++) {
         System.out.println("Predicted Class: " + predictions[i] + " Actual Class: " + Utils.argmax(y_train[i]));
         }**/


        return loss_per_epoch;
    }

    /**
     * This method returns a string representation of the neural network.
     * It contains the topology and the number of parameters.
     *
     * @see NeuralNetwork#topology()
     * @see NeuralNetwork#parameters()
     */
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();

        s.append("Topology: ").append(Arrays.toString(topology())).append("\n");

        for (int i = 0; i < layers.length; i++) {
            s.append("Weights of Layer ").append(i).append(":\n").append(layers[i]).append("\n");
        }

        s.append("Parameters: ").append(parameters()).append("\n");

        return s.toString();
    }
}
