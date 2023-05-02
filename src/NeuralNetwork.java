import layers.*;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;

public class NeuralNetwork {
    int model_size; //number of layers
    int parameter_size; //parameter Size
    Layer[] structur; //strucktur of the model. contains Layers and Activations.
    Activation[] activations; //needed for test purpose
    FullyConnectedLayer[] Ebenen; //needed for test purpose
    int[] topologie; //the original topologie.
    Losses loss = null; //loss function of our NN. // now only MSE is available.
    SGD optimizer = null; // right now not really supported.

    /**
     * cerates the model with a given Topologie.
     *
     * @param Topologie strucktur addeds an activation after every Layer.
     */
    public void create(int[] Topologie) {
        //size -1, weil die erste Zahl die größe der Eingabe Daten entspricht.
        model_size = Topologie.length - 1; //länge der Topologie
        topologie = Topologie;
        this.structur = new Layer[model_size];
        this.activations = new Activation[model_size];


        for (int i = 0; i < model_size; i++) {
            structur[i] = new FullyConnectedLayer(Topologie[i], Topologie[i + 1]);
            activations[i] = new Tanh();
            parameter_size += Topologie[i] * Topologie[i + 1] + Topologie[i + 1];

        }

    }

    /**
     * uses Same Activation Function for every Layer.
     *
     * @param Topologie model strucktur.
     * @param function  activation Function.
     */
    public void create(int[] Topologie, String function) {
        //size -1, weil die erste Zahl die größe der Eingabe Daten entspricht.
        model_size = (Topologie.length - 1) * 2; //länge der Topologie

        topologie = Topologie;
        this.structur = new Layer[model_size];


        int count = 0;
        for (int i = 0; i < model_size; i += 2) {
            structur[i] = new FullyConnectedLayer(Topologie[count], Topologie[count + 1]);
            structur[i + 1] = Utils.getActivation(function);
            count += 1;


        }
        updateParameterSize();
    }

    /**
     * calculates the parameter Size of the NN.
     */
    public void updateParameterSize() {
        this.parameter_size = 0;
        for (int i = 0; i < model_size; i += 2) {
            this.parameter_size += this.structur[i].parameter_size;
        }
    }

    public Layer[] getModel() {
        return this.structur;
    }

    public int[] getTopologie() {
        int[] t;
        int count = 0;
        for (Layer l : this.structur) {
            if (l.weights != null) {
                count += 1;
            }
        }
        t = new int[count];

        count = 0;
        for (Layer l : this.structur) {
            if (l.weights != null) {
                t[count] = l.weights.length;
                count += 1;
            }
        }
        return t;
    }

    /**
     * creates the model with a given Layer Array.
     *
     * @param layers
     */
    public void create(Layer[] layers) {
        //size -1, weil die erste Zahl die größe der Eingabe Daten entspricht.
        model_size = layers.length; //länge der Topologie

        this.structur = layers;
        this.topologie = this.getTopologie();
        updateParameterSize();
        int count = 0;
        for (int i = 0; i < model_size; i++) {
            if (this.structur[i].weights != null) {
                topologie[count] = this.structur[i].weights.length;
                count += 1;
            }
        }


    }

    /**
     * @param Topologie
     * @param function  activation it is expected to get the same size as the
     *                  given Topologie -1. if only 2 Functions are given, is the meaning,
     *                  that the first 1 is used after every Layer and the last One is for the
     *                  Output.
     * @throws IllegalArgumentException
     */
    public void create(int[] Topologie, String[] function) throws IllegalArgumentException {
        //size -1, weil die erste Zahl die größe der Eingabe Daten entspricht.
        model_size = Topologie.length - 1; //länge der Topologie
        topologie = Topologie;

        if (model_size != function.length && !(function.length == 2 && model_size > 1)) {
            throw new IllegalArgumentException("Methode create got mismatching Size activations");
        }


        model_size = model_size * 2;

        this.structur = new Layer[model_size];


        if (function.length != 2) {
            for (int i = 0; i < model_size; i += 2) {
                structur[i] = new FullyConnectedLayer(Topologie[i], Topologie[i + 1]);
                structur[i + 1] = Utils.getActivation(function[i]);

            }
        } else {
            //meaning the function got one Activation function for every Layer
            // and the last function as last activation function.
            for (int i = 0; i < model_size; i += 2) {
                if (model_size == i + 2) {
                    structur[i] = new FullyConnectedLayer(Topologie[i], Topologie[i + 1]);
                    structur[i + 1] = Utils.getActivation(function[1]);

                } else {
                    structur[i] = new FullyConnectedLayer(Topologie[i], Topologie[i + 1]);
                    structur[i + 1] = Utils.getActivation(function[0]);

                }
            }

        }


    }

    /**
     * computes the backpropagation
     *
     * @param dinput        delta input of NN.
     * @param learning_rate learning rate of the NN. (adjustment weights rate.)
     * @throws Exception shape Errors
     */
    public void computeBackward(double[] dinput, double learning_rate) throws Exception {


        double[] doutput = dinput;
        for (int i = 1; i < model_size; i++) {
            doutput = this.structur[model_size - i].backward(doutput, learning_rate);
        }


    }

    /**
     * computes the backpropagation
     *
     * @param dinput delta input of NN.
     *               has noo learning rate because the optimizer updates the parameter.
     * @throws Exception Shape Errors
     */
    public void computeBackward(double[] dinput) throws Exception {

        double[] doutput = dinput;
        for (int i = 1; i < structur.length; i++) {
            doutput = this.structur[structur.length - 1 - i].backward(doutput);
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
        for (int i = 0; i < structur.length; i++) {
            doutputs = this.structur[structur.length - 1 - i].backward(doutputs, learning_rate);

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


        for (int i = 0; i < model_size; i++) {
            doutputs = this.structur[model_size - i].backward(doutputs);

        }

    }

    /**
     * forward pass of the NN with a batch.
     *
     * @param inputs batch of inputs.
     * @return the computed Output of the NN.
     * @throws Exception
     */
    public double[][] computeAll(double[][] inputs) throws Exception {

        double[][] outputs = inputs;
        for (Layer layer : this.structur) {
            outputs = layer.forward(outputs);

        }

        return outputs;
    }

    /**
     * forward pass of the NN with a single input.
     *
     * @param input batch of inputs.
     * @return the computed Output of the NN.
     */
    public double[] compute(double[] input) {

        double[] output = input;
        for (int i = 0; i < model_size; i++) {
            output = this.structur[i].forward(output);

        }


        return output;
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
        } else if (topologie[topologie.length - 1] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + topologie[topologie.length - 1]);
        } else if (topologie[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + topologie[0]);
        }


        double loss_per_epoch;
        int step_size = x_train.length;

        double[] step_losses = new double[step_size];

        for (int i = 0; i < epoch; i++) {
            double[][] outs;

            for (int j = 0; j < step_size; j++) {
                outs = computeAll(x_train[j]);

                //one epoch is finished.
                //calculates Loss
                step_losses[j] = loss.forward(outs, y_train[j]);

                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does back propagation
                this.computeAllBackward(outs);

                //updates values.
                for (int k = 0; k < model_size; k++) {
                    if (structur[k].weights != null) {
                        this.optimizer.calculate(structur[k]);
                    }
                }

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

        //checks for rxceptions
        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (topologie[topologie.length - 1] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + topologie[topologie.length - 1]);
        } else if (topologie[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + topologie[0]);
        }


        double loss_per_epoch;

        int step_size = x_train.length;

        double[] step_losses = new double[step_size];


        for (int i = 0; i < epoch; i++) {

            for (int j = 0; j < step_size; j++) {
                double[][] outs;
                outs = computeAll(x_train[j]);
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
     * has no learning rate because Optimizer is used.
     *
     * @param epoch   epochs to train for
     * @param x_train single data input for the NN.
     * @param y_train single y the NN shall give.
     * @throws Exception More.
     */
    public void train_single(int epoch, double[][] x_train, double[][] y_train) throws Exception {


        //checks for rxceptions
        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (topologie[topologie.length - 1] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + topologie[topologie.length - 1]);
        } else if (topologie[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + topologie[0]);
        } else if (this.optimizer == null) {
            throw new IllegalArgumentException("Got no Optimizer and no Learning rate");
        }


        double loss_per_epoch;
        int step_size = x_train.length;
        double step_loss;
        for (int i = 0; i < epoch; i++) {
            loss_per_epoch = 0;

            double[] outs;

            for (int j = 0; j < step_size; j++) {
                outs = this.compute(x_train[j]);

                //one epoch is finished.
                //calculates Loss
                step_loss = loss.forward(outs, y_train[j]);
                loss_per_epoch += step_loss;
                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does back propagation
                this.computeBackward(outs);

                //updates values.
                for (int k = 0; k < model_size; k++) {
                    if (structur[k].weights != null) {
                        this.optimizer.calculate(structur[k]);
                    }
                }

            }
            loss_per_epoch = loss_per_epoch / x_train.length;
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


        //checks for rxceptions
        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have diffrent Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (topologie[topologie.length - 1] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + topologie[topologie.length - 1]);
        } else if (topologie[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + topologie[0]);
        }


        double loss_per_epoch;
        double step_loss;
        int step_size = x_train.length;
        for (int i = 0; i < epoch; i++) {
            loss_per_epoch = 0;

            double[] outs;

            for (int j = 0; j < step_size; j++) {
                compute(x_train[j]);

                outs = new double[y_train[0].length];
                outs = Utils.clean_input(outs, y_train[0].length);

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

    //TODO topologie musst be correkted
    public void set_model(Layer[] layers) {

        this.structur = layers;


    }

    public double[][][] get_weights() {

        int layers_got = Ebenen.length;
        double[][][] w = new double[layers_got][][];
        for (int i = 0; i < layers_got; i++) {
            w[i] = this.Ebenen[i].get_weights();

        }

        return w;
    }

    public void load_weights(double[][][] w) {
        int layers_got = w.length;

        for (int i = 0; i < layers_got; i++) {
            this.Ebenen[i].set_weights(w[i]);

        }


    }

    public void test_with_batch(double[][][] x_train, double[][][] y_train) throws Exception {

        int step_size = x_train.length;
        int batch_size = x_train[0].length;
        int total_size = step_size * batch_size;

        double[][] compare = new double[total_size][2];

        double loss_per_epoch;
        double[] step_losses = new double[step_size];

        this.Ebenen = new FullyConnectedLayer[3];
        Ebenen[0] = new FullyConnectedLayer(784, 49);
        Ebenen[1] = new FullyConnectedLayer(49, 20);
        Ebenen[2] = new FullyConnectedLayer(20, 10);

        Activation act = new Tanh();

        double[][] outs;
        for (int j = 0; j < step_size; j++) {
            outs = x_train[j];
            for (FullyConnectedLayer fullyConnectedLayer : Ebenen) {
                outs = fullyConnectedLayer.forward(outs);
                outs = act.forward(outs);
            }

            //one epoch is finished.
            //calculates Loss
            step_losses[j] = this.loss.forward(outs, y_train[j]);
            for (int y_out = 0; y_out < y_train[0].length; y_out++) {


                compare[j * batch_size + y_out][0] = Utils.argmax(outs[y_out]);
                compare[j * batch_size + y_out][1] = Utils.argmax(y_train[j][y_out]);


            }
        }
        loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
        System.out.println("Loss per epoch: " + loss_per_epoch);

        for (int preds = 0; preds < compare.length; preds++) {
            System.out.println("Predicted Class: " + compare[preds][0]
                    + " Y true Class: " + compare[preds][1]);
            if (preds == 20) {
                break;
            }

        }
    }

    public void train_batch_new(int epoch, double[][][] x_train, double[][][] y_train, double learning_rate) throws Exception {

        double loss_per_epoch;

        int step_size = x_train.length;

        double[] step_losses = new double[step_size];

        FullyConnectedLayer[] Ebenen = new FullyConnectedLayer[2];
        Ebenen[0] = new FullyConnectedLayer(784, 49);
        Ebenen[1] = new FullyConnectedLayer(49, 10);

        Activation act = new Tanh();

        int ownSize = Ebenen.length;
        double[][] outs;
        for (int i = 0; i < epoch; i++) {

            for (int j = 0; j < step_size; j++) {

                double[][] data;

                data = x_train[j];
                outs = data;
                for (FullyConnectedLayer fullyConnectedLayer : Ebenen) {
                    outs = fullyConnectedLayer.forward(outs);
                    outs = act.forward(outs);


                }

                //one epoch is finished.
                //calculates Loss
                step_losses[j] = loss.forward(outs, y_train[j]);
                //calculates prime Loss
                double[][] grad = loss.backward(outs, y_train[j]);
                // now does back propagation //updates values.
                for (int l = 0; l < Ebenen.length; l++) {
                    grad = act.backward(grad);
                    grad = Ebenen[ownSize - 1 - l].backward(grad, learning_rate);

                }


            }

            loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }
    }

    private String layer2b_w() {
        StringBuilder s_out = new StringBuilder();


        for (int k = 0; k < structur.length; k++) {

            if (structur[k].weights != null) {
                s_out.append("Layer ").append(k).append(": \n");
                s_out.append(Utils.weightsAndBiases_toString(structur[k].weights,
                        structur[k].biases));
            }


        }
        return s_out.toString();
    }

    private String layer_export() {
        StringBuilder s_out = new StringBuilder();


        for (int k = 0; k < structur.length; k++) {

            if (structur[k].weights != null) {
                s_out.append("Layer ").append(k).append(": \n");
                s_out.append(Utils.weightsAndBiases_export(structur[k].weights,
                        structur[k].biases));
            } else {
                if (k < structur.length - 1) {
                    s_out.append(structur[k].name);
                } else {
                    s_out.append(structur[k].name).append("\n");
                }


            }
        }
        return s_out.toString();
    }

    public String modelExport() {
        String s_out = "layers: " + Arrays.toString(topologie) + "\n";
        s_out += this.layer_export();
        return s_out;
    }

    public void modelExport2file(String fpath) throws Exception {
        String s_out = "layers: " + Arrays.toString(topologie) + "\n";
        s_out += this.layer_export();

        BufferedWriter writer = new BufferedWriter(new FileWriter(fpath));
        writer.write(s_out);
        writer.close();
    }

    public String toString() {
        String s_out = "Strucktur: " + Arrays.toString(topologie) + "\n";
        s_out += "Parameter: " + parameter_size + "\n";
        s_out += layer2b_w();

        return s_out;
    }
}
