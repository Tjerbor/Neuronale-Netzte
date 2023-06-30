package main;

import function.Activation;
import layer.FCL;
import layer.FastLinearLayer;
import layer.FullyConnectedLayer;
import layer.Layer;
import loss.Loss;
import optimizer.AdamNew;
import optimizer.Optimizer;
import utils.Array_utils;
import utils.Matrix;
import utils.TrainUtils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This class models a fully connected feed-forward artificial neural network.
 */
public class NeuralNetwork {
    /**
     * This variable contains the first layer of the neural network.
     */
    private Layer inputLayer;
    /**
     * This variable contains the last layer of the neural network.
     */
    private Layer outputLayer;
    /**
     * This variable contains the number of layers of the neural network.
     */
    private int size;
    /**
     * This variable contains the loss function of the neural network.
     */
    private Loss loss;
    /**
     * This variable contains the optimizer of the neural network.
     */
    private Optimizer optimizer;

    /**
     * This method initializes the neural network with the given topology.
     * A {@link FullyConnectedLayer} is created for each edge layer.
     */
    public void create(int[] topology) {
        size = topology.length - 1;

        for (int i = 0; i < size; i++) {
            add(new FullyConnectedLayer(topology[i], topology[i + 1]));
        }
    }

    /**
     * This method initializes the neural network with the given topology and activation function.
     * A {@link FullyConnectedLayer} is created for each edge layer.
     */
    public void create(int[] topology, Activation function) {
        size = topology.length - 1;

        for (int i = 0; i < size; i++) {
            add(new FullyConnectedLayer(topology[i], topology[i + 1], function));
        }
    }

    /**
     * This method initializes the neural network with the given topology and activation functions.
     * A {@link FullyConnectedLayer} is created for each edge layer.
     * The method throws an exception if the number of activation functions is not correct.
     */
    public void create(int[] topology, Activation[] functions) {
        if (functions.length != topology.length - 1) {
            throw new IllegalArgumentException("The number of activation functions is not correct.");
        }

        size = topology.length - 1;

        for (int i = 0; i < size; i++) {
            add(new FullyConnectedLayer(topology[i], topology[i + 1], functions[i]));
        }
    }


    public void create(int[] topology, int weightGenType) {

        size = topology.length - 1;
        for (int i = 0; i < size; i++) {
            FullyConnectedLayer f = new FullyConnectedLayer(topology[i], topology[i + 1]);
            f.genWeights(weightGenType);
            this.add(f);
        }
    }

    public int[] topology() {

        Layer[] layerNews = this.layers2Array();
        int[] topologie = new int[size + 1];

        int count = 0;
        for (Layer l : layerNews) {
            if (l instanceof FastLinearLayer || l instanceof FCL || l instanceof FullyConnectedLayer) {
                topologie[count] = l.getInputShape()[0];
                topologie[count + 1] = l.getOutputShape()[0];
                count += 1;

            }
        }
        return topologie;


    }


    public void create(Layer[] layers) {

        if (layers.length == 2) {
            this.size = 2;
            this.inputLayer = layers[0];
            this.outputLayer = layers[1];
            this.inputLayer.setNextLayer(this.outputLayer);
            this.outputLayer.setPreviousLayer(this.inputLayer);
        } else {
            for (Layer l : layers
            ) {
                this.add(l);
            }

        }
    }

    public Matrix compute(Matrix m) {

        inputLayer.forward(m);

        if (this.outputLayer == null) {
            return inputLayer.getOutput();
        }

        return outputLayer.getOutput();

    }


    public void computeBackward(Matrix m, double learningRate) {

        if (this.outputLayer == null) {
            inputLayer.backward(m, learningRate);
        } else {
            outputLayer.backward(m, learningRate);
        }


    }

    public void computeBackward(Matrix m, int iterationAt, double learningRate) {

        if (this.outputLayer == null) {
            inputLayer.setIterationAt(iterationAt);
            inputLayer.backward(m, learningRate);
        } else {
            outputLayer.setIterationAt(iterationAt);
            outputLayer.backward(m, learningRate);
        }


    }

    public double[] compute(double[] input) {
        inputLayer.forward(new Matrix(input));

        if (this.outputLayer == null) {
            return inputLayer.getOutput().getData1D();
        }
        return outputLayer.getOutput().getData1D();

    }

    public double[][] compute(double[][] inputs) {
        inputLayer.forward(new Matrix(inputs));
        if (this.outputLayer == null) {
            return inputLayer.getOutput().getData2D();
        }
        return outputLayer.getOutput().getData2D();
    }

    public void computeBackward(double[] input, int epochAt) {
        if (this.outputLayer == null) {
            this.build();
        }
        outputLayer.setIterationAt(epochAt);
        outputLayer.backward(new Matrix(input));
    }

    public void computeBackward(double[] input) {
        if (this.outputLayer == null) {
            this.build();
        }
        outputLayer.backward(new Matrix(input));
    }

    public void computeBackward(double[] input, int epochAt, double learningRate) {
        if (this.outputLayer == null) {
            this.build();
        }
        outputLayer.setIterationAt(epochAt); // start with zero.
        outputLayer.setLearningRate(learningRate);
        outputLayer.backward(new Matrix(input));
    }

    public void computeBackward(double[][] inputs, int epochAt) {
        if (this.outputLayer == null) {
            this.build();
        }

        outputLayer.setIterationAt(epochAt + 1); // start with zero.
        outputLayer.backward(new Matrix(inputs));
    }

    public void computeBackward(double[][] inputs, int epochAt, double learningRate) {
        if (this.outputLayer == null) {
            this.build();
        }
        outputLayer.setIterationAt(epochAt + 1); // start with zero.
        //lastLayer.setLearningRate(learningRate);
        outputLayer.backward(new Matrix(inputs), learningRate);
    }

    /**
     * only needed for training.
     */
    public void build() {
        this.setTrainingsStuff();
    }

    public int size() {
        return size;
    }

    /**
     * public void addConvLayer(int[] shape, int numbFilter, int kernelSize, int stride){
     * <p>
     * if(shape[0] == 1 || shape[0] == 3){
     * Conv2D  conv2D = new Conv2D(shape, numbFilter, kernelSize, stride);
     * }else  {
     * <p>
     * }
     * <p>
     * }
     **/


    public void add(Layer l) {

        if (l != null) {
            if (this.inputLayer == null) {
                inputLayer = l;

            } else if (outputLayer == null) {
                outputLayer = l;
                inputLayer.setNextLayer(outputLayer);
                outputLayer.setPreviousLayer(inputLayer);
            } else {
                Layer before = outputLayer;
                outputLayer = l;
                before.setNextLayer(outputLayer);
                outputLayer.setPreviousLayer(before);

            }
            size++;
        }
    }

    public void setLoss(Loss loss) {
        this.loss = loss;
    }


    public void setAdamOptimizer() {

        Layer tmp = inputLayer;
        while (tmp != null) {
            tmp.setOptimizer(new AdamNew());
            tmp = tmp.nextLayer;

        }

    }

    public void genWeights(int type) {

        Layer tmp = inputLayer;
        while (tmp != null) {
            tmp.genWeights(type);
            tmp = tmp.nextLayer;

        }

    }

    public void setAdamOptimizer(double lr) {

        Layer tmp = inputLayer;
        while (tmp != null) {
            AdamNew adamNew = new AdamNew();
            adamNew.setLearningRate(lr);
            tmp.setOptimizer(adamNew);
            tmp = tmp.nextLayer;

        }

    }


    private Layer[] layers2Array() {
        Layer[] layers = new Layer[size];

        layers[0] = inputLayer;

        Layer tmp = inputLayer;
        for (int i = 1; i < layers.length; i++) {
            tmp = tmp.getNextLayer();
            layers[i] = tmp;

        }
        return layers;
    }


    /**
     * set Training for all Layers.
     * Also set Optimizer.
     * Maybe set new Strides.
     */
    private void setTrainingsStuff() {


        if (this.optimizer != null) {
            inputLayer.setOptimizer(this.optimizer);

        }
        inputLayer.setTraining(true);

        Layer tmp = inputLayer;
        for (int i = 1; i < size; i++) {
            tmp = tmp.getNextLayer();
            tmp.setTraining(true);
            if (this.optimizer != null) {
                tmp.setOptimizer(this.optimizer);
            }


        }


    }


    public Layer[] getLayers() {
        return layers2Array();
    }

    public void setLayers(Layer[] layers) {

        inputLayer = layers[0];

        for (int i = 1; i < layers.length; i++) {
            this.add(layers[i]);
        }


    }

    public void checkTraining(double[][] x_train, double[][] y_train) {


        if (x_train.length != y_train.length) {
            throw new IllegalArgumentException("x und y Data have different Size.");
        } else if (this.loss == null) {
            throw new IllegalArgumentException("loss function is not set.");
        } else if (this.outputLayer.getOutputShape()[0] != y_train[0].length) {
            throw new IllegalArgumentException("y has " + y_train[0].length + " classes but " +
                    "model output shape is: " + outputLayer.getOutputShape()[0]);
        } else if (inputLayer.getInputShape()[0] != x_train[0].length) {
            throw new IllegalArgumentException("x has " + x_train[0].length + " input shape but " +
                    "model inputs shape is: " + inputLayer.getInputShape()[0]);
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

    public void test(double[][][][] inputDaten, double[][] classes) {

        int step_size = inputDaten.length;

        double acc = 0;
        for (int j = 0; j < step_size; j++) {
            Matrix out;
            out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));

            if (Utils.argmax(out.getData1D()) == Utils.argmax(classes[j])) {
                acc += 1;
            }


        }

        System.out.println("Acc: " + acc / inputDaten.length);


    }

    public void testBatch(double[][][] inputDaten, double[][][] classes) {

        int step_size = inputDaten.length;

        double acc = 0;
        double[][] tmp;
        for (int j = 0; j < step_size; j++) {
            Matrix out;
            out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));

            tmp = out.getData2D();
            for (int i = 0; i < classes[j].length; i++) {
                if (Utils.argmax(tmp[i]) == Utils.argmax(classes[j][i])) {
                    acc += 1;
                }
            }


        }

        System.out.println("Acc: " + acc / inputDaten.length);


    }

    public void printTestStats(double[][] inputDaten, double[][] classes) {
        System.out.println(getTestStats(inputDaten, classes));

    }

    public String getTestStats(double[][] inputDaten, double[][] classes) {

        StringBuilder s = new StringBuilder();
        int step_size = inputDaten.length;

        int[][] prediction = new int[classes[0].length][2];
        double acc = 0;
        for (int j = 0; j < step_size; j++) {
            double[] out;
            out = compute(Array_utils.copyArray(inputDaten[j]));

            if (Utils.argmax(out) == Utils.argmax(classes[j])) {
                acc += 1;
                prediction[Utils.argmax(classes[j])][0] += 1;
                prediction[Utils.argmax(classes[j])][1] += 1;
            } else {
                prediction[Utils.argmax(classes[j])][1] += 1;
            }


        }

        s.append("Acc Overall: ").append(acc / inputDaten.length).append("\n");
        for (int i = 0; i < prediction.length; i++) {
            s.append("class: ").append(i).append(" was correct predicted: ").append(prediction[i][0]).append(" / ").append(prediction[i][1]).append(" per: ");
            s.append((prediction[i][0] / prediction[i][1]))
                    .append("\n");
        }

        return s.toString();

    }

    public void setFunction(int index, Activation act) {

        System.out.println(size);
        if (index > size() - 1) {
            throw new IllegalArgumentException("set Activation: index out of bounds");
        }

        if (index == 0) {

            this.inputLayer.setActivation(act);
        } else {

            Layer[] layers = layers2Array();
            layers[index].setActivation(act);

        }

    }


    public void train(int epochs, double[][] inputDaten, double[][] classes) {

        this.build();
        this.checkTraining(inputDaten, classes); //check if got valid Arguments for Training.

        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double[] stepLosses = new double[step_size];
            double[][][] data = TrainUtils.shuffle(inputDaten, classes);


            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(data[0][j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, data[1][j]);
                //calculates backward Loss
                out = loss.backward(out, data[1][j]);
                // now does back propagation
                this.computeBackward(out, i);
            }
            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));


        }


    }

    public void train(int epochs, double[][] inputDaten, double[][] classes, double learningRate) {

        this.build();
        this.checkTraining(inputDaten, classes); //check if got valid Arguments for Training.

        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(inputDaten[j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, classes[j]);
                //calculates backward Loss
                out = loss.backward(out, classes[j]);
                // now does back propagation
                this.computeBackward(out, i);
            }
            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));


        }


    }

    public void trainBatch(int epochs, double[][][] inputDaten, double[][][] classes, double learningRate) {

        this.build();
        //this.checkTraining(inputDaten, classes); //check if got valid Arguments for Training.

        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            Matrix out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {

                out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));
                //calculates Loss
                stepLosses[j] = loss.forward(out.getData2D(), classes[j]);
                //calculates backward Loss
                out = new Matrix(loss.backward(out.getData2D(), classes[j]));
                // now does back propagation
                this.computeBackward(out, i, learningRate);
            }
            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));


        }


    }


    public void train(int epochs, double[][] inputDaten, double[][] classes, double[][] testInput, double[][] testClasses) {
        this.build();
        this.checkTraining(inputDaten, classes);


        this.printSummary();
        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(inputDaten[j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, classes[j]);
                //calculates backward Loss
                out = loss.backward(out, classes[j]);
                // now does back propagation
                this.computeBackward(out, i);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
            test(testInput, testClasses);
        }


    }

    public void train(int epochs, double[][] inputDaten, double[][] classes, double[][] testInput, double[][] testClasses, double learningRate) {
        this.build();
        this.checkTraining(inputDaten, classes);


        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(inputDaten[j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, classes[j]);
                //calculates backward Loss
                out = loss.backward(out, classes[j]);
                // now does back propagation
                this.computeBackward(out, i, learningRate);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
            test(testInput, testClasses);
        }


    }

    public void trainBatch(int epochs, double[][][] inputDaten, double[][][] classes, double[][][] testInput, double[][][] testClasses, double learningRate) {
        this.build();
        //this.checkTraining(inputDaten, classes);


        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            Matrix out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));
                //calculates Loss
                stepLosses[j] = loss.forward(out.getData2D(), classes[j]);
                //calculates backward Loss
                out = new Matrix(loss.backward(out.getData2D(), classes[j]));
                // now does back propagation
                this.computeBackward(out, i, learningRate);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
            testBatch(testInput, testClasses);
        }


    }

    public void train(int epochs, double[][][][] inputDaten, double[][] classes, double learningRate) {
        this.build();

        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            Matrix out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));
                //calculates Loss
                stepLosses[j] = loss.forward(out.getData1D(), classes[j]);
                //calculates backward Loss
                out = new Matrix(loss.backward(out.getData1D(), classes[j]));
                // now does back propagation
                this.computeBackward(out.getData1D(), i, learningRate);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
        }


    }

    public void trainPerStep(int stepsToTrain, int stepsStats, double[][] inputDaten, double[][] classes, double[][] testDaten, double[][] testClasses, double learningRate) {

        long st = System.currentTimeMillis();
        Matrix out;

        int count = 0;
        int dSize = inputDaten.length;
        double[] stepLosses = new double[stepsStats];

        for (int j = 0; j < stepsToTrain; j++) {
            out = compute(new Matrix(Array_utils.copyArray(inputDaten[j % dSize])));
            //calculates Loss
            stepLosses[j % stepsStats] = loss.forward(out.getData1D(), classes[j % dSize]);
            //calculates backward Loss
            out = new Matrix(loss.backward(out.getData1D(), classes[j % dSize]));
            // now does back propagation
            this.computeBackward(out.getData1D(), (int) j / dSize, learningRate);

            if (j % stepsStats == 0) {
                long end = System.currentTimeMillis();
                System.out.println("Time: " + ((end - st) / 1000));
                System.out.println("Round: " + count + " Loss: " + Utils.mean(stepLosses));
                test(testDaten, testClasses);
                st = System.currentTimeMillis();
                stepLosses = new double[stepsStats];
                count += 1;

            }

        }


    }

    public void trainTesting(int epochs, double[][][][] inputDaten, double[][] classes, double[][][][] testDaten, double[][] testClasses, double learningRate) {
        this.build();

        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            Matrix out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));
                //calculates Loss
                stepLosses[j] = loss.forward(out.getData1D(), classes[j]);
                //calculates backward Loss
                out = new Matrix(loss.backward(out.getData1D(), classes[j]));
                // now does back propagation
                this.computeBackward(out.getData1D(), i, learningRate);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
            test(testDaten, testClasses);
        }


    }

    public void trainBatch(int epochs, double[][][][][] inputDaten, double[][] classes, double learningRate) {
        this.build();

        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            Matrix out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                out = compute(new Matrix(Array_utils.copyArray(inputDaten[j])));
                System.out.println(out);
                //calculates Loss
                stepLosses[j] = loss.forward(out.getData1D(), classes[j]);
                //calculates backward Loss
                out = new Matrix(loss.backward(out.getData1D(), classes[j]));
                // now does back propagation
                this.computeBackward(out.getData1D(), i, learningRate);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
        }


    }

    public String trainTesting(int epochs, double[][] inputDaten, double[][] classes, double[][] testInput, double[][] testClasses, double learningRate) {
        this.build();
        this.checkTraining(inputDaten, classes);


        String s = "";

        s += "learningRate: " + learningRate + "\n";
        long st = 0;
        long end;
        int step_size = inputDaten.length;

        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double[] stepLosses = new double[step_size];

            s += "epoch: " + i + "\n";
            for (int j = 0; j < step_size; j++) {
                out = compute(Array_utils.copyArray(inputDaten[j]));
                //calculates Loss
                stepLosses[j] = loss.forward(out, classes[j]);
                //calculates backward Loss
                out = loss.backward(out, classes[j]);
                // now does back propagation
                this.computeBackward(out, i, learningRate);
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
            s += ("Time: " + ((end - st) / 1000)) + "\n";
            s += ("Loss: " + Utils.mean(stepLosses)) + "\n";
            String tmp = getTestStats(testInput, testClasses);
            System.out.println(tmp);
            s += tmp;
            s += "\n\n";
        }

        return s;
    }


    public int parameters() {
        int sum = 0;

        Layer tmp = inputLayer;
        while (tmp != null) {
            sum += tmp.parameters();
            tmp = tmp.nextLayer;
        }
        return sum;
    }

    @Override
    public String toString() {
        return "NN_New{" +
                "fristLayer=" + inputLayer.summary() +
                ", lastLayer=" + outputLayer.summary() +
                ", loss=" + loss +
                ", optimizer=" + optimizer +
                ", size=" + size +
                ", parameters=" + parameters() +
                '}';
    }

    public void printSummary() {
        StringBuilder s = new StringBuilder();
        s.append("NN: Parameters: ").append(parameters()).append("\n");
        Layer tmp = inputLayer;
        while (tmp != null) {
            s.append(tmp.summary());
            tmp = tmp.nextLayer;
        }

        System.out.println(s);

    }

    public void printSummaryBackward() {
        StringBuilder s = new StringBuilder();
        Layer tmp = outputLayer;
        while (tmp != null) {
            s.append(tmp.summary());
            tmp = tmp.getPreviousLayer();
        }

        System.out.println(s);

    }


    public String export() {

        Layer[] tmp = layers2Array();

        String s = "NN;" + tmp.length + "\n";
        for (int i = 0; i < tmp.length; i++) {
            if (i != tmp.length - 1) {
                s += tmp[i].export() + "\n";
            } else {
                s += tmp[i].export();
            }

        }

        return s;
    }

    public void writeModelFast(String fileName) {


        Layer[] tmp = layers2Array();

        System.out.println("Started Writing weights.");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            String s = "NN;" + tmp.length + ";" + parameters() + "\n";
            writer.write(s);
            for (int i = 0; i < tmp.length; i++) {
                if (i != tmp.length - 1) {
                    s = tmp[i].export() + "\n";
                    writer.write(s);
                } else {
                    s = tmp[i].export();
                    writer.write(s);
                }

            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        System.out.println("wrote weights: " + fileName);

    }

    public void writeModel(String fileName) {

        String s = this.export();

        System.out.println("Started Writing weights.");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            writer.write(s);
            System.out.println("wrote weights: " + fileName);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Layer getInputLayer() {
        return inputLayer;
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }


    public boolean isEqual(NeuralNetwork other) {

        if (other.getLayers().length != this.getLayers().length) {
            System.out.println("Layers have different Length: this: " + this.getLayers().length + " other: " + other.getLayers().length);
            return false;
        }

        Layer[] thisLayers = this.getLayers();
        Layer[] otherLayers = other.getLayers();
        for (int i = 0; i < thisLayers.length; i++) {

            if (!thisLayers[i].isEqual(otherLayers[i])) {
                System.out.println("Layer on Position: " + i + " was different.");
                System.out.println("this: " + thisLayers[i].export());
                System.out.println("other: " + otherLayers[i].export());
                return false;
            }

        }

        return true;
    }


}
