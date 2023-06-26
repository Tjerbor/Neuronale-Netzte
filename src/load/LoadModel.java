package load;

import extraLayer.*;
import layer.Activation;
import layer.ActivationLayer;
import layer.ReLu;
import layer.TanH;
import main.LayerNew;
import main.NN_New;
import utils.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LoadModel {


    private static List<String> read(String path) throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(path))) {
            String line;

            List<String> list = new ArrayList<>();

            while ((line = in.readLine()) != null) {
                list.add(line.replace("\n", ""));
            }

            return list;
        }
    }

    public static Activation getActivation(String s) {

        s = s.toLowerCase();

        if (s.equals("tanh")) {
            return new TanH();
        } else if (s.equals("relu")) {
            return new ReLu();
        } else {
            return null;
        }

    }

    public static void getWeightsFromLine(double[] w, String s) {

        String[] ss = s.split(";");
        int count = 0;
        for (int i = 0; i < w.length; i++) {
            w[i] = Double.parseDouble(ss[count]);
            count += 1;
        }

    }

    public static void getWeightsFromLine(double[][] w, String s) {

        String[] ss = s.split(";");
        int count = 0;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = Double.parseDouble(ss[count]);
                count += 1;
            }
        }

    }

    public static void getWeightsFromLine(double[][][][] w, String s) {

        String[] ss = s.split(";");
        int count = 0;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    for (int l = 0; l < w[0][0][0].length; l++) {
                        w[i][j][k][l] = Double.parseDouble(ss[count]);
                        count += 1;
                    }
                }

            }
        }

    }

    public static void getWeightsFromLine(double[][][] w, String s) {

        String[] ss = s.split(";");
        int count = 0;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {

                    w[i][j][k] = Double.parseDouble(ss[count]);
                    count += 1;
                }
            }
        }

    }

    public static LayerNew loadMaxPooling2D_Last(String[] config) {
        //first Starts with name, poolSize1, poolSize2, stride1, stride2,inputShape(1, 2, 3)
        int[] inputShape = new int[]{Integer.parseInt(config[5]), Integer.parseInt(config[6]), Integer.parseInt(config[7])};
        int[] poolSize = new int[]{Integer.parseInt(config[1]), Integer.parseInt(config[2])};
        int[] strides = new int[]{Integer.parseInt(config[3]), Integer.parseInt(config[4])};
        return new MaxPooling2D_Last(inputShape, poolSize, strides);


    }

    public static LayerNew loadFlatten(String[] config) {
        int[] inputShape = new int[]{Integer.parseInt(config[1]), Integer.parseInt(config[2]), Integer.parseInt(config[3])};
        return new Flatten(inputShape);
    }

    public static LayerNew loadFastLinearLayer(String[] config, String nextLine) {

        /**
         * start with the name, weight.length, weight[0].length, act
         */

        int a = Integer.parseInt(config[1]);
        int b = Integer.parseInt(config[2]);
        FastLinearLayer f = new FastLinearLayer(a, b);
        if (config.length == 4) {
            Activation act = getActivation(config[3]);
            f.setActivation(act);
        }


        double[][] w = new double[a][b];
        getWeightsFromLine(w, nextLine);

        f.setWeights(new Matrix<>(w));
        return f;

    }

    public static LayerNew loadFullyConnectedLayer(String[] config, String nextLine) {

        /**
         * start with the name, useBiases, weight.length, weight[0].length, act, dropoutRate
         */

        int a = Integer.parseInt(config[1]);
        int b = Integer.parseInt(config[2]);
        FastLinearLayer f = new FastLinearLayer(a, b);
        if (config.length >= 5) {
            Activation act = getActivation(config[3]);
            if (act != null) {
                f.setActivation(act);
            }
        }

        if (config.length == 6) {
            f.setDropout(Double.parseDouble(config[5]));
        }

        f.setUseBiases(false);


        double[][] w = new double[a][b];
        getWeightsFromLine(w, nextLine);

        f.setWeights(new Matrix<>(w));
        return f;

    }

    public static LayerNew loadFullyConnectedLayer(String[] config, String nextLine, String biasesLine) {

        /**
         * start with the name, useBiases, weight.length, weight[0].length, act, dropoutRate
         */

        int a = Integer.parseInt(config[1]);
        int b = Integer.parseInt(config[2]);
        FastLinearLayer f = new FastLinearLayer(a, b);
        if (config.length >= 5) {
            Activation act = getActivation(config[3]);
            if (act != null) {
                f.setActivation(act);
            }
        }

        if (config.length == 6) {
            f.setDropout(Double.parseDouble(config[5]));
        }

        f.setUseBiases(false);


        double[][] w = new double[a][b];
        double[] biases = new double[b];
        getWeightsFromLine(w, nextLine);
        getWeightsFromLine(biases, biasesLine);

        f.setWeights(new Matrix<>(w));
        return f;

    }

    public static LayerNew loadConv2D_Last(String[] config, String nextLine) {

        //congig -> name, useBiases, numFilter, kernelSize1, kernelSize2, strides1, strides2, inputShape1, inputShape2, inputShape3

        int[] inputShape = new int[]{Integer.parseInt(config[7]), Integer.parseInt(config[8]), Integer.parseInt(config[9])};

        int[] kernelsSize = new int[]{Integer.parseInt(config[3]), Integer.parseInt(config[4])};
        int[] strides = new int[]{Integer.parseInt(config[5]), Integer.parseInt(config[6])};

        int numFilter = Integer.parseInt(config[2]);

        double[][][][] w = new double[kernelsSize[0]][kernelsSize[1]][inputShape[2]][numFilter];
        getWeightsFromLine(w, nextLine);

        Conv2D_Last c = new Conv2D_Last(inputShape, numFilter, kernelsSize, strides);

        c.setWeights(w);
        c.setUseBiases(false);
        return c;


    }

    public static LayerNew loadActivation(String[] config) {
        return new ActivationLayer(config[1]);

    }

    public static LayerNew loadConv2D_Last(String[] config, String nextLine, String biasesLine) {

        //congig -> name, useBiases, numFilter, kernelSize1, kernelSize2, strides1, strides2, inputShape1, inputShape2, channels

        int[] inputShape = new int[]{Integer.parseInt(config[7]), Integer.parseInt(config[8]), Integer.parseInt(config[9])};

        int[] kernelsSize = new int[]{Integer.parseInt(config[3]), Integer.parseInt(config[4])};
        int[] strides = new int[]{Integer.parseInt(config[5]), Integer.parseInt(config[6])};

        int numFilter = Integer.parseInt(config[2]);

        double[][][][] w = new double[kernelsSize[0]][kernelsSize[1]][inputShape[2]][numFilter];

        Conv2D_Last c = new Conv2D_Last(inputShape, numFilter, kernelsSize, strides);

        double[] b = new double[numFilter];
        getWeightsFromLine(b, biasesLine);
        getWeightsFromLine(w, nextLine);

        c.setUseBiases(true);
        c.setWeights(w, b);

        return c;


    }

    public static LayerNew loadConv2D(String[] config, String nextLine) {

        //congig -> name, useBiases, numFilter, kernelSize1, kernelSize2, strides1, strides2, inputShape1, inputShape2, inputShape3

        int[] inputShape = new int[]{Integer.parseInt(config[7]), Integer.parseInt(config[8]), Integer.parseInt(config[9])};

        int[] kernelsSize = new int[]{Integer.parseInt(config[3]), Integer.parseInt(config[4])};
        int[] strides = new int[]{Integer.parseInt(config[5]), Integer.parseInt(config[6])};

        int numFilter = Integer.parseInt(config[2]);

        double[][][][] w = new double[kernelsSize[0]][kernelsSize[1]][inputShape[0]][numFilter];
        getWeightsFromLine(w, nextLine);

        Conv2D c = new Conv2D(inputShape, numFilter, kernelsSize, strides);

        c.setWeights(w);
        c.setUseBiases(false);
        return c;


    }

    public static LayerNew loadConv2D(String[] config, String nextLine, String biasesLine) {

        //congig -> name, useBiases, numFilter, kernelSize1, kernelSize2, strides1, strides2, inputShape1, inputShape2, inputShape3

        int[] inputShape = new int[]{Integer.parseInt(config[7]), Integer.parseInt(config[8]), Integer.parseInt(config[9])};

        int[] kernelsSize = new int[]{Integer.parseInt(config[3]), Integer.parseInt(config[4])};
        int[] strides = new int[]{Integer.parseInt(config[5]), Integer.parseInt(config[6])};

        int numFilter = Integer.parseInt(config[2]);
        Conv2D c = new Conv2D(inputShape, numFilter, kernelsSize, strides);

        int[] outputShape = c.getOutputShape();
        double[][][][] w = new double[kernelsSize[0]][kernelsSize[1]][inputShape[0]][numFilter];
        double[][][] b = new double[outputShape[0]][outputShape[1]][outputShape[2]];
        getWeightsFromLine(b, biasesLine);
        getWeightsFromLine(w, nextLine);


        c.setUseBiases(true);
        c.setWeights(w, b);

        return c;


    }

    public static LayerNew loadBatchNorm(String[] config, String nextLine) {

        //config name, Size, inputShape Length 1 oder 3
        int inputSize = Integer.parseInt(config[1]);
        String[] w = nextLine.split(";");

        double[][] weights;
        if (w.length == inputSize * 4) {
            weights = new double[4][inputSize];
            getWeightsFromLine(weights, nextLine);

        } else if (w.length == inputSize * 2) {
            weights = new double[2][inputSize];
            getWeightsFromLine(weights, nextLine);
        } else {
            throw new IllegalArgumentException("BatchNorm Weight-Size error." + "got Size weights: " + w.length + " and inputSize: " + inputSize);
        }
        BatchNorm b = new BatchNorm(inputSize);
        b.setWeights(new Matrix(weights));
        if (config.length == 3) {
            b.setInputShape(new int[]{Integer.parseInt(config[2])});
        } else if (config.length == 5) {
            int[] inputShape = new int[]{Integer.parseInt(config[2]), Integer.parseInt(config[3]), Integer.parseInt(config[4])};
            b.setInputShape(inputShape);
        }
        return b;
    }

    public static LayerNew loadDropout(String[] config) {
        //config -> name, rate
        return new DropoutLayer(Double.parseDouble(config[1]));
    }

    public NN_New loadModel(String fpath) throws IOException {

        List<String> lines = read(fpath);

        List<LayerNew> layers = new ArrayList<>();

        int size = lines.size();

        int LineCount = 1;

        while (true) {

            String[] config = lines.get(LineCount).split(";");

            if (LineCount >= lines.size()) {
                break;
            }

            if (config[0].equals("conv2d_last")) {
                if (config[1].equals("true")) {
                    layers.add(loadConv2D_Last(config, lines.get(LineCount + 1), lines.get(LineCount + 2)));
                    LineCount += 2;
                } else {
                    layers.add(loadConv2D_Last(config, lines.get(LineCount + 1)));
                    LineCount += 1;
                }
                LineCount += 1;

            } else if (config[0].equals("fullyconnectedlayer")) {
                if (config[1].equals("true")) {
                    layers.add(loadConv2D(config, lines.get(LineCount + 1), lines.get(LineCount + 2)));
                    LineCount += 2;
                } else {
                    layers.add(loadConv2D(config, lines.get(LineCount + 1)));
                    LineCount += 1;
                }
                LineCount += 1;

            } else if (config[0].equals("conv2d")) {
                if (config[1].equals("true")) {
                    layers.add(loadConv2D(config, lines.get(LineCount + 1), lines.get(LineCount + 2)));
                    LineCount += 2;
                } else {
                    layers.add(loadConv2D(config, lines.get(LineCount + 1)));
                    LineCount += 1;
                }
                LineCount += 1;

            } else if (config[0].equals("batchnorm")) {

                layers.add(loadBatchNorm(config, lines.get(LineCount + 1)));
                LineCount += 2;
            } else if (config[0].equals("dropout")) {
                layers.add(loadDropout(config));
                LineCount += 1;

            } else if (config[0].equals("fastlinearlayer")) {
                layers.add(loadFastLinearLayer(config, lines.get(LineCount + 1)));
                LineCount += 2;


            } else if (config[0].equals("flatten")) {
                layers.add(loadFlatten(config));
            } else if (config[0].equals("maxpooling2d_last")) {
                layers.add(loadMaxPooling2D_Last(config));
                LineCount += 1;
            } else if (config[0].equals("maxpooling2d")) {
                layers.add(loadMaxPooling2D_Last(config));
                LineCount += 1;
            } else {
                LineCount += 1;
            }

        }

        NN_New nn = new NN_New();

        LayerNew[] l = new LayerNew[layers.size()];

        for (int i = 0; i < l.length; i++) {
            l[i] = layers.get(i);
        }

        nn.setLayers(l);
        return nn;
    }


}
