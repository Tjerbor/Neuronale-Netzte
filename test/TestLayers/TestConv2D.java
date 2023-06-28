package TestLayers;

import layer.Conv2D_Last;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * can read Weights for the Layer.
 * missing Testing Scripts. TODO needs to be added.
 */

public class TestConv2D {


    private static List<String[]> read(String path) throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(path))) {
            String line;

            List<String[]> list = new ArrayList<>();

            while ((line = in.readLine()) != null) {
                String[] values = line.split(";");

                list.add(values);
            }

            return list;
        }
    }


    public static List<Object> readFromFile(String fpath) throws IOException {

        List<String[]> data = read(fpath);

        //0 Dimesnion is config
        //1 is weighst
        //2biases  zeros.
        //3 BatchSize
        //4 input
        //5 output


        //conv2D;useBiases;numFillter,kernelSize1,kernelSize2,stride1,stride2,inputShape1-3
        int[] config = IntStream.range(1, data.get(0).length).map(i -> Integer.parseInt(data.get(0)[i])).toArray();
        String[] w = data.get(1);

        String[] output = data.get(5);
        String[] inputD = data.get(4);


        Conv2D_Last conv2D = new Conv2D_Last(config);


        double[][][][] weights = new double[config[1]][config[1]][config[config.length - 1]][config[0]];


        int count = 0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                for (int k = 0; k < weights[0][0].length; k++) {
                    for (int l = 0; l < weights[0][0][0].length; l++) {
                        weights[i][j][k][l] = Double.parseDouble(w[count]);
                        count += 1;
                    }
                }
            }
        }

        double[] biases = new double[config[0]];
        for (int i = 0; i < config[0]; i++) {
            biases[i] = Double.parseDouble(data.get(2)[i]);
        }


        int[] outShape = conv2D.getOutputShape();

        int BatchSize = Integer.parseInt(data.get(3)[1]);
        int n = config[config.length - 3];
        int m = config[config.length - 2];
        int channels = config[config.length - 1];


        double[][][][] input = new double[BatchSize][n][m][channels];

        count = 0;
        for (int i = 0; i < BatchSize; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    for (int l = 0; l < channels; l++) {
                        input[i][j][k][l] = Double.parseDouble(inputD[count]);
                        count += 1;
                    }
                }
            }
        }


        double[][][][] soll = new double[BatchSize][outShape[0]][outShape[1]][outShape[2]];


        count = 0;
        for (int i = 0; i < BatchSize; i++) {
            for (int j = 0; j < outShape[0]; j++) {
                for (int k = 0; k < outShape[1]; k++) {
                    for (int l = 0; l < outShape[2]; l++) {
                        soll[i][j][k][l] = Double.parseDouble(output[count]);
                        count += 1;
                    }
                }
            }
        }

        List<Object> out = new ArrayList<>();


        conv2D.setWeights(weights, biases);


        out.add(conv2D);
        out.add(input);
        out.add(soll);


        return out;

    }


}
