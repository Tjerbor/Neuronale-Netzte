package utils;

import extraLayer.FullyConnectedLayer;
import main.LayerNew;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * This class contains utility methods for creating a neural network from a CSV file.
 */
public class Reader {
    /**
     * This method attempts to read the given CSV file and return the values it contains.
     * It throws an exception if the file does not exist or an I/O error occurs.
     */
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

    /**
     * This method attempts to read the given CSV file and create a neural network with the values it contains.
     * It throws an exception if the file does not exist or an I/O error occurs.
     */
    public static FullyConnectedLayer[] create(String path) throws IOException {
        List<String[]> list = read(path);

        if (!list.get(0)[0].equals("layers")) {
            throw new IllegalArgumentException("The file must start with the keyword \"layers\".");
        }

        int[] topologie = IntStream.range(1, list.get(0).length).map(i -> Integer.parseInt(list.get(0)[i])).toArray();

        System.out.println(Arrays.toString(topologie));
        FullyConnectedLayer[] structur = new FullyConnectedLayer[topologie.length - 1];

        int count = 0;
        int list_pos = 1; //da die erste Zeile die Topologie bestimmt.

        for (int i = 0; i < (topologie.length - 1); i++) {
            double[][] w = Utils.genRandomWeights(topologie[i] + 1, topologie[i + 1]);

            structur[i] = new FullyConnectedLayer(topologie[i], topologie[i + 1]);
            structur[i].setUseBiases(true);

            for (int j = 0; j < (topologie[i]) + 1; j++) {
                for (int wi = 0; wi < list.get(list_pos).length; wi++) {
                    w[j][wi] = Double.parseDouble(list.get(list_pos)[wi]);
                }

                list_pos += 1; //update layer position.
            }

            list_pos += 1; //wegen der leerzeile.
            //structur[i + count].weights = Utils.split_for_weights(w);
            structur[i].setWeights(w);
            //structur[i + count].biases = Utils.split_for_biases(w);
            count += 1;
        }

        return structur;
    }

    public static LayerNew[] createNew(String path) throws IOException {
        List<String[]> list = read(path);

        if (!list.get(0)[0].equals("layers")) {
            throw new IllegalArgumentException("The file must start with the keyword \"layers\".");
        }

        int[] topologie = IntStream.range(1, list.get(0).length).map(i -> Integer.parseInt(list.get(0)[i])).toArray();

        System.out.println(Arrays.toString(topologie));
        LayerNew[] structur = new LayerNew[topologie.length - 1];

        int count = 0;
        int list_pos = 1; //da die erste Zeile die Topologie bestimmt.

        for (int i = 0; i < (topologie.length - 1); i++) {
            double[][] w = Utils.genRandomWeights(topologie[i] + 1, topologie[i + 1]);

            structur[i] = new FullyConnectedLayer(topologie[i], topologie[i + 1]);

            for (int j = 0; j < (topologie[i]) + 1; j++) {
                for (int wi = 0; wi < list.get(list_pos).length; wi++) {
                    w[j][wi] = Double.parseDouble(list.get(list_pos)[wi]);
                }

                list_pos += 1; //update layer position.
            }


            list_pos += 1; //wegen der leerzeile.
            //structur[i + count].weights = Utils.split_for_weights(w);
            structur[i].setWeights(new Matrix(w));
            //structur[i + count].biases = Utils.split_for_biases(w);
            count += 1;
        }

        return structur;

    }

    public static double[][] getTrainDataOutputs(String path, int outputsSize) throws IOException {


        List<String[]> list = read(path);
        int list_size = list.size(); //wird verwendet, damit sichergestellt ist,
        // dass die letzte Zeile keine Leerzeile ist.
        if (Arrays.toString(list.get(list_size - 1)).equals("")) {
            list_size -= 1;
        }

        double[][] outputs = new double[list_size][outputsSize];
        String s1;
        String s;
        int count = 0; //notwenidg weil die i die Position in der Liste ist.
        boolean start = false;
        for (int j = 0; j < list.size(); j++) {
            //System.out.println(Arrays.deepToString(list.get(j)));
            for (int i = 0; i < list.get(j).length; i++) {
                s = list.get(j)[i];

                if (start) {
                    //System.out.println(s);
                    outputs[j][count] = Double.parseDouble(s);
                    count += 1;
                } else {
                    s1 = list.get(j)[i].strip();
                    if (!s.equals(s1)) {
                        start = true;
                        //System.out.println(s);
                        outputs[j][count] = Double.parseDouble(s1);
                        count += 1;
                    }

                }
                if (i == list.get(j).length - 1) {
                    start = false; //needs to be reseted for the next Line
                    count = 0;
                }

            }
        }
        return outputs;
    }

    public static double[][] getTrainDataInputs(String path, int inputSize) throws IOException {


        List<String[]> list = read(path);
        int list_size = list.size(); //wird verwendet, damit sichergestllt ist,
        // dass die letzte Zeile keine Leerezeile ist.
        if (Arrays.toString(list.get(list_size - 1)).equals("")) {
            list_size -= 1;
        }
        double[][] inputs = new double[list_size][inputSize];
        String s1;
        String s;
        for (int j = 0; j < list_size; j++) {
            for (int i = 0; i < list.get(j).length; i++) {
                s = list.get(j)[i];
                s1 = list.get(j)[i].strip();
                if (s.equals(s1)) {
                    inputs[j][i] = Double.parseDouble(s);
                } else {
                    break;
                }
            }
        }
        return inputs;
    }

}
