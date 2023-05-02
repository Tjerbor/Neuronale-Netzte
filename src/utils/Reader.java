package utils;

import layers.FullyConnectedLayer;
import layers.Layer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
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
    public static Layer[] create(String path) throws IOException {
        List<String[]> list = read(path);

        if (!list.get(0)[0].equals("layers")) {
            throw new IllegalArgumentException("The file must start with the keyword \"layers\".");
        }

        int[] topologie = IntStream.range(1, list.get(0).length).map(i -> Integer.parseInt(list.get(0)[i])).toArray();

        Layer[] structur = new Layer[(topologie.length - 1) * 2];

        int count = 0;
        int list_pos = 1; //da die erste Zeile die Topologie bestimmt.

        for (int i = 0; i < (topologie.length - 1); i++) {
            double[][] w = Utils.genRandomWeights(topologie[i] + 1, topologie[i + 1]);

            structur[count] = new FullyConnectedLayer(topologie[i], topologie[i + 1]);
            structur[count + 1] = Utils.getActivation();

            for (int j = 0; j < (topologie[i]) + 1; j++) {
                for (int wi = 0; wi < list.get(list_pos).length; wi++) {
                    w[j][wi] = Double.parseDouble(list.get(list_pos)[wi]);
                }

                list_pos += 1; //update layer position.
            }

            list_pos += 1; //wegen der leerzeile.
            //structur[i + count].weights = Utils.split_for_weights(w);
            structur[count].weights = w;
            //structur[i + count].biases = Utils.split_for_biases(w);
            count += 2;
        }

        return structur;
    }
}
