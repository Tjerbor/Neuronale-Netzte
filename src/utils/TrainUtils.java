package utils;

import java.util.*;

public class TrainUtils {


    public static double[][][] shuffle(double[][] x_train, double[][] y_train) {

        double[][][] out = new double[2][x_train.length][];
        Object[] tmp = genRandomOrder(x_train.length);

        for (int i = 0; i < x_train.length; i++) {
            int index = (Integer) tmp[i];
            out[0][i] = x_train[index];
            out[1][i] = y_train[index];
        }

        return out;

    }

    public static Object[] genRandomOrder(int numbersNeeded) {

        Random rng = new Random(); // Ideally just create one instance globally
// Note: use LinkedHashSet to maintain insertion order
        Set<Integer> generated = new LinkedHashSet<Integer>();
        while (generated.size() < numbersNeeded) {
            Integer next = rng.nextInt(numbersNeeded);
            generated.add(next);
        }

        return generated.toArray();
    }


    public static double[][][] reshapeToChannelsLast(double[] x, int[] shape) {
        int count = 0;
        double[][][] c = new double[shape[0]][shape[1]][shape[2]];
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    c[i][j][k] = x[count];
                    count += 1;
                }
            }
        }

        return c;
    }

    public static List yieldXY_Last(int batchSize, double[][] x_train, double[][] y_train, int step) {


        double[][][][] tmpX = new double[batchSize][][][];
        double[][] tmpY = new double[batchSize][];


        for (int i = 0; i < batchSize; i++) {
            tmpX[i] = reshapeToChannelsLast(x_train[step + i], new int[]{28, 28, 1});
            tmpY[i] = y_train[step + i];

        }

        int s = x_train.length / batchSize;
        
        List e = new ArrayList();

        e.add(tmpX);
        e.add(tmpY);


        return e;

    }

}
