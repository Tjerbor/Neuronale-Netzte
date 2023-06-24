package Train;

import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;

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


}
