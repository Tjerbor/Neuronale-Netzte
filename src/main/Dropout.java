package main;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;

public class Dropout {

    boolean Training = false;

    int size; //the size of the deactivate neurons.
    //no make sure teh number is exact and not rounded.

    double[] input1D;
    double[][] input2D;
    double[][][] input3D;
    double[][][][] input4D;
    private double rate;

    /**
     * expected the percentage to deactivate.
     * exp. 0.5
     *
     * @param percentage
     */
    public Dropout(double percentage) {
        rate = percentage;
    }


    private void genNoDuplicateRandom(int numbersNeeded, int maxNumber) {

        if (maxNumber < numbersNeeded) {
            throw new IllegalArgumentException("Can't ask for more numbers than are available");
        }
        Random rng = new Random(); // Ideally just create one instance globally
// Note: use LinkedHashSet to maintain insertion order
        Set<Integer> generated = new LinkedHashSet<Integer>();
        while (generated.size() < numbersNeeded) {
            Integer next = rng.nextInt(maxNumber) + 1;
            // As we're adding to a set, this will automatically do a containment check
            generated.add(next);
        }

        Integer[] tmp = (Integer[]) generated.toArray();
        this.input1D = new double[maxNumber];

        Arrays.fill(input1D, 1);

        for (int i = 0; i < numbersNeeded; i++) {
            input1D[tmp[i]] = 0;

        }
    }

    public double[] forward(double[] a) {

        if (Training) {

            int size;
            if (this.size == 0) {
                size = (int) Math.floor(a.length * this.rate);

            } else {
                size = this.size;
            }
            this.genNoDuplicateRandom(size, a.length);
            for (int i = 0; i < a.length; i++) {
                if (this.input1D[i] == 0) {
                    a[i] = 0;
                }


            }
        }

        return a;
    }

    public double[][] forward(double[][] a) {

        if (Training) {

            this.input2D = new double[a.length][];
            for (int i = 0; i < a.length; i++) {
                a[i] = this.forward(a[i]);
                this.input2D[i] = this.input1D;
            }
        }


        return a;
    }

    public double[][][] forward(double[][][] a) {

        if (Training) {
            this.input3D = new double[a.length][][];
            for (int i = 0; i < a.length; i++) {
                a[i] = this.forward(a[i]);
                this.input3D[i] = this.input2D;
            }
        }


        return a;
    }

    public double[][][][] forward(double[][][][] a) {

        if (Training) {

            this.input4D = new double[a.length][][][];
            for (int i = 0; i < a.length; i++) {
                a[i] = this.forward(a[i]);
                this.input4D[i] = this.input3D;
            }
        }

        return a;
    }

    public double[] backward(double[] a) {

        for (int i = 0; i < a.length; i++) {
            if (this.input1D[i] == 0) {
                a[i] = 0;
            }


        }

        return a;
    }

    public double[][] backward(double[][] a) {

        for (int i = 0; i < a.length; i++) {
            this.input1D = this.input2D[i];
            this.backward(a[i]);
        }
        return a;
    }

    public double[][][] backward(double[][][] a) {

        for (int i = 0; i < a.length; i++) {
            this.input2D = this.input3D[i];
            this.backward(a[i]);
        }
        return a;
    }

    public double[][][][] backward(double[][][][] a) {

        for (int i = 0; i < a.length; i++) {
            this.input3D = this.input4D[i];
            this.backward(a[i]);
        }
        return a;
    }


}
