package function;

import java.util.*;

public class Dropout {

    boolean training = true;

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

    public void setTraining(boolean training) {
        this.training = training;
    }

    public double getRate() {
        return this.rate;
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

        Integer[] tmp = new Integer[numbersNeeded];
        Iterator iter = generated.iterator();

        int count = 0;
        while (iter.hasNext()) {
            tmp[count] = (Integer) iter.next();
            count += 1;
        }


        this.input1D = new double[maxNumber];

        Arrays.fill(input1D, 1);

        for (int i = 0; i < numbersNeeded; i++) {
            input1D[Integer.valueOf(tmp[i])] = 0;

        }
    }


    public double[] forward(double[] a) {

        if (training) {

            this.input1D = new double[a.length];
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

        if (training) {
            this.input2D = new double[a.length][a[0].length];
            for (int i = 0; i < a.length; i++) {
                a[i] = this.forward(a[i]);
                this.input2D[i] = this.input1D;
            }
        }


        return a;
    }

    public double[][][] forward(double[][][] a) {

        if (training) {
            this.input3D = new double[a.length][][];
            for (int i = 0; i < a.length; i++) {
                a[i] = this.forward(a[i]);
                this.input3D[i] = this.input2D;
            }
        }


        return a;
    }

    public double[][][][] forward(double[][][][] a) {

        if (training) {

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

    /**
     * is meant to use to have still the same good Speed performance.
     *
     * @param a
     * @return
     */
    public double backward(double a, int position) {

        if (this.input1D[position] == 0) {
            return 0;
        }
        return a;

    }

    public double backward(double a, int i, int j) {

        if (this.input2D[i][j] == 0) {
            return 0;
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
