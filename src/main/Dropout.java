package main;

import java.util.Random;

public class Dropout {

    boolean Training = false;
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


    public double[] forward(double[] a) {

        if (Training) {
            Random random = new Random();
            int size = (int) Math.floor(a.length * this.rate);


            int[] out = new int[size];
            for (int i = 0; i < size; i++) {
                out[random.nextInt(0, a.length - 1)] = 1;
            }

            for (int i = 0; i < a.length; i++) {
                if (out[i] == 0) {
                    a[i] = 0;
                }


            }
        }


        return a;
    }


}
