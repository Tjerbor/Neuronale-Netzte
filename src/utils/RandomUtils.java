package utils;

import java.util.Random;

public class RandomUtils {

    static Random r = new Random(); //random to generate missing weights.

    public static double genRandomWeight() {
        return r.nextDouble(-0.1, 0.1);
    }

    public static double genGaussianRandomWeight() {
        return r.nextGaussian();
    }

    public static double genGaussianRandomWeight(double mean, double std) {
        return r.nextGaussian(mean, std);
    }

    public static void genRandomWeight(double[][][][] d) {


        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {
                for (int k = 0; k < d[0][0].length; k++) {
                    for (int l = 0; l < d[0][0][0].length; l++) {
                        d[i][j][k][l] = genRandomWeight();
                    }

                }
            }
        }


    }

    public static void genGaussianRandomWeight(double[][][][] d) {


        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {
                for (int k = 0; k < d[0][0].length; k++) {
                    for (int l = 0; l < d[0][0][0].length; l++) {
                        d[i][j][k][l] = genGaussianRandomWeight();
                    }

                }
            }
        }


    }

    public static void genRandomWeightConv(double[][][][] d) {


        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {
                for (int k = 0; k < d[0][0].length; k++) {
                    for (int l = 0; l < d[0][0][0].length; l++) {
                        d[i][j][k][l] = r.nextDouble(-0.1, 0.1);
                        ;
                    }

                }
            }
        }


    }

    public static void genGaussianRandomWeight(double[][][][] d, double mean, double std) {


        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {
                for (int k = 0; k < d[0][0].length; k++) {
                    for (int l = 0; l < d[0][0][0].length; l++) {
                        d[i][j][k][l] = genGaussianRandomWeight(mean, std);
                    }

                }
            }
        }


    }

    public static void genRandomWeight(double[][][] d) {


        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {
                for (int k = 0; k < d[0][0].length; k++) {
                    d[i][j][k] = genRandomWeight();


                }
            }
        }


    }

    public static void genRandomWeight(double[][] d) {


        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {

                d[i][j] = genRandomWeight();
            }
        }


    }

    public static void genRandomWeight(double[] d) {


        for (int i = 0; i < d.length; i++) {


            d[i] = genRandomWeight();
        }


    }

    public static void genRandomWeightConv(double[] d) {


        for (int i = 0; i < d.length; i++) {


            d[i] = r.nextDouble(-1, 1);
        }


    }


    public static double[][][] genRandomWeight(int[] a) {

        double[][][] c = new double[a[0]][a[1]][a[2]];

        for (int i = 0; i < a[0]; i++) {
            for (int j = 0; j < a[1]; j++) {
                for (int k = 0; k < a[2]; k++) {
                    c[i][j][k] = genRandomWeight();
                }
            }
        }

        return c;


    }

    public static void genTypeWeights(int type, double[][][][] weights) {

        if (type == 0) {

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = r.nextGaussian(0, 1);
                        }

                    }

                }

            }


        } else if (type == 1) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = r.nextGaussian(0, 2);
                        }

                    }

                }

            }

        } else if (type == 2) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = r.nextDouble(-0.1, 0.1);
                        }

                    }

                }

            }
        } else if (type == 3) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = r.nextDouble(-0.01, 0.01);
                        }

                    }

                }

            }
        } else {

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = r.nextDouble(-1, 1);
                        }

                    }

                }

            }

        }
    }

    public static void genTypeWeights(int type, double[][][] weights) {

        if (type == 0) {

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        weights[i][j][k] = r.nextGaussian();
                    }

                }

            }


        } else if (type == 1) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        weights[i][j][k] = r.nextGaussian(0, 2);
                    }

                }

            }

        } else if (type == 2) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        weights[i][j][k] = r.nextDouble(-0.1, 0.1);
                    }

                }

            }
        } else if (type == 3) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        weights[i][j][k] = r.nextDouble(-0.01, 0.01);
                    }

                }

            }
        } else {

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        weights[i][j][k] = r.nextDouble(-1, 1);
                    }

                }

            }

        }
    }


    public static void genTypeWeights(int type, double[][] weights) {

        if (type == 0) {


            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = r.nextGaussian(0, 1);
                }
            }


        } else if (type == 1) {
            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = r.nextGaussian(0, 2);
                }
            }

        } else if (type == 2) {
            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = r.nextDouble(-0.1, 0.1);
                }
            }

        } else if (type == 3) {
            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = r.nextDouble(-0.01, 0.01);
                }


            }
        } else {

            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = r.nextDouble(-0.1, 0.1);
                }

            }

        }
    }

    public static void genTypeWeights(int type, double[] weights) {

        if (type == 0) {

            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextGaussian();
            }

        } else if (type == 1) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextGaussian(0, 2);
            }

        } else if (type == 2) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextDouble(-0.1, 0.1);
            }

        } else if (type == 3) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextDouble(-0.01, 0.01);
            }
        } else {

            for (int i = 0; i < weights.length; i++) {
                weights[i] = r.nextDouble(-1, 1);
            }


        }
    }


}
