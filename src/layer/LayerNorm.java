package layer;

import utils.Array_utils;
import utils.Matrix;
import utils.Utils;

import java.util.Arrays;

import static load.writeUtils.writeShape;
import static load.writeUtils.writeWeights;

/**
 * has different weights with different Shape
 */
public class LayerNorm extends Layer {


    double epsilon = 1e-5;


    Matrix gamma;
    Matrix beta;
    Matrix std; //for now could also be 1D-Array because only supports around axis -1.

    Matrix x_norm;
    Matrix x_centered;

    boolean axisMinusOne = true;


    public LayerNorm(int shape) {
        double[] gamma = new double[shape];
        double[] beta = new double[shape];
        Arrays.fill(gamma, 1);

        this.gamma = new Matrix(gamma);
        this.beta = new Matrix(beta);

        inputShape = new int[]{shape};


    }

    public LayerNorm() {
    }

    //normally cleaner way but we only use this layer for Images,
    //expect BatchSize
    public LayerNorm(int[] shape) {

        this.axisMinusOne = false;

        double[][][] gamma = new double[shape[0]][shape[1]][shape[2]];
        double[][][] beta = new double[shape[0]][shape[1]][shape[2]];
        for (int i = 0; i < shape[2]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                Arrays.fill(gamma[i][j], 1);

            }
        }

        this.gamma = new Matrix(gamma);
        this.beta = new Matrix(beta);

        inputShape = shape;


    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setAxisMinusOne(boolean axisMinusOne, int shape) {
        this.axisMinusOne = axisMinusOne;

        if (axisMinusOne) {
            double[] gamma = new double[shape];
            double[] beta = new double[shape];
            Arrays.fill(gamma, 1);

            this.gamma = new Matrix(gamma);
            this.beta = new Matrix(beta);
        }

    }

    @Override
    public void forward(Matrix m) {


    }

    /**
     * expected input BatchSize, sequenceLength, embed-Dim.
     * RNN use case
     *
     * @param inputs
     * @return
     */
    public double[][][] forward(double[][][] inputs) {


        double[] mean = Array_utils.mean_axis_1_2(inputs);
        double[] var = Array_utils.var_axis_1_2(inputs);

        var = Array_utils.add(var, epsilon);


        double[] std = Array_utils.sqrtArray1D(var);


        //new Matrix(inputs).subDim3(mean.getData1D());
        double[][][] minusMean = Array_utils.sub3DRE(inputs, mean, 0);
        x_centered = new Matrix(minusMean);

        double[][][] standart_inputs = Array_utils.zerosLike(minusMean);

        for (int i = 0; i < standart_inputs.length; i++) {
            for (int j = 0; j < standart_inputs.length; j++) {
                for (int k = 0; k < standart_inputs.length; k++) {
                    standart_inputs[i][j][k] = minusMean[i][j][k] / std[k];
                }

            }
        }

        this.std = new Matrix(std);
        x_norm = new Matrix(standart_inputs);

        double[] gama = this.gamma.getData1D();
        double[] beta = this.beta.getData1D();

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs.length; k++) {
                    standart_inputs[i][j][k] = standart_inputs[i][j][k] * gama[j] + beta[i];
                }
            }
        }

        return standart_inputs;


    }

    /**
     * expect batchSize, inputs
     *
     * @param inputs
     * @return
     */
    public double[][] forward(double[][] inputs) {

        double[] mean = Array_utils.mean_axis_1(inputs);
        double[] var = Array_utils.mean_axis_1(inputs);
        ;
        var = Array_utils.addMatrixScalar(var, epsilon);

        double[] std = Array_utils.sqrtArrayRE(var);

        double[][] minusMean = Array_utils.sub2DRE(inputs, mean, 0);

        double[][] standart_inputs = Array_utils.copyArray(minusMean);

        for (int i = 0; i < standart_inputs.length; i++) {
            for (int j = 0; j < standart_inputs.length; j++) {
                standart_inputs[i][j] /= std[i];
            }
        }

        this.std = new Matrix(std);
        x_centered = new Matrix(minusMean);
        x_norm = new Matrix(standart_inputs);


        double[] gama = this.gamma.getData1D();
        double[] beta = this.beta.getData1D();

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                standart_inputs[i][j] = standart_inputs[i][j] * gama[j] + beta[i];

            }
        }


        return standart_inputs;


    }

    //use cases for Images not axis = -1. but axis 0.
    //uses batch-Input
    public double[][][][] forward(double[][][][] inputs) {

        double[][][] mean = Array_utils.mean4D_axis_0(inputs);
        double[][][] var = Array_utils.var4D_axis_0(inputs);
        ;
        var = Array_utils.add(var, epsilon);

        double[][][] std = Array_utils.sqrtArrayRE(var);

        double[][][][] minusMean = Array_utils.sub4D(inputs, mean);
        x_centered = new Matrix(minusMean);

        double[][][][] standart_inputs = Array_utils.zerosLike(minusMean);

        for (int i = 0; i < standart_inputs.length; i++) {
            for (int j = 0; j < standart_inputs[0].length; j++) {
                for (int k = 0; k < standart_inputs[0][0].length; k++) {
                    for (int l = 0; l < standart_inputs[0][0][0].length; l++) {
                        standart_inputs[i][j][k][l] = minusMean[i][j][k][l] / std[j][k][l];
                    }

                }

            }
        }

        this.std = new Matrix(std);
        x_norm = new Matrix(standart_inputs);

        double[][][] gama = this.gamma.getData3D();
        double[][][] beta = this.beta.getData3D();

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs.length; k++) {
                    for (int l = 0; l < inputs.length; l++) {
                        standart_inputs[i][j][k][l] = standart_inputs[i][j][k][l] * gama[j][k][l] + beta[i][k][l];
                    }

                }
            }
        }

        return standart_inputs;


    }

    public double[][][][] forwardAxis0(double[][][][] inputs) {

        double[] mean = Array_utils.mean_axis_1_2_3(inputs);
        double[] var = Array_utils.var_axis_1_2_3(inputs);
        var = Array_utils.add(var, epsilon);

        double[] std = Array_utils.sqrtArrayRE(var);

        double[][][][] minusMean = Array_utils.sub4D(inputs, mean);
        x_centered = new Matrix(minusMean);

        double[][][][] standart_inputs = Array_utils.zerosLike(minusMean);

        for (int i = 0; i < standart_inputs.length; i++) {
            for (int j = 0; j < standart_inputs[0].length; j++) {
                for (int k = 0; k < standart_inputs[0][0].length; k++) {
                    for (int l = 0; l < standart_inputs[0][0][0].length; l++) {
                        standart_inputs[i][j][k][l] = minusMean[i][j][k][l] / std[i];
                    }

                }

            }
        }

        this.std = new Matrix(std);
        x_norm = new Matrix(standart_inputs);

        double[][][] gama = this.gamma.getData3D();
        double[][][] beta = this.beta.getData3D();

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs.length; k++) {
                    for (int l = 0; l < inputs.length; l++) {
                        standart_inputs[i][j][k][l] = standart_inputs[i][j][k][l] * gama[j][k][l] + beta[i][k][l];
                    }

                }
            }
        }

        return standart_inputs;


    }

    public double[][] backward(double[][] grad_inputs) {

        int N = grad_inputs.length;
        double[] std = this.std.getData1D();
        double[][] x_centered = this.x_centered.getData2D();
        double[][] x_norm = this.x_norm.getData2D();

        double[] dgama = Array_utils.sum_axis_1(Array_utils.multiply2D(grad_inputs, x_norm));
        double[] dbeta = Array_utils.sum_axis_1(grad_inputs);

        double[] gamma = this.gamma.getData1D();
        double[][] dx_norm = Array_utils.zerosLike(grad_inputs);
        double[][] dx_centered = Array_utils.zerosLike(grad_inputs);


        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {

                dx_norm[i][j] = grad_inputs[i][j] * gamma[j];
                dx_centered[i][j] = dx_norm[i][j] / std[j];


            }
        }


        double[] dmean = Array_utils.add(Array_utils.sum_axis_1(dx_centered), (double) 2 / N);
        dmean = Utils.multiply(dmean, Array_utils.sum_axis_1(x_centered));

        double[][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                dstdTmp[i][j] = dx_norm[i][j] * x_centered[i][j] * -Math.pow(std[j], -2);
            }
        }


        double[] dstd = Array_utils.sum_axis_1(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            dvar[i] = dstd[i] / 2 / std[i];

        }
        double[][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {

                dx[i][j] = dx_centered[i][j] + (dmean[j] + dvar[j] * 2 * x_centered[i][j]) / N;
            }

        }


        return dx;


    }

    public double[][][] backward(double[][][] grad_inputs) {

        int N = grad_inputs.length;
        double[] std = this.std.getData1D();
        double[][][] x_centered = this.x_centered.getData3D();
        double[][][] x_norm = this.x_norm.getData3D();

        double[] dgama = Array_utils.sum_axis_1_2(Utils.multiply(grad_inputs, x_norm));
        double[] dbeta = Array_utils.sum_axis_1_2(grad_inputs);

        double[] gamma = this.gamma.getData1D();
        double[][][] dx_norm = Array_utils.zerosLike(grad_inputs);
        double[][][] dx_centered = Array_utils.zerosLike(grad_inputs);


        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    dx_norm[i][j][k] = grad_inputs[i][j][k] * gamma[k];
                    dx_centered[i][j][k] = dx_norm[i][j][k] / std[k];

                }
            }
        }


        double[][] dmean = Array_utils.addMatrixScalar(Array_utils.sum_axis_0(dx_centered), 2 / N);
        dmean = Utils.multiply(dmean, Array_utils.sum_axis_1(x_centered));

        double[][][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    dstdTmp[i][j][k] = dx_norm[i][j][k] * x_centered[i][j][k] * -Math.pow(std[j], -2);
                }

            }
        }


        double[] dstd = Array_utils.sum_axis_1_2(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {

            dvar[i] = dstd[i] / 2 / std[i];

        }

        double[][][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                for (int k = 0; k < dx[0][0].length; k++) {
                    dx[i][j][k] = dx_centered[i][j][k] + (dmean[i][k] + dvar[k] * 2 * x_centered[i][j][k]
                    ) / N;
                }

            }
        }
        return dx;


    }

    /**
     * backward for images
     *
     * @param grad_inputs
     * @return
     */
    public double[][][][] backward(double[][][][] grad_inputs) {

        int N = grad_inputs.length;
        double[][][] std = this.std.getData3D();
        double[][][][] x_centered = this.x_centered.getData4D();
        double[][][][] x_norm = this.x_norm.getData4D();

        double[][][] dgama = Array_utils.sum_axis_0_4D(Array_utils.multiply4D(grad_inputs, x_norm));
        double[][][] dbeta = Array_utils.sum_axis_0_4D(grad_inputs);

        double[] gamma = this.gamma.getData1D();
        double[][][][] dx_norm = Array_utils.zerosLike(grad_inputs);
        double[][][][] dx_centered = Array_utils.zerosLike(grad_inputs);


        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    for (int l = 0; l < dx_norm[0][0][0].length; l++) {
                        dx_norm[i][j][k][l] = grad_inputs[i][j][k][l] * gamma[l];
                        dx_centered[i][j][k][l] = dx_norm[i][j][k][l] / std[j][k][l];
                    }
                }
            }
        }


        double[][][] dmean = Array_utils.add(Array_utils.sum_axis_0_4D(dx_centered), 2 / N);
        dmean = Utils.multiply(dmean, Array_utils.sum_axis_0_4D(x_centered));

        double[][][][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    for (int l = 0; l < dx_norm[0][0][0].length; l++) {
                        dstdTmp[i][j][k][l] = dx_norm[i][j][k][l] * x_centered[i][j][k][l] * -Math.pow(std[j][k][l], -2);
                    }

                }

            }
        }


        double[][][] dstd = Array_utils.sum_axis_0(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[][][] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            for (int j = 0; j < dstd[0].length; j++) {
                for (int k = 0; k < dstd[0][0].length; k++) {
                    dvar[i][j][k] = dstd[i][j][k] / 2 / std[i][j][k];
                }
            }


        }
        double[][][][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                for (int k = 0; k < dx[0][0].length; k++) {
                    for (int l = 0; l < dx_norm[0][0][0].length; l++) {


                        dx[i][j][k][l] = dx_centered[i][j][k][l] + (dmean[j][k][l] + dvar[j][k][l] * 2 * x_centered[i][j][k][l]
                        ) / N;
                    }
                }

            }
        }

        this.updateParameter(dgama, dbeta);

        return dx;


    }


    public void updateParameter(double[][][] dgamma, double[][][] dbeta) {

        double[][][] gamma = this.gamma.getData3D();
        double[][][] beta = this.beta.getData3D();

        Utils.updateParameter(gamma, dgamma, learningRate);
        Utils.updateParameter(beta, dbeta, learningRate);

        this.gamma = new Matrix(gamma);
        this.beta = new Matrix(beta);

    }

    public void updateParameter(double[] dgamma, double[] dbeta) {

        double[] gamma = this.gamma.getData1D();
        double[] beta = this.beta.getData1D();

        Utils.updateParameter(gamma, dgamma, learningRate);
        Utils.updateParameter(beta, dbeta, learningRate);

        this.gamma = new Matrix(gamma);
        this.beta = new Matrix(beta);

    }


    public double[][][][] backwardAxis1_2_3(double[][][][] grad_inputs) {

        int N = grad_inputs.length;
        double[] std = this.std.getData1D();
        double[][][][] x_centered = this.x_centered.getData4D();
        double[][][][] x_norm = this.x_norm.getData4D();

        double[] dgama = Array_utils.sum_axis_1_2_3(Utils.multiply(grad_inputs, x_norm));
        double[] dbeta = Array_utils.sum_axis_1_2_3(grad_inputs);

        double[] gamma = this.gamma.getData1D();
        double[][][][] dx_norm = Array_utils.zerosLike(grad_inputs);
        double[][][][] dx_centered = Array_utils.zerosLike(grad_inputs);


        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    for (int l = 0; l < dx_norm[0][0][0].length; l++) {
                        dx_norm[i][j][k][l] = grad_inputs[i][j][k][l] * gamma[l];
                        dx_centered[i][j][k][l] = dx_norm[i][j][k][l] / std[l];
                    }
                }
            }
        }


        double[] dmean = Array_utils.add(Array_utils.sum_axis_1_2_3(dx_centered), 2 / N);
        dmean = Array_utils.multiply1D(dmean, Array_utils.sum_axis_1_2_3(x_centered));

        double[][][][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    for (int l = 0; l < dx_norm.length; l++) {
                        dstdTmp[i][j][k][l] = dx_norm[i][j][k][l] * x_centered[i][j][k][l] * -Math.pow(std[i], -2);
                    }

                }

            }
        }


        double[] dstd = Array_utils.sum_axis_1_2_3(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            dvar[i] = dstd[i] / 2 / std[i];

        }
        double[][][][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                for (int k = 0; k < dx[0][0].length; k++) {
                    for (int l = 0; l < dx_norm[0][0][0].length; l++) {


                        dx[i][j][k][l] = dx_centered[i][j][k][l] + (dmean[i] + dvar[l] * 2 * x_centered[i][j][k][l]
                        ) / N;
                    }
                }

            }
        }
        return dx;


    }


    @Override
    public void backward(Matrix m) {

    }

    @Override
    public void backward(Matrix m, double learningRate) {

    }

    @Override
    public Matrix getWeights() {
        return null;
    }

    @Override
    public void setWeights(Matrix m) {

    }

    @Override
    public String export() {
        
        String s = "layernorm;" + writeShape(inputShape) + "\n";


        if (this.gamma.getDim() == 1) {
            s += writeWeights(gamma.getData1D()) + "\n";
            s += writeWeights(beta.getData1D());


        } else if (gamma.getDim() == 2) {
            s += writeWeights(gamma.getData2D()) + "\n";
            s += writeWeights(beta.getData2D());
        } else if (gamma.getDim() == 3) {
            s += writeWeights(gamma.getData3D()) + "\n";
            s += writeWeights(beta.getData3D());
        }
        return s;
    }

    @Override
    public boolean isEqual(Layer other) {
        return false;
    }


    @Override
    public int parameters() {
        return Array_utils.sumUpMult(gamma.getShape()) * 2;
    }

    @Override
    public String summary() {
        return "LayerNorm inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameters: " + parameters() + "\n";
    }


    public void setAxis(int axis) {

        if (axis == 0 && inputShape.length == 3) {
            double[][][] g = new double[inputShape[0]][inputShape[1]][inputShape[2]];
            double[][][] b = new double[inputShape[0]][inputShape[1]][inputShape[2]];
            for (int i = 0; i < g.length; i++) {
                for (int j = 0; j < g[0].length; j++) {
                    Arrays.fill(g[i][j], 1);
                }
            }
            this.gamma = new Matrix(g);
            this.beta = new Matrix(b);

        }


    }
}
