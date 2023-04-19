package layers;

public class Reshape {

    int[] input_shape;
    int output_shape;

    public Reshape(int[] input_shape, int output_shape) {
        this.output_shape = output_shape;
        this.input_shape = input_shape;
    }


    public double[] forward(double[][] inputs) {
        double[] output = new double[this.output_shape];

        int count = 0;
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                output[count] = inputs[i][j];
                count += 1;
            }

        }
        return output;

    }


    public double[][] backward(double[] dinput) throws Exception {
        double[][] outputs = new double[input_shape[0]][input_shape[1]];
        int count = 0;
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                outputs[i][j] = dinput[count];
                count += 1;
            }

        }
        return outputs;
    }
}
