package layer;

import utils.Array_utils;

import java.util.Arrays;

//https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
public class Conv2D {

    /**
     * def apply_filter_to_image(image: np.ndarray,
     * kernel: List[List[float]]) -> np.ndarray:
     * """Applies filter to the given image.
     * Args:
     * image (np.ndarray): 3D matrix to be convolved. Shape must be in HWC format.
     * kernel (List[List[float]]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
     * Returns:
     * np.ndarray: image after applying kernel.
     * """
     * kernel = np.asarray(kernel)
     * b = kernel.shape
     * return np.dstack([conv2d(image[:, :, z], kernel, padding=(b[0]//2,  b[1]//2))
     * for z in range(3)]).astype('uint8')
     **/

    int kernelSize = 3;
    Conv c = new Conv(32);

    public double[][][][] forward(double[][][][] inputs) {

        double[][][][] out = Array_utils.zerosLike(inputs);

        for (int i = 0; i < inputs.length; i++) {
            out[i] = fordward(inputs[i]);
        }


        return out;

    }

    public double[][] cutOffLastDim(double[][][] a, int d) {

        if (d > 3) {
            System.out.println("A Picture can not have more than " +
                    "3 Colour Channels");
        }

        System.out.println(d);
        System.out.println(Arrays.toString(
                Array_utils.getShape(a)));
        double[][] c = new double[a.length][a[0].length];
        System.out.println(Arrays.toString(
                Array_utils.getShape(c)));
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j][d];
            }
        }

        return c;
    }

    public double[][][] fordward(double[][][] input) {


        int count = 0;
        double[][][] out;
        double[][][] n = new double[input.length][input.length]
                [c.num_filters *
                input[0][0].length];
        for (int dim = 0; dim < input[0][0].length; dim++) {
            out = c.forward(cutOffLastDim(input, dim));

            System.out.println(Array_utils.getShape(out));
            System.out.println(Array_utils.getShape(n));
            for (int i = 0; i < c.num_filters; i++) {
                for (int j = 0; j < input.length; j++) {
                    for (int k = 0; k < input[0].length; k++) {
                        System.out.println(Array_utils.getShape(out));
                        System.out.println(Array_utils.getShape(n));
                        n[j][k][count] = out[j][k][i];
                    }
                }
                count += 1;
            }

        }


        return n;
    }


}
