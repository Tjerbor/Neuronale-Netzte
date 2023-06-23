package autograd.utils;

import autograd.Tensor;

public class TensorLoss {


    public Tensor MSE_loss(Tensor ist, double soll) {
        //Math.pow(actual[i] - expected[i], 2)
        return (ist.sub(soll)).pow(2);

    }

    public Tensor[] MSE_loss(Tensor[] ist, double[] soll) {
        //Math.pow(actual[i] - expected[i], 2)

        Tensor[] out = new Tensor[ist.length];
        for (int i = 0; i < ist.length; i++) {
            out[i] = MSE_loss(ist[i], soll[i]);
        }
        return out;

    }


}
