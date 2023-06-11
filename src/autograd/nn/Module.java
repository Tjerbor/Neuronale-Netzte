package autograd.nn;

import autograd.Tensor;

import java.util.Collections;
import java.util.List;

public class Module {


    public void zero_grad() {

        for (Tensor p : this.getParameters()) {
            p.grad = 0;
        }
    }


    public List<Tensor> getParameters() {
        return Collections.EMPTY_LIST;
    }
}
