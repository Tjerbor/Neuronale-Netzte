package autograd.nn;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Neuron extends Module {


    List<Tensor> w;
    Tensor b;
    boolean nonlin;

    public Neuron(int nin) {
        this.nonlin = true;
        b = new Tensor(nin);
    }


    public Neuron(int nin, boolean nonlin) {
        this.w = new ArrayList<Tensor>();
        IntStream.range(0, nin).boxed().forEach(i -> this.w.add(new Tensor(-1 + (float) Math.random() * 2)));
        this.b = new Tensor(0);
        this.nonlin = nonlin;
    }


    public Tensor forward(List<Tensor> x) {

        List<Tensor> multListValue = new ArrayList<Tensor>();
        for (int index = 0; index < x.size(); index++) {
            multListValue.add(this.w.get(index).mult(x.get(index)));
        }
        Tensor act = null;
        act = multListValue.get(0);
        for (Tensor value : multListValue) {
            act = act.add(value);
        }
        act = act.add(this.b);
        if (this.nonlin) {
            return act.relu();
        } else {
            return act;
        }

    }

    @Override
    public List<Tensor> getParameters() {
        List<Tensor> parameterList = new ArrayList(this.w);
        parameterList.add(this.b);
        return parameterList;
    }

    
}
