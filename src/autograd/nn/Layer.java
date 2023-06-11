package autograd.nn;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Layer extends Module {

    List<Neuron> neuronList;

    public Layer(int nin, int nout, boolean nonlin) {
        this.neuronList = new ArrayList<>();
        IntStream.range(0, nout).boxed().forEach(i -> this.neuronList.add(new Neuron(nin, nonlin)));
    }

    @Override
    public List<Tensor> getParameters() {
        List<Tensor> parameterList = new ArrayList<>();
        for (Neuron n : this.neuronList) {
            for (Tensor p : n.getParameters()) {
                parameterList.add(p);
            }
        }
        return parameterList;
    }


    public List<Tensor> forward(List<Tensor> x) {
        List<Tensor> outNeuronList = new ArrayList<>();
        for (Neuron n : this.neuronList) {
            outNeuronList.add(n.forward(x));
        }
//        return outNeuronList.size()==1?outNeuronList.get(0):outNeuronList;
        return outNeuronList;
    }

    @Override
    public String toString() {
        List<String> layerStringList = new ArrayList<>();
        for (Neuron n : this.neuronList) {
            layerStringList.add(n.toString());
        }
        return "Layer of [" + String.join(",", layerStringList) + "]";
    }

}
