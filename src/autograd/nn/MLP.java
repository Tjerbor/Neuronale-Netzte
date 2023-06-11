package autograd.nn;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class MLP extends Module {

    List<Layer> layerList;

    public MLP(int nin, List<Integer> nouts) {
        layerList = new ArrayList();
        List<Integer> sz = new ArrayList();
        sz.add(nin);
        sz.addAll(nouts);
        IntStream.range(0, nouts.size()).boxed().forEach(i -> this.layerList.add(
                new Layer(sz.get(i), sz.get(i + 1), i != (nouts.size() - 1))
        ));
    }

    public List<Tensor> forward(List<Tensor> x) {
        for (Layer layer : this.layerList) {
            x = layer.forward(x);
        }
        return x;
    }

    @Override
    public List<Tensor> getParameters() {
        List<Tensor> parameterList = new ArrayList();
        for (Layer layer : this.layerList) {
            for (Tensor p : layer.getParameters()) {
                parameterList.add(p);
            }
        }
        return parameterList;
    }

    @Override
    public String toString() {
        List<String> layerStringList = new ArrayList();
        for (Layer layer : this.layerList) {
            layerStringList.add(layer.toString());
        }
        return "MLP of [" + String.join(",", layerStringList) + "]";
    }
}
