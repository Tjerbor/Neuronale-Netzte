package autograd;


/**
 * Based on the Code and Video from
 * Andrej Karpathy
 * https://github.com/karpathy/micrograd
 * https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb
 */


import java.util.*;

public class Tensor {
    public double data;
    public double grad;
    Backward backward = null;
    Set<Tensor> _prev;
    String _op;

    boolean isScalar = true;

    public Tensor(double data) {
        this.data = data;
        this._prev = Collections.EMPTY_SET;
        this._op = "";
    }

    public Tensor(double data, Set<Tensor> _children) {
        this.data = data;
        this._prev = _children;
        this._op = "";
    }

    public Tensor(double data, Set<Tensor> _children, String _op) {
        this.data = data;
        this._prev = _children;
        this._op = _op;
    }


    public Tensor add(Tensor otherValue) {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        _children.add(otherValue);
        Tensor outValue = new Tensor(this.data +
                otherValue.data, _children, "+");

        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                grad += outValue.grad;
                otherValue.grad += outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    public Tensor sub(Tensor otherValue) {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        _children.add(otherValue);
        Tensor outValue = new Tensor(this.data -
                otherValue.data, _children, "-");
        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                grad -= outValue.grad;
                otherValue.grad -= outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    public Tensor neg() {
        return this.mult(new Tensor(-1));
    }

    public Tensor add(double other) {
        return this.add(new Tensor(other));
    }

    public Tensor sub(double other) {
        other = other * (-1);
        return this.add(new Tensor(other));
    }

    public Tensor div(double other) {
        return this.mult(new Tensor(other).pow(-1));
    }

    public Tensor div(Tensor other) {
        return this.mult((other).pow(-1));
    }

    public Tensor rdiv(double other) {
        return new Tensor(other).mult(this.pow(-1));
    }

    public Tensor mult(double other) {
        return this.mult(new Tensor(other));
    }

    public Tensor mult(Tensor otherValue) {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        _children.add(otherValue);
        Tensor outValue = new Tensor(this.data * otherValue.data, _children, "*");
        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                grad += otherValue.data * outValue.grad;
                otherValue.grad += data * outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    /**
     * @param otherValue
     * @return
     */


    public Tensor pow(double otherValue) {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        Tensor outValue = new Tensor(Math.pow(this.data, otherValue), _children, "**");
        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                grad += (otherValue * Math.pow(data, (otherValue - 1))) * outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    public Tensor exp() {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        Tensor out = new Tensor(Math.exp(this.data), _children, "exp");


        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                grad += out.data * out.grad;
            }
        }
        out.backward = new Backward();
        return out;
    }

    public Tensor tanh() {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        double t = Math.tanh(this.data);
        Tensor out = new Tensor(t, _children, "Tanh");

        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                grad += (1 - Math.pow(t, 2)) * out.grad;
            }
        }
        out.backward = new Backward();
        return out;

    }

    public Tensor relu() {
        Set<Tensor> _children = new HashSet<>();
        _children.add(this);
        Tensor outValue;
        if (this.data < 0) {
            outValue = new Tensor(0, _children, "ReLu");
        } else {
            outValue = new Tensor(this.data, _children, "ReLu");
        }
        class Backward implements autograd.Backward {
            @Override
            public void _backward() {
                if (outValue.data > 0) {
                    grad += 1 * outValue.grad;
                } else {
                    grad += 0 * outValue.grad;
                }
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }


    /**
     * topological Search for the nodes.
     *
     * @param value        Tensor to search other noeds
     * @param topologies   List of the end topologie for this node.
     * @param visitedNodes
     */
    public void buildTopo(Tensor value, List<Tensor> topologies, Set<Tensor> visitedNodes) {
        if (!visitedNodes.contains(value)) {
            visitedNodes.add(value);
            for (Tensor child : value._prev) {
                buildTopo(child, topologies, visitedNodes);
            }
            topologies.add(value);
        }
    }

    public void backward() {
        // topological order all of the children in the graph
        List<Tensor> topologies = new ArrayList<Tensor>();
        Set<Tensor> visitedNodes = new HashSet<Tensor>();
        buildTopo(this, topologies, visitedNodes);
        // go one variable at a time and apple the chain rule to get its gradient
        this.grad = 1;
        Collections.reverse(topologies);
        for (Tensor value : topologies) {
            if (value.backward != null) {
                value.backward._backward();
            }
        }
    }


    public String toString() {
        return "Tensor(data=" + this.data + ", " + "grad=" + grad + ")";

    }

    public String toString(boolean b) {
        if (b) {
            return "data=" + this.data + ", " + "grad=" + grad;
        } else {
            return "data=" + data;
        }


    }

}
