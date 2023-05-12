package main;

/**
 * is not really implemented. Shall be in the future.
 * He calculates the updated Weights with either options such as decay and momentum
 * or uses vanilla gradient. which is just learning rate.
 */
public class SGD {
    double learning_rate;
    double current_learning_rate;
    double momentum;
    int iterations = 0;
    double decay = 0;

    public SGD(double learning_rate, double momentum, double decay) {
        this.learning_rate = learning_rate;
        this.current_learning_rate = learning_rate;
        this.momentum = momentum;
        this.decay = decay;
    }

    public SGD(double learning_rate, double momentum) {
        this.learning_rate = learning_rate;
        this.current_learning_rate = learning_rate;
        this.momentum = momentum;
    }

    public SGD(double learning_rate) {
        this.current_learning_rate = learning_rate;
    }

    // sets new learning_rate if decay is activ.
    public void pre_epoch() {
        if (this.decay != 0) {
            this.current_learning_rate = this.learning_rate * (1. / (1. + this.decay * this.iterations));
        }


    }

    public void past_epoch() {
        this.iterations += 1;
    }
}

