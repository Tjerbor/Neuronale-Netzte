package main;

import layer.Layer;

public interface Optimizer {

    default void pre_epoch() {
    }

    default void past_epoch() {
    }

    default void update(Layer l) {
    }


}
