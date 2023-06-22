package main;

abstract public class LayerNew {


    abstract public LayerNew getNextLayer();

    abstract public void setNextLayer(LayerNew l);

    abstract public LayerNew getPreviousLayer();

    abstract public void setPreviousLayer(LayerNew l);


}
