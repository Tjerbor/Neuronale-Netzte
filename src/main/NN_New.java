package main;

public class NN_New {

    private LayerNew fristLayer;
    private LayerNew nextLayer;
    private int size;

    public compute() {


    }

    public int size() {
        return size;
    }

    public void add(LayerNew l) {

        if (this.fristLayer == null) {
            fristLayer = l;

        } else {
            LayerNew tmp = fristLayer;
            while (tmp.getNextLayer() != null) {
                tmp = tmp.getNextLayer();


            }
            LayerNew before = tmp;
            before.setNextLayer(l);
            l.setPreviousLayer(before);
        }
        size++;
    }


}
