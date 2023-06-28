package export;

import builder.BuildNetwork;
import layer.*;
import main.NeuralNetwork;
import org.junit.jupiter.api.Test;
import utils.Matrix;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;


public class TestExportAndLoad {


    @Test
    public void testExportAndLoad_CompleteModel() throws IOException {
        int numFilter = 8;
        int numClasses = 10;
        int[] inputShape = new int[]{28, 28, 1};
        int kernelSize = 5;
        int strides = 2; //stepSize.
        BuildNetwork builder = new BuildNetwork(inputShape);
        builder.addConv2D_Last(numFilter, kernelSize, strides);
        builder.addDropout(0.5);
        builder.addMaxPooling2D_Last(); //uses for standard strides2 and poolSize2
        builder.addBatchNorm();
        builder.addFlatten();
        builder.addFastLayer(numClasses);

        NeuralNetwork nn = builder.getModel();
        nn.writeModel("model.txt");
        NeuralNetwork nn2 = load.LoadModel.loadModel("model.txt");

        assert nn.isEqual(nn2);

    }

    @Test
    public void testExportAndLoad() {


        BatchNorm b = new BatchNorm(8);


        Matrix m = b.getWeights();

        String s = b.export();

        String[] strings = s.split("\n");
        assert strings.length == 2;
        BatchNorm b2 = (BatchNorm) load.LoadModel.loadBatchNorm(strings[0].split(";"), strings[1]);

        assert s.equals(b2.export());


    }

    @Test
    public void test_FastLinearLayer_ExportAndLoad() {


        FastLinearLayer b = new FastLinearLayer(4, 8);

        String s = b.export();

        String[] strings = s.split("\n");
        assert strings.length == 2;
        FastLinearLayer b2 = (FastLinearLayer) load.LoadModel.loadFastLinearLayer(strings[0].split(";"), strings[1]);

        assert s.equals(b2.export());


    }

    @Test
    public void test_DropoutLayer_ExportAndLoad() {


        DropoutLayer b = new DropoutLayer(0.4);

        String s = b.export();

        String[] strings = s.split("\n");
        assert strings.length == 1;
        DropoutLayer b2 = (DropoutLayer) load.LoadModel.loadDropout(strings[0].split(";"));

        assert s.equals(b2.export());


    }

    @Test
    public void test_FlattenLayer_ExportAndLoad() {


        Flatten b = new Flatten(new int[]{28, 28, 8});

        String s = b.export();

        String[] strings = s.split("\n");
        assert strings.length == 1;
        Flatten b2 = (Flatten) load.LoadModel.loadFlatten(strings[0].split(";"));

        assert s.equals(b2.export());


    }

    @Test
    public void test_MaxPooling2D_Last_ExportAndLoad() {


        MaxPooling2D_Last b = new MaxPooling2D_Last(new int[]{28, 28, 8});

        String s = b.export();

        String[] strings = s.split("\n");
        assert strings.length == 1;
        MaxPooling2D_Last b2 = (MaxPooling2D_Last) load.LoadModel.loadMaxPooling2D_Last(strings[0].split(";"));

        assert s.equals(b2.export());


    }

    @Test
    public void test_Conv2D_ExportAndLoad() {


        Conv2D b = new Conv2D(new int[]{8, 28, 28}, 8);

        String s = b.export();
        System.out.println("now: " + s);

        String[] strings = s.split("\n");
        assert strings.length == 2;
        Conv2D b2 = (Conv2D) load.LoadModel.loadConv2D(strings[0].split(";"), strings[1]);


        System.out.println(b2.export());
        assert s.equals(b2.export());

        b = new Conv2D(new int[]{8, 28, 28}, 8);

        b.activateBias();
        s = b.export();

        strings = s.split("\n");
        assert strings.length == 3;
        b2 = (Conv2D) load.LoadModel.loadConv2D(strings[0].split(";"), strings[1], strings[2]);

        assert s.equals(b2.export());


    }

    @Test
    public void test_Conv2D_Last_ExportAndLoad() {


        Conv2D_Last b = new Conv2D_Last(new int[]{28, 28, 3}, 8, new int[]{5, 3}, new int[]{1, 1});

        String s = b.export();

        String[] strings = s.split("\n");
        assert strings.length == 2;
        Conv2D_Last b2 = (Conv2D_Last) load.LoadModel.loadConv2D_Last(strings[0].split(";"), strings[1]);


        assert s.equals(b2.export());

        b = new Conv2D_Last(new int[]{28, 28, 1}, 8);

        b.activateBias();
        s = b.export();


        strings = s.split("\n");
        assert strings.length == 3;
        b2 = (Conv2D_Last) load.LoadModel.loadConv2D_Last(strings[0].split(";"), strings[1], strings[2]);


        double[][][][] w = b.getWeights().getData4D();
        double[][][][] w2 = b2.getWeights().getData4D();
        assertArrayEquals(w, w2);
        assert s.equals(b2.export());


    }


}
