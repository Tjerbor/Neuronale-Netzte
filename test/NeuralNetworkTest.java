import layers.CustomActivation;
import layers.Layer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import utils.Reader;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @see <a href="https://junit.org/junit5/docs/current/user-guide/">JUnit Jupiter</a>
 */
public class NeuralNetworkTest {
    private NeuralNetwork neuralNetwork;

    @BeforeEach
    void initialize() {
        neuralNetwork = new NeuralNetwork();
    }

    @Test
    void trafficLight() throws IOException {
        Layer[] l2 = Reader.create("csv/topology/trafficLight.csv");


        neuralNetwork.create(Reader.create("csv/topology/trafficLight.csv"));

        Layer[] l = neuralNetwork.getModel();
        Layer cF = new CustomActivation(new String[]{"logi", "logi", "logi", "id"});

        for (Layer l1 : l) {
            System.out.println(l1.name);
            System.out.println(Arrays.deepToString(l1.weights));
        }

        //l[0].biases = new double[]{1, 1, 1, 1};
        //l[2].biases = new double[]{1, 1, 1};

        l[1] = cF;
        //neuralNetwork.create(l);


        //double[] result = new double[]{1, 0, 0};


        double[] result = neuralNetwork.compute(new double[]{1, 0, 0});

        assertEquals(-0.03504739174185117, result[0]);
        assertEquals(0.06379840926271282, result[1]);
        assertEquals(0.11976621345436994, result[2]);
        assertEquals(0.032371141308330784, result[3]);
    }
}
