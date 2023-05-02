import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import utils.Reader;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * This class contains various tests to assert the correctness of the neural network.
 *
 * @see <a href="https://junit.org/junit5/docs/current/user-guide/">JUnit Jupiter</a>
 */
public class NeuralNetworkTest {
    private NeuralNetwork neuralNetwork;

    @BeforeEach
    void initialize() {
        neuralNetwork = new NeuralNetwork();
    }

    @Test
    void logicalConjunction() throws Exception {
        neuralNetwork.create(Reader.create("csv/topology/logicalConjunction.csv"));

        // TODO: Set Activation Function

        double[][] result = neuralNetwork.computeAll(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});

        assertAll(
                () -> assertEquals(0, result[0][0]),
                () -> assertEquals(0, result[1][0]),
                () -> assertEquals(0, result[2][0]),
                () -> assertEquals(1, result[3][0])
        );
    }

    @Test
    void trafficLight() throws IOException {
        neuralNetwork.create(Reader.create("csv/topology/trafficLight.csv"));

        double[] result = neuralNetwork.compute(new double[]{1, 0, 0});

        assertAll(
                () -> assertEquals(-0.03504739174185117, result[0]),
                () -> assertEquals(0.06379840926271282, result[1]),
                () -> assertEquals(0.11976621345436994, result[2]),
                () -> assertEquals(0.032371141308330784, result[3])
        );
    }
}
