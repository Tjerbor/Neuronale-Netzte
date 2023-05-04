import layers.CustomActivation;
import layers.StepFunc;
import org.junit.jupiter.api.*;
import utils.Reader;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

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

    @AfterEach
    void tearDown() {
        neuralNetwork = null;
    }

    @Test
    void logicalConjunction() throws Exception {
        neuralNetwork.setLayers(Reader.create("csv/topology/logicalConjunction.csv"));

        neuralNetwork.setFunction(1, new StepFunc(1.5));

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
        neuralNetwork.setLayers(Reader.create("csv/topology/trafficLight.csv"));

        neuralNetwork.setFunction(1, new CustomActivation(new String[]{"logi", "logi", "logi", "id"}));

        double[] result = neuralNetwork.compute(new double[]{1, 0, 0});

        assertAll(
                () -> assertEquals(-0.03504739174185117, result[0]),
                () -> assertEquals(0.06379840926271282, result[1]),
                () -> assertEquals(0.11976621345436994, result[2]),
                () -> assertEquals(0.032371141308330784, result[3])
        );
    }

    @Nested
    class UnitTest {
        @Test
        @DisplayName("Create: One Activation Function")
        void createOne() {
            int[] topology = {3, 3, 4};

            neuralNetwork.create(topology, "id");

            assertArrayEquals(topology, neuralNetwork.getTopology(), "The returned topology is not correct.");

            assertEquals(4, neuralNetwork.size(), "The size of the neural network is not correct.");
        }

        @Test
        @DisplayName("Create: Two Activation Functions")
        void createTwo() {
            int[] topology = {3, 3, 3, 3, 4, 4};

            neuralNetwork.create(topology, new String[]{"id", "id"});

            assertArrayEquals(topology, neuralNetwork.getTopology(), "The returned topology is not correct.");

            assertEquals(10, neuralNetwork.size(), "The size of the neural network is not correct.");
        }

        @Test
        @DisplayName("Create: Multiple Activation Functions")
        void createMultiple() {
            int[] topology = {3, 3, 3, 3, 4, 4};

            neuralNetwork.create(topology, new String[]{"id", "id", "id", "id", "id"});

            assertArrayEquals(topology, neuralNetwork.getTopology(), "The returned topology is not correct.");

            assertEquals(10, neuralNetwork.size(), "The size of the neural network is not correct.");

            assertThrows(IllegalArgumentException.class, () -> neuralNetwork.create(topology, new String[]{"id"}));
        }
    }
}
