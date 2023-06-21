package main;

import layer.Activation;
import layer.CustomActivation;
import layer.FullyConnectedLayer;
import layer.StepFunc;
import org.junit.jupiter.api.*;
import utils.Reader;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * This class contains various tests to assert the correctness of the neural network.
 *
 * @see <a href="https://junit.org/junit5/docs/current/user-guide/">JUnit Jupiter</a>
 */
class NeuralNetworkTest {
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
        neuralNetwork.setLayers(Reader.create("data/weights/logicalConjunction.csv"));

        neuralNetwork.setFunction(0, new StepFunc(1.5));

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
        neuralNetwork.setLayers(Reader.create("data/weights/trafficLight.csv"));

        neuralNetwork.setFunction(0, new CustomActivation(new String[]{"logi", "logi", "logi", "id"}));
        neuralNetwork.setFunction(1, new CustomActivation(new String[]{"id", "id", "id", "id"}));

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

            neuralNetwork.create(topology, new Activation());

            assertArrayEquals(topology, neuralNetwork.topology(), "The returned topology is not correct.");

            assertEquals(2, neuralNetwork.size(), "The size of the neural network is not correct.");
        }

        @Test
        @DisplayName("Create: Two Activation Functions")
        void createTwo() {
            int[] topology = {3, 3, 3, 3, 4, 4};

            neuralNetwork.create(topology, new Activation[]{new Activation(), new Activation()});

            assertArrayEquals(topology, neuralNetwork.topology(), "The returned topology is not correct.");

            assertEquals(5, neuralNetwork.size(), "The size of the neural network is not correct.");
        }

        @Test
        @DisplayName("Create: Multiple Activation Functions")
        void createMultiple() {
            int[] topology = {3, 3, 3, 3, 4, 4};

            neuralNetwork.create(topology, new Activation[]{new Activation(), new Activation(), new Activation(), new Activation(), new Activation()});

            assertArrayEquals(topology, neuralNetwork.topology(), "The returned topology is not correct.");

            assertEquals(5, neuralNetwork.size(), "The size of the neural network is not correct.");

            Activation[] activation = new Activation[]{new Activation()};

            assertThrows(IllegalArgumentException.class, () -> neuralNetwork.create(topology, activation));
        }

        @Test
        @DisplayName("Fully Connected Layer")
        void denseLayer() {
            FullyConnectedLayer layer = new FullyConnectedLayer(2, 1);

            double[][] weights = new double[][]{{1}, {1}, {0.5}};

            layer.setWeights(weights);

            layer.setActivation(new Activation());

            assertArrayEquals(weights, layer.getWeights(), "The returned weights are not correct.");

            double[][] inputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            double[][] result = new double[][]{{0.5}, {1.5}, {1.5}, {2.5}};

            assertAll(
                    "The forward pass did not return the correct values.",
                    () -> assertArrayEquals(result[0], layer.forward(inputs[0])),
                    () -> assertArrayEquals(result[1], layer.forward(inputs[1])),
                    () -> assertArrayEquals(result[2], layer.forward(inputs[2])),
                    () -> assertArrayEquals(result[3], layer.forward(inputs[3]))
            );

            assertArrayEquals(result, layer.forward(inputs), "The forward pass did not return the correct values.");

            assertThrows(IllegalArgumentException.class, () -> layer.setWeights(new double[][]{{1}}));
        }
    }
}
