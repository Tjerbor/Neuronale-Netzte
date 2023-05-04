import layers.*;
import org.junit.jupiter.api.*;
import utils.Reader;
import utils.Utils;

import java.io.IOException;
import java.util.Arrays;

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

    @Test
    public void train_batch_new() throws Exception {

        double loss_per_epoch;

        double[][] y_train2 = Mnist_reader.getTrainData_y("src/utils/mnist_data_full.txt");
        double[][] x_train2 = Mnist_reader.getTrainData_x("src/utils/mnist_data_full.txt");

        double[][][] x_train = Mnist_reader.x_train_2_batch(x_train2, 4);
        double[][][] y_train = Mnist_reader.y_train_2_batch(y_train2, 4);

        int step_size = x_train.length;

        double[] step_losses = new double[step_size];

        FullyConnectedLayer[] Ebenen = new FullyConnectedLayer[2];
        Ebenen[0] = new FullyConnectedLayer(784, 49);
        Ebenen[1] = new FullyConnectedLayer(49, 10);

        Activation[] acts = new Activation[2];

        acts[0] = new Tanh();
        acts[1] = new Tanh();

        Losses loss = new MSE();
        int ownSize = Ebenen.length;

        int epoch = 30;
        double[][] outs;
        for (int e = 0; e < epoch; e++) {
            for (int j = 0; j < step_size; j++) {
                outs = x_train[j];
                ;
                for (int k = 0; k < Ebenen.length; k++) {
                    outs = Ebenen[k].forward(outs);
                    outs = acts[k].forward(outs);
                }

                step_losses[j] = loss.forward(outs, y_train[j]);
                //calculates prime Loss
                double[][] grad = loss.backward(outs, y_train[j]);
                // now does back propagation //updates values.
                for (int i = 0; i < Ebenen.length; i++) {
                    if (i == 0) {
                        grad = acts[Ebenen.length - 1 - i].backward(grad, 0.1);
                    }

                    grad = Ebenen[Ebenen.length - 1 - i].backward(grad, 0.1);
                }


            }

            loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }

    }

    @Test
    public void train_single_test() throws Exception {

        double loss_per_epoch;

        double[][] y_train = Mnist_reader.getTrainData_y("src/utils/mnist_data_full.txt");
        double[][] x_train = Mnist_reader.getTrainData_x("src/utils/mnist_data_full.txt");

        int step_size = x_train.length;

        double[] step_losses = new double[step_size];

        FullyConnectedLayer[] Ebenen = new FullyConnectedLayer[2];
        Ebenen[0] = new FullyConnectedLayer(784, 49);
        Ebenen[1] = new FullyConnectedLayer(49, 10);

        Activation[] acts = new Activation[2];

        acts[0] = new Tanh();
        acts[1] = new Tanh();

        Losses loss = new MSE();
        int ownSize = Ebenen.length;


        int epoch = 30;
        double[] outs;
        for (int e = 0; e < epoch; e++) {
            for (int j = 0; j < step_size; j++) {
                outs = x_train[j];
                ;
                for (int k = 0; k < Ebenen.length; k++) {
                    outs = Ebenen[k].forward(outs);
                    outs = acts[k].forward(outs);
                }

                step_losses[j] = loss.forward(outs, y_train[j]);
                //calculates prime Loss
                double[] grad = loss.backward(outs, y_train[j]);
                // now does back propagation //updates values.
                for (int i = 0; i < Ebenen.length; i++) {
                    if (i == 0) {
                        grad = acts[Ebenen.length - 1 - i].backward(grad, 0.1);
                    }

                    grad = Ebenen[Ebenen.length - 1 - i].backward(grad, 0.1);
                }


            }

            loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }

    }

    @Test
    public void test_csv() throws Exception {

        int epoch = 30;
        double[] outs;

        double[][] x_train = Reader.getTrainDataInputs("csv/trainData/KW16_traindata_trafficlights_classification(1).csv", 3);
        double[][] y_train = Reader.getTrainDataOutputs("csv/trainData/KW16_traindata_trafficlights_classification(1).csv", 4);

        int step_size = x_train.length;
        double[] step_losses = new double[step_size];
        double loss_per_epoch;

        FullyConnectedLayer[] Ebenen = new FullyConnectedLayer[2];
        Ebenen[0] = new FullyConnectedLayer(3, 3);
        Ebenen[1] = new FullyConnectedLayer(3, 4);

        Activation[] acts = new Activation[2];

        acts[0] = new Tanh();
        acts[1] = new Tanh();

        Losses loss = new MSE();

        for (int e = 0; e < epoch; e++) {
            for (int j = 0; j < step_size; j++) {
                outs = x_train[j];
                ;
                for (int k = 0; k < Ebenen.length; k++) {
                    outs = Ebenen[k].forward(outs);
                    outs = acts[k].forward(outs);
                }

                step_losses[j] = loss.forward(outs, y_train[j]);
                //calculates prime Loss
                double[] grad = loss.backward(outs, y_train[j]);
                // now does back propagation //updates values.
                for (int i = 0; i < Ebenen.length; i++) {
                    if (i == 0) {
                        grad = acts[Ebenen.length - 1 - i].backward(grad, 0.1);
                    }

                    grad = Ebenen[Ebenen.length - 1 - i].backward(grad, 0.1);
                }


            }

            loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }


        System.out.println(Arrays.deepToString(Ebenen[0].getWeights()));
        System.out.println(Arrays.deepToString(Ebenen[1].getWeights()));
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
