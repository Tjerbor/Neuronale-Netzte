import layers.MSE;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");


        Model model = new Model();

        int[] topologie = {784, 20, 10};
        model.create(topologie, "tahn");


        //System.out.println(Arrays.toString(c));

        /*
        System.out.println(model.structur.length);
        for(int i=0; i < model.structur.length ;i++){
            System.out.println(model.structur[i].name);

        }

        */
        //System.out.println(model.toString()); is slow.

        System.out.println(model.parameter_size);

        try {


            String fpath = "src/utils/mnist_data_full.txt";
            double[][] y_train = Mnist_reader.getTrainData_y(fpath);
            double[][] x_train = Mnist_reader.getTrainData_x(fpath);


            //System.out.println(Arrays.toString(x_train[0]));
            //System.out.println(Arrays.toString(x_train[1]));


            double[][][] x_train2 = Mnist_reader.x_train_2_batch(x_train, 4);
            double[][][] y_train2 = Mnist_reader.x_train_2_batch(y_train, 4);


            model.loss = new MSE();
            //model.train_single(30, x_train, y_train, 0.01);
            //model.train_with_batch(30, x_train2, y_train2, 0.05);
            model.train_batch_new(30, x_train2, y_train2, 0.5);
            //model.test_with_batch(x_train2, y_train2);
        } catch (Exception e) {
            System.out.println(e);
            throw new RuntimeException(e);
        }


    }
}