package layers;

public class Softmax extends Activation{

    double exp = 0;

    public Softmax(){
        this.name = "softmax";
    }

    public double softmax(double x, double sum){
        return Math.exp(x) / sum;

    }
    public double softmax_prime(double x){
        return 0.0;
    }

    public double[][] forward(double[][] inputs){

        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            for (int j=0;j <inputs[0].length;j++){
                exp += this.softmax(inputs[j][i], 1);

            }

        }
        for (int i = 0; i < inputs.length; i++) {
            for (int j=0;j <inputs[0].length;j++){
                inputs[j][i] += this.softmax(inputs[j][i], exp);
            }

        }

        return inputs;

    }
}
