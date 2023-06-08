package autograd.utils;

import autograd.Tensor;

public class testTensor {

    public static void main(String[] args) {


        Tensor x1 = new Tensor(2.0);
        Tensor w1 = new Tensor(-3.0);
        Tensor w2 = new Tensor(1.0);
        Tensor x2 = new Tensor(0.0);

        Tensor b = new Tensor(6.8813735870195432);

        Tensor n = x1.mult(w1).add(x2.mult(w2)).add(b);

        n.backward();

        System.out.print(n.grad);

        testTensor();
        matmulTest();

        mMTest();
    }


    public static void mMTest() {


    }

    public static void matmulTest() {


        Tensor[][] a = new Tensor[3][4];
        Tensor[][] b = new Tensor[4][3];
        Tensor[][] c = new Tensor[3][3];


        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                a[i][j] = new Tensor(i + j).pow(2);
                b[j][i] = new Tensor(j * i).pow(3);

            }
        }


        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                c[i][j] = new Tensor(0.0);
            }
        }


        for (int e = 0; e < a.length; e++) {
            for (int x = 0; x < b[1].length; x++) {
                for (int j = 0; j < a[0].length; j++) {
                    c[e][x] = c[e][x].add(a[e][j].mult(b[j][x]));

                }
            }
        }


        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                c[i][j].backward();
            }
        }

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                System.out.print(a[i][j].grad + ", ");
            }
        }
        System.out.println("\n");

        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                System.out.print(b[i][j].grad + ", ");
            }
        }


    }


    public static void testTensor() {
        Tensor a = new Tensor(-4.0);
        Tensor b = new Tensor(2.0);
        Tensor c = a.add(b);
        Tensor d = a.mult(b).add(b.pow(3));
        c = c.add(c.add(1));
        c = c.add(c.add(1).add(a.neg()));
        d = d.add(d.mult(2).add((b.add(a).relu())));
        d = d.add(d.mult(3).add((b.add(a.neg())).relu()));
        Tensor e = c.add(d.neg());
        Tensor f = e.pow(2);
        Tensor g = f.div(2.0);
        g = g.add(f.rdiv(10.0));

        System.out.println(f.data); // prints 24.7041, the outcome of this forward pass
        g.backward();
        System.out.println(a.grad); // prints 138.8338, i.e. the numerical value of dg/da
        System.out.println(b.grad); // prints 645.5773
        System.out.println(g.grad); // prints 645.5773
    }

}
