package Matrix;

import org.junit.jupiter.api.Test;
import utils.Array_utils;
import utils.Matrix;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TestIterator {

    @Test
    public void testGetItem() {

        Random r = new Random();

        int x = r.nextInt(3, 10);
        int y = r.nextInt(3, 10);

        double[][] d = new double[x][y];
        Array_utils.arange(d);

        Matrix m = new Matrix(d);

        System.out.println(Arrays.deepToString(d));
        System.out.println("x: " + x + " y: " + y);
        System.out.println(m.getItem(7));

        assert (double) m.getItem(7) == 7;


    }

    @Test
    public void testIterator() {

        Random r = new Random();


        int x = r.nextInt(3, 10);
        int y = r.nextInt(3, 10);

        double[][] d = new double[x][y];

        Array_utils.arange(d);

        Matrix m = new Matrix(d);

        Iterator iter = m.iterator();

        Object[] ist = new Object[x * y];
        int count = 0;
        while (iter.hasNext()) {
            ist[count] = iter.next();
            count += 1;
        }


        double[][] istS = new double[x][y];

        count = 0;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                istS[i][j] = (double) (Double) ist[count];
                count += 1;
            }
        }


        assertArrayEquals(d, istS);


    }

    @Test
    public void testGetItem3D() {

        Random r = new Random();

        //int x = r.nextInt(3, 10);
        //int y = r.nextInt(3, 10);
        //int z = r.nextInt(3, 10);

        int x = 2;
        int y = 3;
        int z = 4;

        double[][][] d = new double[x][y][z];

        Array_utils.arange(d);

        Matrix m = new Matrix(d);

        Iterator iter = m.iterator();

        Object[] ist = new Object[x * y * z];
        int count = 0;
        while (iter.hasNext()) {
            Object o = iter.next();
            ist[count] = o;
            count += 1;
        }


        System.out.println(Arrays.deepToString(ist));
        double[][][] istS = new double[x][y][z];

        count = 0;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < z; k++) {
                    istS[i][j][k] = (double) (Double) ist[count];
                    count += 1;
                }
            }
        }

        System.out.println(Arrays.deepToString(d));
        System.out.println(Arrays.deepToString(istS));

        assertArrayEquals(d, istS);

    }

    @Test
    public void testGetItem4D() {

        Random r = new Random();

        //int x = r.nextInt(3, 10);
        //int y = r.nextInt(3, 10);
        //int z = r.nextInt(3, 10);

        int x = 2;
        int y = 3;
        int z = 4;
        int z2 = 6;

        double[][][][] d = new double[x][y][z][z2];

        Array_utils.arange(d);

        Matrix m = new Matrix(d);

        Iterator iter = m.iterator();

        Object[] ist = new Object[x * y * z * z2];
        int count = 0;
        while (iter.hasNext()) {
            Object o = iter.next();
            ist[count] = o;
            count += 1;
        }


        System.out.println(Arrays.deepToString(ist));
        double[][][][] istS = new double[x][y][z][z2];

        count = 0;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < z; k++) {
                    for (int l = 0; l < z2; l++) {
                        istS[i][j][k][l] = (double) (Double) ist[count];
                        count += 1;
                    }

                }
            }
        }

        System.out.println(Arrays.deepToString(d));
        System.out.println(Arrays.deepToString(istS));

        assertArrayEquals(d, istS);

    }


}
