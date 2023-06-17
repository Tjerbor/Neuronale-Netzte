def writeConv(fpath, weights, numFillter=8, k1=3
              , k2=3, stride1=1, stride2=1, inputShape=(28, 28, 1)):
    f = open(fpath, "w", encoding="utf-8")

    s = "conv2D;" + ";".join([str(numFillter), str(k1), str(k2),
                              str(stride1), str(stride2), str(inputShape[0]), str(inputShape[1]),
                              str(inputShape[2])]) + "\n"

    print(weights)
    print(weights.shape)
    n, m, o, p = weights.shape
    expectedSize = n * m * o * p
    count = 0
    for i in range(n):
        for j in range(m):
            for k in range(o):
                for l in range(p):
                    if (count != expectedSize - 1):
                        s += str(weights[i][j][k][l]) + ";"
                    else:
                        s += str(weights)

    f.write(s)
    f.close()
