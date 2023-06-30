import matplotlib.pyplot as plt

lossList1 = [1, 0.032, 0.0170, 0.017, 0.0090, 0.008]
accList1 = [0.03, 0.8027, 0.8465, 0.87, 0.8903, 0.8953]
lossList2 = [1, 0.03, 0.0255, 0.018, 0.018, 0.016]
accList2 = [0.03, 0.75, 0.77, 0.8043, 0.8212, 0.8335]
lossList3 = [1, 0.03, 0.0255, 0.018, 0.018, 0.018]
accList3 = [0.0249, 0.7290, 0.755, 0.76, 0.77, 0.77]

lr = 0.4
legend = ["{784, 625, 125,104,26}", "{784, 625,100, 26}, ", "{784,70,40,26}", "loss", "loss", "loss"]

plt.title("Emnist Test-Buchstaben Datensatz Genauigkeit")
plt.xlabel("x Anzahl der Epochen")
plt.ylabel("Wahrscheinlichkeit")
plt.plot(accList1, "green")
plt.plot(accList2, "blue")
plt.plot(accList3, "red")
# plt.plot(lossList2, "blue")
# plt.plot(lossList1, "green")
# plt.plot(lossList3, "red")

plt.legend(legend)

plt.show()
