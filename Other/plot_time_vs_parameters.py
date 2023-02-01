import matplotlib.pyplot as plt

y1 = [1.3, 2, 3.3, 5.3, 8, 13, 213]
y2 = [0.16, 0.25, 0.42, 0.66, 1, 1.63, 26.63]

# x = [125_000_000,   350_000_000,    1_300_000_000,  2_700_000_000,  6_700_000_000,     13_000_000_000,  30_000_000_000,     30_000_000_000]
x = [0.125, 0.350, 1.3, 2.7, 6.7, 13, 30]

# plot
plt.plot(x, y1, linewidth=2.0, marker="o", label="Using 1 GPU")
plt.plot(x, y2, linewidth=2.0, marker="o", label="Using 8 GPUs")

plt.yscale("log")
plt.title("Time to generate the dataset\n(Number of parameters vs. time)")
plt.xlabel("Number of parameters (in billions)")
plt.ylabel("Number of days (log scale)")
plt.legend()

plt.savefig("./plot_time_vs_parameters.png")
