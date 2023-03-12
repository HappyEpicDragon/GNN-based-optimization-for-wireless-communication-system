import matplotlib.pyplot as plt

filenames = ["pilot=3.txt",
             "pilot=6.txt",
             "pilot=9.txt",
             "pilot=15.txt",
             "pilot=30.txt",
             "pilot=60.txt",
             "pilot=120.txt",
             ]

labels = [
    "pilot_length = 3",
    "pilot_length = 6",
    "pilot_length = 9",
    "pilot_length = 15",
    "pilot_length = 30",
    "pilot_length = 60",
    "pilot_length = 120",
]


def draw_main():
    draw()


def fetch_data(filename):
    # numbers = []
    # for filename in filenames:
    #     with open(filename) as f:
    #         number = f.readlines()
    #         numbers.append(number)
    with open(filename) as f:
        number = f.readlines()
    return number


def draw():
    for i in range(len(filenames)):
        filename = "./GNN_data/" + filenames[i]
        number = fetch_data(filename)
        x = []
        for j in range(len(number)):
            x.append(j)
            number[j] = -float(number[j]) / 10000
        plt.plot(x, number, label=labels[i])
    # plt.draw()
    plt.xlabel("Iteration Epochs")
    plt.ylabel("SINR Rate")
    plt.legend()
    plt.show()
