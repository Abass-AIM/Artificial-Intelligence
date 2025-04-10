import csv
import pickle
import os
import codecs
import numpy as np
import matplotlib.pyplot as plt

class TravelingSalesmanProblem:
    def __init__(self, name):
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0
        self.__initData()

    def __len__(self):
        return self.tspSize

    def __initData(self):
        try:
            self.locations = pickle.load(open(os.path.join("tsp-data", self.name + "-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join("tsp-data", self.name + "-dist.pickle"), "rb"))
        except (OSError, IOError):
            self.__createData()

        self.tspSize = len(self.locations)

    def __createData(self):
        self.locations = []

        # Read from local file instead of URL
        with open(os.path.join("tsp-data", self.name + ".tsp"), "r") as f:
            reader = csv.reader(f, delimiter=" ", skipinitialspace=True)

            for row in reader:
                if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break

            for row in reader:
                if row[0] != 'EOF':
                    del row[0]
                    self.locations.append(np.asarray(row, dtype=np.float32))
                else:
                    break

            self.tspSize = len(self.locations)
            print("length = {}, locations = {}".format(self.tspSize, self.locations))

            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    distance = np.linalg.norm(self.locations[j] - self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, self.locations[i], self.locations[j], distance))

            if not os.path.exists("tsp-data"):
                os.makedirs("tsp-data")
            pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))

    def getTotalDistance(self, indices):
        distance = self.distances[indices[-1]][indices[0]]
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]
        return distance

    def plotData(self, indices):
        plt.scatter(*zip(*self.locations), marker='.', color='red')
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])
        plt.plot(*zip(*locs), linestyle='-', color='blue')
        return plt

def main():
    tsp = TravelingSalesmanProblem("bayg29")
    optimalSolution = [0, 27, 5, 11, 8, 25, 2, 28, 4, 20, 1, 19, 9, 3, 14, 17, 13, 16, 21, 10, 18, 24, 6, 22, 7, 26, 15, 12, 23]
    print("Problem name: " + tsp.name)
    print("Optimal solution = ", optimalSolution)
    print("Optimal distance = ", tsp.getTotalDistance(optimalSolution))
    plot = tsp.plotData(optimalSolution)
    plot.show()

if __name__ == "__main__":
    main()