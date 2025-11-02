from operator import index

from Libraries import adjacency_matrix_graph
from Libraries import dijkstra

# Create graph
stations_graph = adjacency_matrix_graph.AdjacencyMatrixGraph(6, weighted=True)

stations_graph.insert_edge(0, 1, 5)  # A<=>B 5 minutes
stations_graph.insert_edge(0, 2, 8)  # A<=>B 8 minutes
stations_graph.insert_edge(1, 2, 10) # A<=>B 10 minutes
stations_graph.insert_edge(1, 3, 6)  # A<=>B 6 minutes
stations_graph.insert_edge(2, 3, 7)  # A<=>B 7 minutes
stations_graph.insert_edge(3, 4, 3)  # A<=>B 3 minutes
stations_graph.insert_edge(3, 5, 9)  # A<=>B 9 minutes
stations_graph.insert_edge(4, 5, 2)  # A<=>B 2 minutes


# Maps stations names to indexes
station_names = ["A", "B", "C", "D", "E", "F"] #A=0, B=1 ... F=5
station_to_index = {name: index for index, name in enumerate(station_names)}

starting_in = 0
duration, predecessors = dijkstra.dijkstra(stations_graph, starting_in)