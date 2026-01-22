import heapq
def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0 # initial condition
    shortest_path = {node: None for node in graph}

    priority_queue_heap = [(0, start)]

    while len(priority_queue_heap) > 0:
        current_distance, current_node = heapq.heappop(priority_queue_heap)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            neighbor_distance = current_distance + weight

            if neighbor_distance < distances[neighbor]:
                distances[neighbor] = neighbor_distance
                shortest_path[neighbor] = current_node
                heapq.heappush(priority_queue_heap, (neighbor_distance, neighbor))

        # distance to end node found
    path = []
    current_node = end
    for node in graph:
        path.append(node)
        current_node = shortest_path[current_node]

    path.reverse()
    return path, distances[end]

