input_source = None
import heapq

def dijkstra(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    pq = [(0, start_node)]
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        if curr_dist > distances[curr_node]:
            continue
        for neighbor, weight in graph[curr_node].items():
            dist = curr_dist + weight
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(pq, (dist, neighbor))
    return distances
python_dijkstra_shortest_path = dijkstra
python_graph_build_cell = python_dijkstra_shortest_path
import heapq

def astar(graph, start, goal, h):
    open_set = [(h(start, goal), 0, start, [start])]
    visited = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in visited:
                heapq.heappush(open_set, (g + weight + h(neighbor, goal), g + weight, neighbor, path + [neighbor]))
    return []
python_astar_pathfinding = astar(python_graph_build_cell, start_node, goal_node, lambda a, b: 0)