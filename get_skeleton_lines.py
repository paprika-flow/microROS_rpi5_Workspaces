from shapely.geometry import Polygon, LineString, Point
from sklearn.base import defaultdict

def remove_junctions_in_bucket_list(dict_ridge_points, vertex):
    if vertex in dict_ridge_points:
        del dict_ridge_points[vertex]
    else:
        return None
    for v, neighbors in dict_ridge_points.items():
        if vertex in neighbors:
            neighbors.remove(vertex)
    return 0


def get_skeleton_lines(vor, polygon=None):
    # Initialize vertex degree dictionary
    dict_ridge_points = {i: list() for i in range(len(vor.vertices))}
    ridge_lines = []

    # Map each vertex index to the edge it belongs to
    vertex_to_edge = {}

  # First pass: count vertex degrees
    for rv in vor.ridge_vertices:
        v0, v1 = vor.vertices[rv[0]], vor.vertices[rv[1]]
        line = LineString([v0, v1])
        if not polygon.covers(line):
            continue
        if -1 in rv:
            continue
        dict_ridge_points[rv[0]].append(rv[1])
        dict_ridge_points[rv[1]].append(rv[0])

  # Second pass: merge edges efficiently
    for rv in vor.ridge_vertices:
        v0, v1 = vor.vertices[rv[0]], vor.vertices[rv[1]]
        line = LineString([v0, v1])
        if not polygon.covers(line):
            continue
        if -1 in rv:
            continue
        v0, v1 = rv

        edge0 = vertex_to_edge.get(v0)
        edge1 = vertex_to_edge.get(v1)

        if edge0 is not None and len(dict_ridge_points[v0]) < 3:
            # Extend edge0 with v1  n
            if edge0[0] == v0:
                edge0[0] = v1
            else:
                edge0[1] = v1
            vertex_to_edge[v1] = edge0

        elif edge1 is not None and len(dict_ridge_points[v1]) < 3:
            # Extend edge1 with v0
            if edge1[0] == v1:
                edge1[0] = v0
            else:
                edge1[1] = v0
            vertex_to_edge[v0] = edge1

        else:
            # Create new edge
            new_edge = [v0, v1]
            ridge_lines.append(new_edge)
            vertex_to_edge[v0] = new_edge
            vertex_to_edge[v1] = new_edge

    dict_ridge_points = {i: list() for i in range(len(vor.vertices))}
    adj = defaultdict(list)
    for a, b in ridge_lines:
        adj[a].append(b)
        adj[b].append(a)     
    junctions = {v for v in adj if len(adj[v]) != 2}
    # print(f"Junction vertices: {junctions}")
    skeleton_lines = []
    for a in ridge_lines:
        # print(f"Processing edge: {a} with neighbors {adj[a[0]]} and {adj[a[1]]}")
        if a[0] in junctions and a[1] in junctions:
            pass
        elif a[0] in junctions:
            curr = a[1]
            prev = a[0]
            while curr not in junctions:
                neighbors = adj[curr]
                next_v = neighbors[0] if neighbors[0] != prev else neighbors[1]
                prev, curr = curr, next_v
            a[1] = curr
            
        elif a[1] in junctions:
            curr = a[0]
            prev = a[1]
            while curr not in junctions:
                neighbors = adj[curr]
                next_v = neighbors[0] if neighbors[0] != prev else neighbors[1]
                prev, curr = curr, next_v
            a[0] = curr
        else:
            continue  # both ends are not junctions; skip
        if a[0] not in dict_ridge_points[a[1]] and a[1] not in dict_ridge_points[a[0]]:
            dict_ridge_points[a[1]].append(a[0])
            dict_ridge_points[a[0]].append(a[1])
        if a not in skeleton_lines and a[::-1] not in skeleton_lines:
            skeleton_lines.append(tuple(a))
    #deleting all duplicate edges
    return skeleton_lines, dict_ridge_points



