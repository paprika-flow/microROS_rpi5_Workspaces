
import math
import numpy as np

def get_midpoint(segment):
    p1, p2 = segment
    x1, y1 = p1
    x2, y2 = p2
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def find_better_edge(vor, v0, v1, dict_ridge_points, ridge_points_v0, path_lines = [], side_vectors = []):
    v1_x, v1_y = vor.vertices[v1]
    is_only_junctions = True
    for rv in ridge_points_v0:
        if rv == v1: # it doesn't make sense to have a path from v0 to v1 through v1
            continue
        if len(dict_ridge_points[rv]) <= 2: # if rv is not a junction, then it's not a valid path
            
            is_only_junctions = False
            # finding vertex that is connected to v0 that has less than 2 ridge points, which means it is a junction
            
            rv2 = rv
            for rrv in ridge_points_v0: 
                if  len(dict_ridge_points[rrv]) >= 3:
                    rv2 = rrv
                    break
            # if rv is greater in y than v1, then the path goes from rv to rv2, otherwise it goes from rv to v1
            if vor.vertices[rv][1] > v1_y and (rv2, rv) not in side_vectors and (rv, rv2) not in side_vectors:
                line = (rv2, rv)
                if line in path_lines or (line[1], line[0]) in path_lines:
                    break  # Skip side vectors
                return line
                
            else:
                line = (rv2, v1)
                if line in path_lines or (line[1], line[0]) in path_lines:
                    break  # Skip side vectors
                return line 
    if is_only_junctions:    
        return (v0, v1)  # No better edge found, return original line
    return None  # No valid path found

def get_angle(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

def get_vectors_area(line, lines, dict_ridge_points, vor, side_vectors, right):
    v0, v1 = line
    v0_x, v0_y = vor.vertices[v0]
    v1_x, v1_y = vor.vertices[v1]
    
    ridge_points_v0 = dict_ridge_points.get(v0, [])
    ridge_points_v1 = dict_ridge_points.get(v1, [])

    if v0_x < v1_x:
        if v1_x - 350 > v0_x:
            # print(f"No triangle found for line {line} (too far apart)")
            return 0, side_vectors
    else:
        if v0_x - 350 > v1_x:
            # print(f"No triangle found for line {line} (too far apart)")
            return 0, side_vectors
    triangle_line = tuple()
    # print(f"Finding triangle line for {line} with ridge points {ridge_points_v0} and {ridge_points_v1}")
    if (len(ridge_points_v0)) >= 3:
        # print(f"V0 has {len(ridge_points_v1)} ridge points")
        is_triangle_line_found = False
        for rv in ridge_points_v0:
            rv_x, rv_y = vor.vertices[rv]
            # print(f"Checking ridge point {rv} at ({rv_x:.0f}, {rv_y:.0f}) for line {line}")
            if rv == v1:
                continue
            if right and rv_x < v0_x:
                continue
            if not right and rv_x > v0_x:
                continue
            ridge_points_rv = dict_ridge_points.get(rv, [])
            is_triangle_line_branching = False
            if len(ridge_points_rv) >= 3:
                most_distant_rv = None
                max_angle = 0
                min_angle = 180
                prev_rv = rv
                for rrv in ridge_points_rv:
                    
                    if rrv == v0 or rrv == v1 or len(dict_ridge_points[rrv]) >= 2:
                        continue
                    rrv_x, rrv_y = vor.vertices[rrv]
                    rrv_angle = get_angle(v0_x, v0_y, rrv_x, rrv_y) 
                    # print(f"Checking potential triangle vertex {rrv} at ({rrv_x:.0f}, {rrv_y:.0f}) with max_angle {max_angle:.0f}")
                    if not right and rrv_angle > max_angle:
                        max_angle = rrv_angle
                        most_distant_rv = rrv
                    if right and rrv_angle < min_angle:
                        min_angle = rrv_angle
                        most_distant_rv = rrv
                if most_distant_rv is not None:
                    rv = most_distant_rv 
                    is_triangle_line_branching = True
                else:
                    continue
            if is_triangle_line_branching:
                side_vectors.append((rv, prev_rv))
            triangle_line = (v0, rv)
            side_vectors.append(triangle_line)
            is_triangle_line_found = True
            break
        if not is_triangle_line_found:
            return 0, side_vectors
        t_x = vor.vertices[triangle_line[1]][0] - v0_x
        t_y = vor.vertices[triangle_line[1]][1] - v0_y
        v_x = v1_x - v0_x
        v_y = v1_y - v0_y
    elif (len(ridge_points_v1)) >= 3:
        # print(f"V1 has {len(ridge_points_v1)} ridge points")
        is_triangle_line_found = False
        for rv in ridge_points_v1:
            rv_x, rv_y = vor.vertices[rv]
            # print(f"Checking ridge point {rv} at ({rv_x:.0f}, {rv_y:.0f}) for line {line}")
            if rv == v0 :
                continue
            
            if right and rv_x < v1_x:
                continue
            if not right and rv_x > v1_x:
                continue
            ridge_points_rv = dict_ridge_points.get(rv, [])
            is_triangle_line_branching = False
            if len(ridge_points_rv) >= 3:
                
                most_distant_rv = None
                prev_rv = rv
                max_angle = 0
                min_angle = 180
                for rrv in ridge_points_rv:
                    if rrv == v0 or rrv == v1 or len(dict_ridge_points[rrv]) >= 2:
                        continue
                    rrv_x, rrv_y = vor.vertices[rrv]
                    rrv_angle = get_angle(v1_x, v1_y, rrv_x, rrv_y) 
                    # print(f"Checking potential triangle vertex {rrv} at ({rrv_x:.0f}, {rrv_y:.0f}) with max_angle {max_angle:.0f}")
                    if not right and rrv_angle > max_angle:
                        max_angle = rrv_angle
                        most_distant_rv = rrv
                    if right and rrv_angle < min_angle:
                        min_angle = rrv_angle
                        most_distant_rv = rrv
                if most_distant_rv is not None:
                    is_triangle_line_branching = True
                    rv = most_distant_rv 
                else:
                    continue
            if is_triangle_line_branching:
                side_vectors.append((rv, prev_rv))
            triangle_line = (v1, rv)
            
            side_vectors.append(triangle_line)
            is_triangle_line_found = True
            break
        if not is_triangle_line_found:
            return 0, side_vectors
        t_x = vor.vertices[triangle_line[1]][0] - v1_x
        t_y = vor.vertices[triangle_line[1]][1] - v1_y
        v_x = v0_x - v1_x
        v_y = v0_y - v1_y
    # print(f"Triangle line for {line} is {triangle_line} with vertex at ({vor.vertices[triangle_line[1]][0]:.0f}, {vor.vertices[triangle_line[1]][1]:.0f})")
    area = 0.5 * abs(v_y * t_x - v_x * t_y)
    return area, side_vectors

def get_path_lines(skeleton_lines, vor, dict_ridge_points, side_vectors):
    path_lines = []
    for line in skeleton_lines:
        v0, v1 = line
        v0_x, v0_y = vor.vertices[v0]
        v1_x, v1_y = vor.vertices[v1]
        
        ridge_points_v0 = dict_ridge_points.get(v0, [])
        ridge_points_v1 = dict_ridge_points.get(v1, [])

        
        if len(ridge_points_v0) >= 3 and len(ridge_points_v1) >= 3:
            continue  # Skip junction-to-junction lines


        if line in side_vectors or (line[1], line[0]) in side_vectors or line in path_lines or (line[1], line[0]) in path_lines:
            continue  # Skip side vectors
        # print(f"Checking skeleton line {line} with coordinates {v0_x:.0f} and {v0_y:.0f} to {v1_x:.0f} and {v1_y:.0f}")                
        # for line in side_vectors:
        #     print(f"({vor.vertices[line[0]][0]:.0f}, {vor.vertices[line[0]][1]:.0f}) to ({vor.vertices[line[1]][0]:.0f}, {vor.vertices[line[1]][1]:.0f}) is a side line.")

        if len(ridge_points_v0) >= 3: # if v0 is a junction, check if there are only junctions connecting to it
            line = find_better_edge(vor, v0, v1, dict_ridge_points, ridge_points_v0, path_lines, side_vectors)
            if line is not None:
                path_lines.append(line)
        elif len(ridge_points_v1) >= 3: # if v1 is a junction, check if there are only junctions connecting to it
            line = find_better_edge(vor, v1, v0, dict_ridge_points, ridge_points_v1, path_lines, side_vectors)
            if line is not None:
                path_lines.append(line)
        
        
    return path_lines

def get_right_lowest_and_left_lowest_line(lines, vor):

    sorted_lines = sorted(
        lines,
        key=lambda l: max(vor.vertices[l[0]][0], vor.vertices[l[1]][0])
    )

    # 2. Take right-most half
    left_lines = sorted_lines[:len(sorted_lines) // 2]
    right_lines = sorted_lines[len(sorted_lines) // 2:]

    # 3. Find lowest endpoint among them
    leftmost_line, v = min(
        ((l, v) for l in left_lines for v in l),
        key=lambda lv: vor.vertices[lv[1]][1]
    )

    rightmost_line, v = min(
        ((l, v) for l in right_lines for v in l),
        key=lambda lv: vor.vertices[lv[1]][1]
    )
    return leftmost_line, rightmost_line

def find_straight_path(path_lines, vor, prev_coordinates):
    if prev_coordinates is not None:
        # print(f"Finding straight path closest to previous coordinates: ({prev_coordinates[0][0]:.0f}, {prev_coordinates[0][1]:.0f}) to ({prev_coordinates[1][0]:.0f}, {prev_coordinates[1][1]:.0f})")
        closest_distance = float('inf')
        possible_path = None
        possible_paths = []
        for line in path_lines:
            v0, v1 = line
            v0_x, v0_y = vor.vertices[v0]
            v1_x, v1_y = vor.vertices[v1]
            if v0_y > v1_y:
                v0_x, v0_y, v1_x, v1_y = v1_x, v1_y, v0_x, v0_y  # Ensure v0 is the lower point
            midpoint = get_midpoint(((v0_x, v0_y), (v1_x, v1_y)))
            distance = math.sqrt((midpoint[0] - prev_coordinates[0][0]) ** 2 + (midpoint[1] - prev_coordinates[0][1]) ** 2)
            if distance < closest_distance and distance < 150:  # Threshold for closeness
                closest_distance = distance
                possible_path = line
                possible_paths.append(line)
        
        
        best_angle = get_angle(vor.vertices[possible_path[0]][0], vor.vertices[possible_path[0]][1], vor.vertices[possible_path[1]][0], vor.vertices[possible_path[1]][1])
        return possible_paths, possible_path, best_angle
    else:
        best_angle = 0
        max_height = 0
        possible_path = None
        possible_paths = []
        for line in path_lines:
            v0, v1 = line
            v0_x, v0_y = vor.vertices[v0]
            v1_x, v1_y = vor.vertices[v1]
            if v0_y > v1_y:
                v0_x, v0_y, v1_x, v1_y = v1_x, v1_y, v0_x, v0_y  # Ensure v0 is the lower point
            
            angle = get_angle(v0_x, v0_y, v1_x, v1_y)
            if angle < 110 and angle > 70:  # Threshold for straightness
                if v1_y > max_height:
                    max_height = v1_y
                    possible_path = line
                    best_angle = angle
                    possible_paths.append(line)
        return possible_paths, possible_path, best_angle

def interpreting_skeletons(skeleton_lines, dict_ridge_points, vor, straight_path=None):
    # print("\nInterpreting skeleton lines:")
    # for vertexes in skeleton_lines:
      
    #     v0 = vor.vertices[vertexes[0]]
    #     v1 = vor.vertices[vertexes[1]]
    #     ridge_points_v0 = dict_ridge_points.get(vertexes[0], [])
    #     ridge_points_v1 = dict_ridge_points.get(vertexes[1], [])
    #     print(
    #         f"Skeleton edge: "
    #         f"[({v0[0]:.0f}, {v0[1]:.0f})] to [({v1[0]:.0f}, {v1[1]:.0f})], "
    #         f"Ridge points at v0: {ridge_points_v0}, Ridge points at v1: {ridge_points_v1}"
    #     )

    side_vectors = list()

    left_lowest_line, right_lowest_line = get_right_lowest_and_left_lowest_line(skeleton_lines, vor)

    side_vectors.append(left_lowest_line)
    side_vectors.append(right_lowest_line)

    area_left, side_vectors = get_vectors_area(left_lowest_line, skeleton_lines, dict_ridge_points, vor, side_vectors,False)
    area_right, side_vectors = get_vectors_area(right_lowest_line, skeleton_lines, dict_ridge_points, vor, side_vectors, True)

    area_difference = area_left - area_right
    area_percentage_difference = (area_difference / max(area_left, area_right)) * 100 if max(area_left, area_right) > 0 else 0


    # print(f"Area difference: {area_percentage_difference:.2f}")
    path_lines = get_path_lines(skeleton_lines, vor, dict_ridge_points, side_vectors)
    
    
    if len(path_lines) == 0:
        print("No path lines found.")
    elif len(path_lines) == 1:
        v0, v1 = path_lines[0]
        v0_x, v0_y = vor.vertices[v0]
        v1_x, v1_y = vor.vertices[v1]
        # Single path line found, likely a straight path.
        if v0_y > v1_y:
            v0_x, v0_y, v1_x, v1_y = v1_x, v1_y, v0_x, v0_y  # Ensure v0 is the lower point
        angle = get_angle(v0_x, v0_y, v1_x, v1_y)
        # print(f"Single path line found: ({v0_x:.0f}, {v0_y:.0f}) to ({v1_x:.0f}, {v1_y:.0f}). Angle: {angle:.2f} degrees.")
    else:
        # path lines found, likely a branching path.
        possible_paths, best_possible_path, best_angle = find_straight_path(path_lines, vor, straight_path)

    return path_lines, best_angle, area_percentage_difference, best_possible_path