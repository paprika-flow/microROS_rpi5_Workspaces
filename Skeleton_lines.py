import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from get_skeleton_lines import get_skeleton_lines
from shapely.geometry import Polygon, LineString, Point
import time
from interpreting_skeleton_lines import interpreting_skeletons

# Combine in continuous order
def process_image_and_compute_skeleton(img, straight_path=None, plot=False):
    # Ensure binary (white = object)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,   # only outer boundary
        cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=cv2.contourArea)
    boundary_points = contour.squeeze()


    # flip Y axis (image → Cartesian)
    h = binary.shape[0]
    boundary_points[:, 1] = h - boundary_points[:, 1]

    # downsample boundary
    boundary_points_ordered = boundary_points[::50]  # take every 20th pointq

    polygon = Polygon(boundary_points).buffer(0)


    # ---------------------------q-
    # 2. Compute Voronoi diagram
    # ----------------------------
    vor = Voronoi(boundary_points_ordered)

    # ----------------------------
    # 3. Function to extract skeleton edges (finite ridges only)
    # ----------------------------
    edge_lines = []
    for rv in vor.ridge_vertices:
        if -1 in rv:
            continue  # skip infinite edges
        v0, v1 = vor.vertices[rv[0]], vor.vertices[rv[1]]
        line = LineString([v0, v1])
        if polygon.covers(line):  # keep only edges fully inside
            edge_lines.append((v0, v1))
    skeleton_lines, dict_ridge_points = get_skeleton_lines(vor, polygon)

    # print(f"Skeleton lines for image {img.shape}:")

    paths, best_angle, area_percentage_difference, best_possible_path = interpreting_skeletons(skeleton_lines, dict_ridge_points, vor, straight_path)

    # print(f"Best angle: {best_angle:.2f} degrees")
    # print(f"Area percentage difference: {area_percentage_difference:.2f}%")
    # if best_possible_path is not None:
    #     print(f"Best possible path: {vor.vertices[best_possible_path[0]][0]:.0f}, {vor.vertices[best_possible_path[0]][1]:.0f} to {vor.vertices[best_possible_path[1]][0]:.0f}, {vor.vertices[best_possible_path[1]][1]:.0f}")
    # else:
    #     print("No straight path found.")
    # ----------------------------
    # 4. Plot polygon + skeleton
    # ----------------------------
    if plot:
        plt.figure(figsize=(7,7))
        plt.plot(boundary_points_ordered[:,0], boundary_points_ordered[:,1], 'ro-', label='Polygon boundary')
        plt.fill(boundary_points_ordered[:,0], boundary_points_ordered[:,1], alpha=0.2)
        for v0, v1 in skeleton_lines:
            plt.plot([vor.vertices[v0][0], vor.vertices[v1][0]], [vor.vertices[v0][1], vor.vertices[v1][1]], 'b-', linewidth=2)
        for v0, v1 in paths:
            plt.plot([vor.vertices[v0][0], vor.vertices[v1][0]], [vor.vertices[v0][1], vor.vertices[v1][1]], 'g-', linewidth=2)
        
        plt.gca().set_aspect('equal')
        plt.title("Voronoi Skeleton Inside Weird Polygon")
        plt.legend()
        plt.show()  
    return ((vor.vertices[best_possible_path[0]][0], vor.vertices[best_possible_path[0]][1]), (vor.vertices[best_possible_path[1]][0], vor.vertices[best_possible_path[1]][1])) if best_possible_path is not None else None, best_angle, area_percentage_difference

if __name__ == "__main__":
    start_time = time.time()

    img = cv2.imread("photos\\img_20260213_041057_sidewalk_mask.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("photos\\img_20260213_041600_sidewalk_mask.png", cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread("photos\\image222.png", cv2.IMREAD_GRAYSCALE)

    # Resize all images to (480, 640)
    target_size = (960, 720)  # OpenCV uses (width, height)

    img = cv2.resize(img, target_size)
    img2 = cv2.resize(img2, target_size)
    img3 = cv2.resize(img3, target_size)

    straight_path, best_angle, area_percentage_difference = process_image_and_compute_skeleton(img, plot=True)
    straight_path, best_angle, area_percentage_difference = process_image_and_compute_skeleton(img2, straight_path, plot=True)
    straight_path, best_angle, area_percentage_difference = process_image_and_compute_skeleton(img3, plot=True)
    print("Execution time: %.2f seconds" % (time.time() - start_time))