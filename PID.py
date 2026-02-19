from Skeleton_lines import process_image_and_compute_skeleton
import cv2
import math

# defining constants for PID to go forward
Kp = 0.5
Ki = 0.005
Kd = 0.25



def function_modifier(function, limit):
    # a function that limit another function regarding limits given
    if function > limit:
        return limit
    elif function < - limit:
        return -limit
    else:
        return function

# all the function for each part of PID calculations
def I_sum_list(list):
    summation = 0
    for i in list:
        summation += i
    return summation

def D_sum_list(list, t):
    derivative = list[t] - list[t-1]
    return derivative

def PID_sidewalk(area_percentage_difference, angle, error_list):
    error_area = 0
    error_angle = 0
    # since the difference of the slope is slower than the intercept, I will divide by a factor
    factor = 240
    factor_angle = 100
    

    if area_percentage_difference > 40:
        error_area = (area_percentage_difference - 40)/ factor
    elif area_percentage_difference < -40:
        error_area = (area_percentage_difference + 40)/factor
   
    if angle > 95:
        error_angle = (angle - 95)/factor_angle
    elif angle < 85:
        error_angle = (angle-85)/factor_angle
    error = error_area + error_angle
    if len(error_list) > 100: # deleting the first element of the list, so that the integral part of the PID is not dominant
        error_list.pop(0) 
    if error_list[-1] != 0 or error != 0:
            error_list.append(error)
    t = len(error_list) - 1 # find what point in position t in the list the robot is now7
    # mulitplying by -1, because positive twist.angular.z turns right
    function = (Kp * error_list[t] + Ki * I_sum_list(error_list) + Kd * D_sum_list(error_list, t))
    angularz = function_modifier(function, 0.2)

    return angularz , error_list