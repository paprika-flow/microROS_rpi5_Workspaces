import math

# defining constants for PID to go forward
Kp = 0.5
Ki = 0.001
Kd = 0.05

#defingin constants for PID to maintain a distance from the wall
Kpw = 2
Kiw = 0.001
Kdw = 2

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

def PID_sidewalk(slope, intercept, error_list):
    error_intercept = 0
    # since the difference of the slope is slower than the intercept, I will divide by a factor
    factor = 200
    print(f"slope = {slope}")
    print(f"intercept = {intercept}")
    if intercept > 530:
        error_intercept = (intercept - 530)/ factor
    elif intercept < 420:
        error_intercept = (intercept - 420)/factor
    error_slope = 0
    if slope > 0.82:
        error_slope = slope - 0.82
    elif slope < 0.64:
        error_slope = slope - 0.64
    error = error_intercept/2 + error_slope
    if len(error_list) > 25: # deleting the first element of the list, so that the integral part of the PID is not dominant
        error_list.pop(0)
    error_list.append(error)
    t = len(error_list) - 1 # find what point in position t in the list the robot is now7
    # mulitplying by -1, because positive twist.angular.z turns right
    function = -1 * (Kp * error_list[t] + Ki * I_sum_list(error_list) + Kd * D_sum_list(error_list, t))
    angularz = function_modifier(function, 2)
    print(angularz)
    return angularz , error_list
