import math

# defining constants for PID to go forward
Kp = 0.5
Ki = 0.005
Kd = 0.7



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
    error_slope = 0
    # since the difference of the slope is slower than the intercept, I will divide by a factor
    factor = 100
    print(f"slope = {slope}")
    print(f"intercept = {intercept}")
    print(f'{len(error_list)}')
    if intercept > 405:
        error_intercept = (intercept - 405)/ factor
    elif intercept < 375:
        error_intercept = (intercept - 375)/factor
    else:
      if slope > 0.47:
          error_slope = 0.46-slope
      elif slope < 0.34:
          error_slope = 0.46-slope
    error = error_intercept + error_slope*2
    if len(error_list) > 100: # deleting the first element of the list, so that the integral part of the PID is not dominant
        error_list.pop(0) 
    if error_list[-1] != 0 or error != 0:
            error_list.append(error)
    t = len(error_list) - 1 # find what point in position t in the list the robot is now7
    # mulitplying by -1, because positive twist.angular.z turns right
    function = -1 * (Kp * error_list[t] + Ki * I_sum_list(error_list) + Kd * D_sum_list(error_list, t))
    angularz = function_modifier(function, 2)
    print(angularz)
    return angularz , error_list
