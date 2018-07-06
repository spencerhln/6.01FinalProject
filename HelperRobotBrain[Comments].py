import math  # Standard math import for using angles
## Special imports from class libraries ##
from soar.robot.pioneer import PioneerRobot  # Robot controller
from soar.robot.arcos import *  # Robot functions
from lib601.dist import *  # Probability functions written as part of the class

#### Helper functions ####

def random_behavior(prev_behav='stay', near_wall=False):
    '''
    returns one of four random behaviors: moving forward, sitting
    still, turning, and begin search. selects one of the previous four states 
    randomly and adjusts the probability of the other states to encourage 
    temporary repeatability but variety over the long-term.
    args: string, boolean
    ret: string
    '''
    global behavior  # Uses global behavior to allow for semi-random movement
    ## Continue search if we are already searching ##
    if prev_behav == 'search':
        return 'search'
    ## If we are near a wall, force behavior to give us 'turn' ##
    elif near_wall:
        return behavior.condition(lambda x: x not in ('frwd', 'search', 'stay')).draw()
    ## If the previos behavior was the same as the highest probability behavior, we update
    ## the distribution in a way that initially favors the highest probability but declines
    ## over time if the behavior stays the same ##
    elif prev_behav == behavior.max_prob_elt():
        behav_elts = behavior.support()  # Returns a list of potential outputs of behavior
        behav_elts.remove(prev_behav)  # Removes the previous behavior from that list
        uni_dist = uniform_dist(behav_elts)  # Creates a new uniform distribution over the new list
        ## Mixes the current behavior distribution with the new uniform one, weighting the
        ## current distribution so that the two are equally likely after 5000 timesteps ##
        behavior = mixture(behavior, uni_dist, (10000 - consecutives) / 10000)
        return behavior.draw()  # Draws and returns a new behavior
    ## If the previous behavior wasn't the highest probability behavior, update the
    ## distribution to favor a new chain of behaviors ##
    else: 
        ## Creates a new distribution with all probability mass on the previous behavior ##
        peak_dist = delta_dist(prev_behav)
        ## Mixes the current behavior distribution with the new spiked one, weighting the 
        ## current distribution so that it is 1/10 as likely as the spiked distribution. After
        ## four timesteps, the two distributions are equally likely, and after nine, the behavior
        ## won't change. This is to encourage picking a high probability behavior initially, 
        ## but allowing the specific probability mass to sort itself out if it chooses a low 
        ## probability outcome several times in a row ##
        behavior = mixture(behavior, peak_dist, min((consecutives + 1) / 10, 1))
        return behavior.draw()  # Draws and returns a new behavior

def is_near_wall():
    '''
    checks to see if the robot is too close to a wall or another object based on
    its sonar readings from the middle six sonars. returns True if the robot 
    is within the defined wall_dist, False if it is farther away.
    args: 
    ret: boolean
    '''
    global robot
    sonars = robot.sonars  # Accesses sonar data
    ## Checks to see if any of the sensors is close enough to warrant a close reading ##
    return any([sonars[x] < WALL_DISTANCE for x in range(1, 6) if sonars[x] != None])
 
def update_voltage_values():
    '''
    updates the global voltage_values list based on the current readings from the robot analogs.
    returns average value of the voltages in the list
    args: 
    returns: double
    '''
    global voltage_values
    global robot
    front, _2, _3, back = robot.analogs  # Unpack robot analog values from front and back mics
    voltage_values.append(max(front, back))  # Adds the higher mic input value
    voltage_values.pop(0)  # Removes the oldest value
    return sum(voltage_values) / len(voltage_values)  # Returns the average of the voltage values

def update_loc_obs():
    '''
    provides a DDist for the probability of a sound coming from a certain spot
    given a certain observation set. normalizes voltage values to create probability
    distribution
    args: 
    ret: DDist
    '''
    global loc_dict # Calls loc_dict for access to radian values and voltage values
    peak = sum(loc_dict.values()) # Sums voltage values
    ## Returns a new probability distribution that uses the voltages recorded at each position
    ## as the probability the other robot is at that position (voltages are normalized first).
    ## For example, if the loudest sound is at pi radians, the robot will read the highest voltage
    ## there and then assign that the highest probability mass by dividing all voltage values
    ## by their sum. ##
    return DDist({x: y / peak for x, y in zip(loc_dict.keys(), loc_dict.values())})



#### Global variables that are used in most functions ####
    
## Initializes the robot, defines pi from import ##
robot = PioneerRobot()
pi = math.pi
    
## Maps robot behaviors returned by random_behavior to robot velocity tuples as (v_forward, v_rotate) ##
BEHAVIOR_DICT = {'stay': (0, 0),
                 'frwd': (0.5, 0),
                 'turn': (0, 1),
                 'search': (0, 0)} 
## Turns the sound and sonar on the robot on if == 1, sound and sonar off if == 0
SENSORS_ON = 1 # 0 for False, 1 for True
## The initial distribution for robot behaviors. Does not include search because search
## is an induced behavior rather than a random one ##
INIT_BEHAV = DDist({'stay': 0.34, 'frwd': 0.33, 'turn': 0.33})
## Double that allows us to set the distance we want the robot to keep from any walls ##
WALL_DISTANCE = 0.5
## Double that sets our angle tolerance for the robot ##
ANGLE_TOL = 2 * pi / (half_length * 20)
## Number of discrete locations to record observations from ##
NUMBER_LOCATIONS = 16

## Behavior variable that is passed around functions. Initialized to inital probability distribution ##
behavior = INIT_BEHAV
## Integer that keeps track of the number of consective movements to help create a more
## realistic movement model ##
consecutives = 0
## Initializes previous behavior variable. Set to 'search' to immediately initiate search ##
prev_behav = 'stay'
## Integer that tracks which how many revolutions the robot has completed. One full observation
## cycle is two rotations $$
to_search = 0
## List that tracks the previous 10 front and rear mic values to detect if the other robot is making a sound
voltage_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
## List of locations in ascending order of radians ##
locations = [x / (NUMBER_LOCATIONS / 2) * pi for x in range(0, NUMBER_LOCATIONS)]
## Dictionary with all the radian values of locations. For storing intermediate reading values ##
loc_dict = {x: 0 for x in locations}
## DDist that represents the robot's belief of what direction the sound is coming from ##
loc_belief = uniform_dist(locations)
## Half the number of locations ##
half_length = int(NUMBER_LOCATIONS / 2)
## Location index that keeps track of which angle we are updating as we increment on_step ##
loc_index = 0


#### Robot functions ####

def on_load():
    '''
    Executes when loading the robot brain.
    '''
    ## Activates or deactivates sonar and sound based on our variables,
    ## only if we are connected to a real robot ##
    if not robot.simulated:
        robot.arcos.send_command(SOUNDTOG, SENSORS_ON)
        robot.arcos.send_command(SONAR, SENSORS_ON)
        
def on_start():
    '''
    Executes when the brain begins running.
    '''
    pass

def on_step(step_duration):
    '''
    Executes every timestep. On each timestep, picks a new behavior from its probability
    distribution and then updates its motion and sound characteristics based on its chosen behavior.
    '''
    ## Global variables, defined to allow use between functions ##
    global behavior
    global consecutives
    global prev_behav
    global INIT_BEHAV
    global locations
    global loc_belief
    global loc_dict
    global loc_index
    global half_length
    global to_search
    global voltage_values
    ### Operates search procedure ###
    ## Initiates search if we are in search mode and not simulated (simulator will
    ## throw error if we try to collect analog inputs from simulated robot) ##
    if prev_behav == 'search' and not robot.simulated:
        (px, py, ptheta) = robot.pose # Unpacks angle value for use in both belief update and movement
        front, _2, _3, back = robot.analogs # Unpacks analog mic values for use in belief update
        ## Checks if the average voltage value from the last 10 timesteps is still occurring. If not,
        ## returns to normal behavior ##
        if update_voltage_values() < 2:
            prev_behav = random_behavior()
            return
        ### Belief update ###
        ## Rotates robot in a complete circle twice, stopping at each angle (based on our provided
        ## division of the unit circle) to record voltage at that position. After each rotation,
        ## it stores these values and updates its belief state. ##
        elif to_search < 2: # Checks to see if we have performed less than two rotations of the robot
            if abs(ptheta - locations[loc_index]) > ANGLE_TOL:  # Moves the robot to the next angle
                robot.rv = 0.5
                return
            ## Otherwise, the robot is at an observable angle. Sets the current angle to the front
            ## mic voltage and the corresponding read angle to the back mic voltage ##
            robot.rv = 0
            loc_index += 1
            loc_dict[locations[loc_index]] = front 
            loc_dict[locations[(loc_index + half_length) % (half_length * 2)]] = back
            if (loc_index + 1) % half_length == 0:  # Robot has finished one rotation, updates belief
                ## Updates the belief probability distribution, giving equal weight to present and
                ## past observations ##
                loc_belief = mixture(loc_belief, update_loc_obs(), 0.5)
                ## If we have completed a whole revolution, reset the angle counter and
                ## increment the number of revolutions ##
                if loc_index == half_length * 2: 
                    loc_index = 0
                    to_search += 1
                return
        ### After two rotations, moves the robot based on its belief state ###
        else:
            desired_angle = loc_belief.max_prob_elt()  # Desired angle for movement is highest prob
            ## If we are at the correct angle but facing an object, associate that angles
            ## probability mass with the angle directly behind it due to sound reflection ##
            if abs(ptheta - desired_angle) < ANGLE_TOL and is_near_wall(): 
                ## Unpacks probability distribution into a dict, then calculates the opposite angle,
                ## reassigns probability masses, creates a new distribution, and chooses the new
                ## desired angle based on those, new probability masses ##
                new_belief = {x: loc_belief.prob(x) for x in loc_belief.support()}
                opposite_angle = locations[(locations.index(desired_angle) + half_length) % (half_length * 2)]
                new_belief[opposite_angle] = sum((new_belief.get(opposite_angle, 0),
                                                  new_belief.get(desired_angle, 0)))
                new_belief[desired_angle] = 0
                loc_belief = DDist(new_belief)
                desired_angle = loc_belief.max_prob_elt()
            ## If we aren't at the correct angle, turn proportionally until we reach it ##
            if abs(ptheta - desired_angle) > ANGLE_TOL:
                (robot.fv, robot.rv) = (0, -1 * (ptheta - desired_angle))
                return
            ## Otherwise, we're at the angle but not there, so move forward ##
            elif not is_near_wall():
                (robot.fv, robot.rv) = (0.5, 0)
                ## If the robot detects a voltage that is less than the one it previously measured,
                ## it has moved away from the sound source, so it stops and re-searches ##
                if (front - loc_dict[desired_angle]) < 0.2: 
                    robot.fv = 0
                    to_search = 0
                return
    ## Operates random movement if the robot hasn't been activated yet ##
    else:
        ## Selects a random behavior based on the probability distribution ## 
        this_behavior = random_behavior(prev_behav, is_near_wall()) 
        consecutives = consecutives + 1 if prev_behav == this_behavior else 0
        ## If the average of the voltage values is greater than 2, initiate search
        ## and activate the search light ##
        if update_voltage_values() > 2: 
            this_behavior = 'search'
            robot.set_analog_voltage(3.3)
        (robot.fv, robot.rv) = BEHAVIOR_DICT[this_behavior] # Sets robots movement based on its new behavior
        prev_behav = this_behavior # Sets this behavior as the previous behavior

def on_stop():
    '''
    Executes when stopping the behavior of the robot.
    '''
    robot.set_analog_voltage(3.3)  # Turns search light on

def on_shutdown():
    '''
    Executes when shutting down the robot.
    '''
    pass
