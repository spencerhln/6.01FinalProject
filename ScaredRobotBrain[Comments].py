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
    
    
    
#### Global constants and variables that are passed around to helpers and robot functions ####

## Initializes the robot ##
robot = PioneerRobot()

## Maps robot behaviors returned by random_behavior to robot velocity tuples as (v_forward, v_rotate) ##
BEHAVIOR_DICT = {'stay': (0, 0),
                 'frwd': (0.5, 0),
                 'turn': (0, 1),
                 'search': (0, 0)}
## Turns the sound and sonar on the robot on if == 1, sound and sonar off if == 0
SENSORS_ON = 1 # 0 for False, 1 for True
## The initial distribution for robot behaviors. Favored to behaviors besides search to encourage
## period of random movement before beginning of search ##
BEHAV_DIST_1 = DDist({'stay': 0.33, 'frwd': 0.33, 'turn': 0.33, 'search': 0.01})
## Double that allows us to set the distance we want the robot to keep from any walls
WALL_DISTANCE = 0.5 # meters

## Behavior variable that is passed around functions. Initialized to inital
## robot probability distribution ##
behavior = BEHAV_DIST_1
## Integer that keeps track of the number of the same consective movements to help
## create a more realistic movement model ##
consecutives = 0
## Initializes previous behavior variable. Set to 'search' to immediately initiate search ##
prev_behav = 'stay'



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
    global BEHAV_DIST_1
    sonars = robot.sonars # Unpack sonar list to keep track of distance from wall
    ## Initiates search if we are in search mode and not simulated (simulator will
    ## throw error if we try to collect analog inputs from simulated robot) ##
    if prev_behav == 'search' and not robot.simulated:
        (robot.fv, robot.rv) = (0, 0) # Sets the fv and rv of the robot to 0
        ## Left to right - Analog inputs 1, 2, 3, 4. Voltages from the head
        ## are defined on 1, 2, 3. We use 2 and 3. ##
        v_neck, v_left, v_right, _ = robot.analogs 
        robot.set_analog_voltage(10) # Activates sound device to signal other robot
        ## Looks for a close enough IR signal. If a signal is found, that means the other robot is nearby
        if v_left > 0.5 or v_right > 0.5:
            behavior = BEHAV_DIST_1 # After search is concluded, returns to random movement
            robot.set_analog_voltage(0) # Disables the sound device
            prev_behav = 'stay' # Re-initializes the previous behavior to allow for random behavior
    else: # Continues random movement if not searching
        ## Checks to see if robot is near a wall, passed to random_behavior ##
        is_near_wall = any([sonars[x] < WALL_DISTANCE for x in range(1, 6) if sonars[x] != None])
        ## Chooses a new behavior using random_behavior ##
        this_behavior = random_behavior(prev_behav, is_near_wall)
        ## If the new behavior and the old behavior are the same, inrement the consecutive tracker.
        ## Otherwise, reset it ##
        consecutives += 1 if prev_behav == this_behavior else -consecutives
        (robot.fv, robot.rv) = BEHAVIOR_DICT[this_behavior] # Sets robots movement based on its new behavior
        prev_behav = this_behavior # Sets this new behavior as the next behavior for the next function pass
    

def on_stop():
    '''
    Executes when stopping the behavior of the robot.
    '''
    ## If we stop the behavior of the robot, disables the
    ## sound device to prevent undesired continued operation ##
    if not robot.simulated:
        robot.set_analog_voltage(0)

def on_shutdown():
    '''
    Executes when shutting down the robot.
    '''
    pass
