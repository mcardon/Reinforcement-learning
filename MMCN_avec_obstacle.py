# -*- coding: utf-8 -*-
"""
May 2016
MMCN - Project n°10
Melissa Cardon - Chloé Quignot
"""

import numpy as np
import matplotlib.pylab as plt
import time

Save = True
Show = False
folder = "res_Obs/"
#=======================================================================================
# 							INITIALISING THE PARAMETERS
#=======================================================================================

#Global parameters
A = 1                       # a position cell's maximum activity
Sigma = 5                   # width of the position cells' receptor field
Eta = 0.01                  # learning rate
Gamma = 0.9                 # discount factor
Platform = [50,25,10]       # coordinates of the platform [x0,y0,r] where (x0,y0) is the
							# centre and r the radius
Pool = [100,100]            # [w, h], width and height of the rectangular pool
Speed = 16                  # speed of the animal
Dt = 0.125                  # time interval between 2 moves
Distance = Speed*Dt         # distance covered in one move
Nb_cells_per_m = 20         # number of position neurones per meter in the pool
Obstacle = [20,40,60,10]	#dimensions of the rectangular obstacle

#Messages to print
fail_txt = "----------------- Did not find the platform after %d steps and made %d hits\
 on the walls"
hit_txt  = "+++++++++++++++++ Found the platform after %d steps and made %d hits on the\
 walls"

#to create an array containing the coordinates of all position cells
def init_pos_cells(nb_cells_by_m, width_pool_in_cm, len_pool_in_cm):
    """
    nb_cells_by_m : int : number of cells by m in the rat model
    width_pool : int : width of the simulated pool
    len_pool : int : length of the simulated pool

    Returns np_array of shape (2,nb_cells) with prefered coordinates of position cells
    in the pool
    """
    width_pool = width_pool_in_cm / float(100)
    len_pool = len_pool_in_cm /float(100)

    nb_cells = len_pool*width_pool*nb_cells_by_m
    area = len_pool * width_pool
    dist_between_cells = area / float(nb_cells)
    half_dist = dist_between_cells / 2

    x = [ [ ((i*dist_between_cells)+half_dist)*100 for i in range(int(nb_cells / len_pool))]  for j in range(int(nb_cells / width_pool )) ]
    preferred_x = []

    for i in range(len(x)):
        preferred_x = preferred_x + x[i]

    y = [ [ ((j*dist_between_cells)+half_dist)*100 for i in range(int(nb_cells / len_pool)) ] for j in range(int(nb_cells / width_pool )) ]
    preferred_y = []

    for i in range(len(y)):
        preferred_y = preferred_y + y[i]

    return np.array([preferred_x, preferred_y])


# np.array of the coordinates of all position cells (xj,yj) = global parameters
Pos_cells = init_pos_cells(Nb_cells_per_m, Pool[0], Pool[1])


# normal law
def norm(x, mu, sigma):
	"""
	x: float
	mu: float, centre of normal law
	sigma: float, standard error of the normal law
	
	Returns the value of the normal function for x
	"""
	y = 1/(sigma*(2*np.pi)**0.5)*np.exp(-0.5*((x-mu)/float(sigma))**2)

	return y

# persistance of the rat
# weighted decision of the next direction (according to the last move made)
def persistance(nb_motor_cells, sig, mu):
	"""
	nb_motor_cells: number of cells per axis of the pool
	sig: standard deviation for the normal law used to calculate the persistance
	mu: centre of the normal law used to calculate the persistance
	
	Returns np.array of 8 list (one for each motor cell)
	each list gives weights of the preference of the rat for the next direction choice 
	(prefers to go in the same direction, and does not want to got to the opposite)

	Persist can be a global parameter, or can vary between differents rats
	"""
	x = np.arange(nb_motor_cells)
	p = [norm(i, mu, sig) for i in x]

	persist = []
	for i in range(nb_motor_cells):
		persist_i = [p[(4-i+j)%nb_motor_cells] for j in range(nb_motor_cells)]
		persist = persist + [persist_i]

	persist = np.array(persist)

	return persist

# persistance of the rat = global parameter
Persist = persistance(8, 2.0, 4.)
# array with dimensions (8,8)
# each line has a set of probabilities that follow the normal law and centered around the index of the line 



#=======================================================================================
# 						BASIC FUNCTIONS FOR THE SIMULATION
#=======================================================================================
#To check if a position is inside the obstacle
def inrectangle(position):
	"""
	position: [x,y]
	
	Returns True if the rat/position is inside the obstacle
	"""
	[x,y] = position
	[x_r,y_r,w_r,h_r] = Obstacle
	
	if x_r <= x <= (x_r + w_r) and y_r <= y <= (y_r + h_r):
		res = True
	else:
		res = False

	return res


def ccw(A,B,C):
	return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


#To check if the rat crosses the obstacle or one of its sides
def crossrectangle(P1, P2):
	"""
	P1 & P2: two positions i.e. current_position and next_position
	
	Returns True if the rat crosses the obstacle or one of its sides
	"""
	res = False
	
	[x_r,y_r,w_r,h_r] = Obstacle
	
	#4 rectangle sides
	rectangle_sides = [[(x_r,y_r),(x_r,y_r+h_r)],[(x_r,y_r),(x_r+w_r,y_r)],[(x_r+w_r,y_r),(x_r+w_r,y_r+h_r)],[(x_r,y_r+h_r),(x_r+w_r,y_r+h_r)]]
	
	for [R1,R2] in rectangle_sides:
		if intersect(P1,P2,R1,R2):
			res = True
			break
    
	return res


#To calculate the distance between the position of the animal and the centre of the platform
def dist2centre(position):
    """
    position: [x,y] coordinates of the animal's current position

    
    Platform: (global parameter) coordinates of the platform [x0,y0,r] where (x0,y0) is the centre and r the radius
    
    Returns the distance between the current position and the centre of the platform
    """
    [x0, y0] = Platform[:2]        #centre coordinates of the platform
    [x, y] = position              #current position of the animal
    
    dist = ((x - x0)**2 + (y - y0)**2)**0.5

    
    return dist
    

def dist2points(pos1, pos2):
    """
    calculate the distance between 2 points
    """
    [x1, y1] = pos1
    [x2, y2] = pos2    
    
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    return dist


#To calculate the activity rj (in [0,1]) of the position cell j when the animal is at the position pos
def r_j(pos, cell):
    """
    pos: [x,y] coordinates of the animal's current position
    cell: [xj,yj] the position cell's centre coordinates 
    
    A: (global parameter) a position cell's maximum activity
    Sigma: (global parameter) width of the position cells' receptor field
    
    Returns the activity of the postition cell j given the animal's position (pos)
    """
    [x,y] = pos         #current position of the animal
    [xj,yj] = cell      #position of the position neurone j
    
    activity = A*np.exp((-(xj-x)**2 - (yj-y)**2) / (2*Sigma**2))
    
    return activity


#To create a table of 30 random initial positions different from the hidden Platform in the watermaze
def init_pos(nb):
    """
    nb: number of initial positions
    
    Platform: (global parameter) coordinates of the platform [x0,y0,r] where (x0,y0) is the centre and r the radius
    Pool: (global parameter) [w, h], width and height of the rectangular pool

    
    Returns an array (2xnb) of nb initial positions (x = 1st line, y = 2nd line)
    """
    [width,height] = Pool        #coordinates of the pool
    r = Platform[2]              #radius of the platform
    
    i = 0
    pos = np.zeros((2,nb))       #array of coordinates [[x_values], [y_values]]
    
    while i < nb:
        rand_x = np.random.uniform(0,width)
        rand_y = np.random.uniform(0,height)
        if rand_x != 0 and rand_y != 0 and dist2centre([rand_x,rand_y]) > 2*r and not inrectangle([rand_x,rand_y]): 
            pos[0,i] = rand_x
            pos[1,i] = rand_y
            i+=1
    
    return pos


#To calculate the new position of the animal from the current position, the direction and the distance
def new_pos(a, current_pos):
    """
    a: index of the motor cell (i.e. direction) that was chosen
    current_pos: current position in centimetres [x1,y1]
    
    Distance: (global parameter) distance in centimetres between 2 positions deduced from the animal's speed and the time interval between each move
    
    Returns a list [x2,y2] of the new position calculated from the current position, the direction angle and the distance between 2 positions
    """
    x = Distance * np.cos(np.pi*a/4) + current_pos[0]
    y = Distance * np.sin(np.pi*a/4) + current_pos[1]
    
    return [x,y]


#To calculate the activities of all postion cell for a given position
def pos_activity(current_pos):
    """
    current_pos: [x,y] position of the animal
    
    Pos_cells: (global parameters) np.array([[x values],[y values]]) of the coordinates of all position cells (xj,yj)
    
    Returns the activity (r_j) of each position cell as a vector of 400 values
    """

    all_activity = [ r_j(current_pos, Pos_cells[:,j]) for j in range(len(Pos_cells[0])) ]
    
    return np.array(all_activity)


def calcQ(W, all_activity, current_pos):
	"""
	W: set of 8 weight vectors (one for each direction i.e motor cell) (8 x 400)
	all_activity : activity (r_j) of each position cell (vector of 400 values)
	current_pos: [x,y] position of the animal
	
	Returns Q (vector containing the activities of the 8 motor cells)
	"""
	Q = [0 for i in range(len(W))]
	for i in range(len(W)):
		Q[i] = sum(W[i]*all_activity)

	return Q


#To calculate the activity of each motor cell and choose the best direction 
def best_direction(W, current_pos, previous_a):
	"""
    W: set of 8 weight vectors (one for each direction i.e motor cell) (8 x 400)
    current_pos: [x,y] position of the animal
    previous_a: index of the direction chosen at the previous position of the animal

    Persist : (global parameters) persistance of the rat
    
    Returns the index of the motor cell with the highest activity (i.e. best direction) 
    chosen with weight according to the previous direction
	"""
	#activity (r_j) of each position cell (vector of 400 values)
	all_activity = pos_activity(current_pos) 
	
	#activity of each motor cell at current_position
	Q_sa = calcQ(W, all_activity, current_pos)  
    
	#weighted activities
	weighted_Q_sa = list(Q_sa*Persist[previous_a])

	best_a = weighted_Q_sa.index(max(weighted_Q_sa)) #direction with best activity

	return best_a


#To choose exploration with a probability epsilon or exploitation with a probability (1-epsilon)
def direction(eps, W, current_pos, previous_a, force_exploration):
    """
    eps: probability of exploration
    W: set of 8 weight vectors (one for each direction i.e motor cell) (8 x 400)
    current_pos: [x,y] position of the animal
    previous_a: index of the direction chosen at the previous position of the animal
    force_exploration : int : nb of turn with forced exploration 
    
    Pos_cells: (global parameters) np.array([[x values],[y values]]) of the coordinates of all position cells (xj,yj)        
    
    Returns the direction in which the animal has to move according to the pobability of exploration/exploitation
    """
    if force_exploration <=0:
        rand = np.random.binomial(1,1-eps)
    else:
        rand = 0

    if rand:    
        #exploitation
        a = best_direction(W, current_pos, previous_a)
    else:       
        #exploration
        a = np.random.choice(range(8), p=Persist[previous_a] / sum(Persist[previous_a]) )
    
    return a


#To update W
def update_W(W, current_pos, next_pos, current_a, next_a, R):
    """
    W: set of 8 weight vectors (one for each direction i.e motor cell) (8 x 400)
    current_pos: [x,y] position of the animal
    current_a: index of the direction chosen in the current position of the animal
    next_a: index of the direction that will be chosen in the next position of the animal
    R: reward (-5, 0 or +20 depending on where the next position lays)

    Pos_cells: (global parameters) np.array([[x values],[y values]]) of the coordinates of all position cells (xj,yj)        
    Eta: (global parameter) learning rate
    Gamma: (global parameter) discount factor

    Returns the updated version of Q and W
    """
    all_activity = pos_activity(current_pos)
    
    #activity of the current motor cell at current_position
    Q_current_a = sum(W[current_a]*all_activity)
    
    #activity of the next motor cell at next_position
    Q_next_a = sum(W[next_a]*pos_activity(next_pos))

    delta_n = R + (Gamma * Q_next_a) - Q_current_a
    W[current_a] += Eta*all_activity*delta_n  

    return W

#=======================================================================================
# 						  MAIN FUNCTIONS FOR THE SIMULATION
#=======================================================================================

def one_simulation(W, pos_init, eps, nb_sim, rat_num):
    """
    W: set of 8 weight vectors (one for each direction i.e motor cell) (8 x 400)
    pos_init: [x,y] initial position of the animal
    eps: pobability of exploration
    nb_sim: simulation number
    rat_num: rat number
    
    Platform: (global parameter) coordinates of the platform [x0,y0,r] where (x0,y0) is the centre and r the radius
    Pool: (global parameter) [w, h], width and height of the rectangular pool    
    
    Returns the path of the animal and the updated Q and W arrays
    """    
    [width, height] = Pool                      #coordinates of the pool
    [x_p,y_p,r_p] = Platform                    #radius of the platform

    path = [[pos_init[0]], [pos_init[1]]]       #array to keen track of the animal's
    											#path

    current_pos = pos_init                      #current position of the animal [x,y]
    pos_black_hole = current_pos                #memory to check if the rat is stuck
    force_exploration = 0                       #nb turn to force random exploration
    nb_forced_explor = 0
    previous_a = np.random.randint(8)           #random previous direction at first

    #direction at current position (= number between 0 and 7):
    current_a = direction(eps, W, current_pos, previous_a, force_exploration)  

    res = 0
    tour = 0 		# compteur
    nb_bruises = 0 	# number of times the animal hits the wall
    nb_max_try = 5000
    
    #while animal not on platform
    while (res == 0) & (tour<nb_max_try) :

        if (tour%50 == 0) & (tour != 0):
            dist_50_before = dist2points(pos_black_hole, current_pos)
            print("Distance efficace depuis 50 tours : " + str(dist_50_before))
            if dist_50_before < 2*Platform[2]:
                force_exploration = min(5*nb_forced_explor, 15)
                nb_forced_explor = min(3, nb_forced_explor + 1)
                print(str(force_exploration) + " tours exploration forcee")
            else:
                nb_forced_explor = max(0, nb_forced_explor -1)
            pos_black_hole = current_pos
            

        # decrease counter of forced exploration
        if force_exploration >0:
            force_exploration -=1

        #define new position
        next_pos = new_pos(current_a, current_pos)
        tour +=1

        #check if new position is inside pool or on platform and doesn't cross the wall
        if (next_pos[0] <= 0 or next_pos[0] >= width) or (next_pos[1] <= 0 or next_pos[1] >= height) :
            #animal will cross the wall
            R = -5                        #negative reward
            nb_bruises +=1
            
        else:
            if crossrectangle(current_pos, next_pos): 
                #animal will cross the wall
                R = -5                        #negative rewar
                nb_bruises +=1
            
            else:
                dist = dist2centre(next_pos)  #distance to platform centre
                if dist > r_p: 
                    R = 0
                else:                         #on platform
                    R = 20                    #positive reward
                    res = 1                   #stop experiment
                    print(hit_txt %(tour, nb_bruises))

        #choose next direction               
        next_a = direction(eps, W, next_pos, current_a, force_exploration)

        W = update_W(W, current_pos, next_pos, current_a, next_a, R) 
        
        if R == -5:
            next_pos = current_pos        #stay at same position
            next_a = direction(eps, W, next_pos, current_a, force_exploration)
            
        if (tour == nb_max_try):
            print(fail_txt %(tour, nb_bruises))
        
        #move to next position and add position to parameterqh
        current_pos = next_pos
        current_a = next_a
        path[0] += [current_pos[0]]
        path[1] += [current_pos[1]]
    
    check_path(path, tour, eps, nb_bruises, nb_sim, rat_num, W)
	
    return np.array(path), W, nb_bruises
    
    
##############################################################################

def sim_one_animal(nb_sim, rat_num):
    """    
    nb_sim: number of simulations
    rat_num: rat number
    
    Returns the first and last itinerary of the animal, the times of each simulation and the initial and final W, i.e. [time, pathi, path, Wi, W]
    """    
    #initialisation of W
    Wi = np.array([[np.random.uniform(-0.4,0.4) for i in range(len(Pos_cells[0]))] for j in range(8)])
    #Wi = np.ones((8,400))         
    pos_init = init_pos(nb_sim)   #set of random initial positions
    time_list = []                #to save the duration of each simulation
    bruises_list = []

    #first simulation
    pathi, W, nb_bruises = one_simulation(Wi.copy(), pos_init[:,0], 1/1.05, 1, rat_num)

    time_list.append((len(pathi[0])-1)*Dt)
    bruises_list.append(nb_bruises)
    
    #next 29 simulations
    for i in range(1,nb_sim):
        print(i)
        path, W, nb_bruises = one_simulation(W, pos_init[:,i], (1.05**(-i-1)), i+1, rat_num)  
        #eps = 1.1^-(i+1), starts close to 1 and decreases to nearly 0 after 30
        #iterations
        time_list.append((len(path[0])-1)*Dt)
        bruises_list.append(nb_bruises)
    
    return [np.array(time_list), pathi, path, Wi, W, np.array(bruises_list)]
    
    
    
##############################################################################
  
def sim_several_animals(nb_sim, nb_an):
    """    
    nb_sim: number of simulations
    nb_an: number of animals
    
    Returns the first and last itinerary, the times of each simulation and the initial and final W for each animal, i.e. [[times], [pathis], [paths], [Wis], [Ws]]]
    """    
    times = []
    pathis = []
    paths = []
    Wis = []
    Ws = []

    for i in range(nb_an):

        [time_list, pathi, path, Wi, W, bruises_list] = sim_one_animal(nb_sim, i+1)   #simulation for one animal
        times.append(time_list)
        pathis.append(pathi)
        paths.append(path)
        Wis.append(Wi)
        Ws.append(W)
        np.save(folder+str(i+1)+"_time",time_list)
        np.save(folder+str(i+1)+"_pathi",pathi)
        np.save(folder+str(i+1)+"_path",path)
        np.save(folder+str(i+1)+"_Wi",Wi)
        np.save(folder+str(i+1)+"_W",W)
 		
 		#plot performance
        bin_count = 1
        perf1 = performance_rat(time_list, bruises_list, bin_count, i+1)

        bin_count = 5
        perf2 = performance_rat(time_list, bruises_list, bin_count, i+1)


    return [np.array(times),np.array(pathis),np.array(paths),np.array(Wis),np.array(Ws)]
    

#=======================================================================================
# 						FUNCTIONS TO PLOT PATH AND EVOLUTION OF TIME
#=======================================================================================

def performance_rat(time_list, bruises_list , bin_count, rat_num):
    """
    save the plot of performance of rat
    """
    #len_paths = [len(paths[i]) for i in range(len(paths))]
    perf = [sum(time_list[i:i+bin_count])/float(bin_count) for i in range(len(time_list)-bin_count+1)]
    bru = [sum(bruises_list[i:i+bin_count])/float(bin_count) for i in range(len(bruises_list)-bin_count+1)]
    x = range(1,len(time_list)-bin_count+2)

    #timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(3)
    plt.plot(x,perf)
    plt.ylim(0,max(500, max(time_list)+20))
    plt.title("Time to reach the platform (mean on " + str(bin_count) + " tests)")
    plt.xlabel("Simulation number")
    if Save == True:
        plt.savefig(folder+"Performance_rat_" + str(rat_num) + "_bin_" + str(bin_count)+".png")
    if Show == True:
        plt.show()
    plt.close()

    plt.figure(4)
    plt.plot(x,bru)
    plt.ylim(0,max(100, max(bru)+2))
    plt.title("Nb hits on the walls (mean on " + str(bin_count) + " tests)")
    plt.xlabel("Simulation number")
    if Save == True:
        plt.savefig(folder+"Hits_walls_rat_" + str(rat_num) + "_bin_" + str(bin_count)+".png")
    if Show == True:
        plt.show()
    plt.close()

    return perf, bru


def check_path(dots,tour,eps, nb_bruises, nb_sim, rat_num, W):
    """
    dots: np.array([[x values],[y values]]), data points/positions in the Pool in centimetres
    nb_sim: simulation number
    rat_num: rat number

    Pool: (global parameter) [width, height], dimensions of the Pool in centimetres
    Platform: (global parameter) [x, y, r], dimensions of the Platform, (x,y) = centre coordinates & r = radius in centimetres
    
    Plots the animal's positions in the pool and the position of the platform and obstacle if there is one
    """
    [w_p,h_p] = Pool
    [x_c,y_c,r_c] = Platform
    [x_r,y_r,w_r,h_r] = Obstacle #dimensions of the rectangular obstacle
    
    a = np.arange(len(Divpool[0]))
    for i in range(len(Divpool[0])):  
        a[i] = np.argmax([W[j].dot(pos_activity([Divpool[0][i],Divpool[1][i]])) for j in range(len(W))])
        
    fig1 = plt.figure()
    
    #path subplot
    ax1 = fig1.add_subplot(1, 2, 1, aspect=1) #aspect=1: the plot has a squar shape
    ax1.plot(dots[0], dots[1], 'g-', markersize=4)
    ax1.plot(dots[0][0], dots[1][0], 'ro') #starting point in red
    circle = plt.Circle((x_c,y_c), radius=r_c, color='k', hatch='///', fill=False)
    ax1.add_patch(circle)
    rectangle = plt.Rectangle((x_r,y_r), w_r, h_r , color='k', hatch='xx', fill=False)
    ax1.add_patch(rectangle)
    ax1.set_xlim([0,w_p])
    ax1.set_ylim([0,h_p])
    ax1.set_title( "Rat number: "+str(rat_num)+"\nSim number: "+str(nb_sim)+"\nSteps : " + str(tour) + "\nHits on the walls : " + str(nb_bruises) + "\neps : " + str(eps))

	#quiver subplot
    ax2 = fig1.add_subplot(1,2,2, aspect=1)
    for i in range(len(Divpool[0])):  
        ax2.quiver(Divpool[0][i], Divpool[1][i], Distance*np.cos(2*np.pi*a[i]/8), Distance*np.sin(2*np.pi*a[i]/8), angles='xy',scale_units='xy',scale=0.5, color='r', headlength=4, headaxislength=3, width=0.005 )
    circle = plt.Circle((x_c,y_c), radius=r_c, color='k', hatch='///', fill=False)
    ax2.add_patch(circle)
    rectangle = plt.Rectangle((x_r,y_r), w_r, h_r , color='k', hatch='xx', fill=False)
    ax2.add_patch(rectangle)
    ax2.set_xlim([0,w_p])
    ax2.set_ylim([0,h_p])
    
    if Save == True:
        plt.savefig(folder+"path_" + str(rat_num) + "_" + str(nb_sim)+ ".png")

    if Show == True:
        plt.show()
    
    plt.close()
    
    return



#=======================================================================================
# 								USING THE FUNCTIONS
#=======================================================================================
#To divide the pool into equal blocs (for the quiver plot)
Divpool = np.zeros((2,96),dtype=int)
i = 0
for xi in range(5,Pool[0],10):
   for yi in range(5,Pool[1],10):
       if dist2centre([xi,yi]) > Platform[2] and not inrectangle([xi,yi]):
           Divpool[0,i] = xi
           Divpool[1,i] = yi	
           i += 1


#50 animals, 100 simulations
[time_list,pathis,paths,Wis,Ws] = sim_several_animals(50, 50) # (nb_sim, nb_animal)

plt.figure()
plt.errorbar(np.arange(1,len(time_list[0])+1),np.mean(time_list,axis=0),yerr=np.std(time_list,axis=0))
plt.savefig(folder+"Performance_all_rats.png")

