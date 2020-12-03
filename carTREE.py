import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
import sklearn
from collections import deque
import random
import sklearn.ensemble
import sklearn.tree
from math import sqrt
from sklearn.exceptions import NotFittedError
import sklearn.multioutput

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

MAP_POINTS = [(0,100), (50,300), (10,400),(60,500), (200,560),
              (400,500),(500,400),(600,350),(700,325),
              (800,315),(900,800),(1100,800),(1000,600),
              (950,500),(975,400),(850,150),(500,325),
              (350,300),(120,350),(100,0)
             ]

#helpers
def slope(p1, p2) :
    return (p2[1] - p1[1]) * 1. / ((p2[0] - p1[0])+1e-5)
   
def y_intercept(slope, p1) :
    return p1[1] - 1. * slope * p1[0]
   
def intersecta(line1, line2) :
    min_allowed = 1e-5
    big_value = 1e10
    m1 = slope(line1[0], line1[1])
    b1 = y_intercept(m1, line1[0])
    m2 = slope(line2[0], line2[1])
    b2 = y_intercept(m2, line2[0])
    if abs(m1 - m2) < min_allowed :
        x = big_value
    else :
        x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    y2 = m2 * x + b2
    return (int(x),int(y))

def segment_intersect(line1, line2) :
    intersection_pt = intersecta(line1, line2)
    
    if (line1[0][0] < line1[1][0]) :
        if intersection_pt[0] < line1[0][0] or intersection_pt[0] > line1[1][0] :
            return None
    else :
        if intersection_pt[0] > line1[0][0] or intersection_pt[0] < line1[1][0] :
            return None
         
    if (line2[0][0] < line2[1][0]) :
        if intersection_pt[0] < line2[0][0] or intersection_pt[0] > line2[1][0] :
            return None
    else :
        if intersection_pt[0] > line2[0][0] or intersection_pt[0] < line2[1][0] :
            return None

    return intersection_pt

class Car():
    def __init__(self):
        self.center_pos = pygame.math.Vector2(50, 50)
        self.corners = [
            pygame.math.Vector2(-10,-20),
            pygame.math.Vector2(10,-20),
            pygame.math.Vector2(10,20),
            pygame.math.Vector2(-10,20)
        ]
        self.corners_pos = [x+self.center_pos for x in self.corners]
        self.sensors = [
            pygame.math.Vector2(0,1000),
            pygame.math.Vector2(966,258),
            pygame.math.Vector2(-966,258)
        ]
        self.sensors_pos = [x+self.center_pos for x in self.sensors]
        self.velocity = pygame.math.Vector2(0,10)
        self.angle = 0
        self.intersects = []
        self.signals = [0, 0, 0]
        
    def update(self, action):
       ##### next_state, reward, done
        angleChange = 10
        if action == 0: 
            angleChange = -10
        elif action == 1:
            angleChange = 0
        
        self.angle += angleChange
        self.angle = self.angle % 360
        self.center_pos = self.center_pos + self.velocity.rotate(self.angle)
        self.corners_pos = [x.rotate(self.angle)+self.center_pos for x in self.corners]
        self.sensors_pos = [x.rotate(self.angle)+self.center_pos for x in self.sensors]
        self.intersects = []
        self.signals = []
        for sensor in self.sensors_pos:
            sensor_intersects = []
            for i in range(len(MAP_POINTS)-1):
                intersect = segment_intersect([MAP_POINTS[i],MAP_POINTS[i+1]],[(self.center_pos.x,self.center_pos.y),(sensor.x,sensor.y)])
                if(intersect != None):
                    sensor_intersects.append(intersect)        
            if len(sensor_intersects) == 0:
                self.intersects.append((-100,-100))
                self.signals.append(1000)
            else:
                sensor_distances = [sqrt((x[0] - self.center_pos.x)**2 +(x[1] - self.center_pos.y)**2) for x in sensor_intersects]
                self.intersects.append(sensor_intersects[sensor_distances.index(min(sensor_distances))])
                self.signals.append(min(sensor_distances))
        
        reset = False
        #check car colision
        reward = 1
        for signal in self.signals:
            if signal < 15:
                self.reset()
                reset = True
                reward = -1
                
        next_state = []        
        for signal in self.signals:
            next_state.append(signal)
        next_state.append(self.angle)
        
        return next_state, reward, reset
                
    #reset car position
    def reset(self):
        self.__init__()

################################
# Trees Q-learning implementation #
################################
EXPLORATION_MIN = 1
EXPLORATION_DECAY = 0.96
BATCH_SIZE = 32
GAMMA = 0.95

exploration_rate = 1
memory = deque(maxlen=1000)

forest = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100,random_state=0)
classifier = sklearn.multioutput.MultiOutputRegressor(forest)

def select_action(state):
    global exploration_rate
    
    if np.random.rand() < exploration_rate:
        action = random.randint(0,2)
        return action
    try:
        q_values = classifier.predict(state)
        return np.argmax(q_values)
    except NotFittedError as e:
        return 0

def remember(last_state,action,reward,next_state,done):
    global memory
    memory.append((last_state,action,reward,next_state,done))

def experience_replay():
    global memory,EXPLORATION_MIN,EXPLORATION_DECAY,BATCH_SIZE,GAMMA,exploration_rate

    if len(memory) < BATCH_SIZE:
        return

#     batch = random.sample(memory,BATCH_SIZE)
    batch = memory
    
    X = np.empty((0,4))
    y = np.empty((0,3))
    for last_state, action, reward, next_state, done in batch:
        q_update = reward
        if not done:
            try:
                q_update = (reward + GAMMA * np.amax(classifier.predict(np.array(next_state, dtype=np.float32).reshape(1, -1))))
            except NotFittedError as e:
                q_update = reward 
        try:
            #print(classifier.predict(np.array(last_state, dtype=np.float32)))
            q_values = classifier.predict(np.array(last_state, dtype=np.float32).reshape(1, -1))
        except NotFittedError as e:
            q_values = np.zeros(3, dtype=np.float32).reshape(1, -1)
        print(q_values)
        q_values[0][action] = q_update
        
        X = np.append(X,np.array([last_state]),axis=0)
        y = np.append(y,np.array([q_values[0]]),axis=0)
    
    #fit
    classifier.fit(X,y)
    
    if exploration_rate > EXPLORATION_MIN:
        exploration_rate *= EXPLORATION_DECAY

##training

pygame.init()
 
screen = pygame.display.set_mode([1000, 700])
 
pygame.display.set_caption('Comp 432 custom car env')

car = Car()
last_state = [1000, 1000, 1000, 0]
clock = pygame.time.Clock()
done = False
steps = 0

while not done:
 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
    steps += 1
    action = select_action(last_state)
    next_state, reward, done1 = car.update(action)
    remember(last_state,action,reward,next_state,done1)
    experience_replay()
        
    #total_reward += reward
    last_state = next_state
    
    # -- Draw everything
    # Clear screen
    screen.fill(BLACK)
    
    #draw map
    pygame.draw.lines(screen,WHITE,False,MAP_POINTS)
    
    #draw goal
    pygame.draw.circle(screen,GREEN,(950,650),20)
    
    #draw car
    pygame.draw.lines(screen,BLUE,True,[(vect.x,vect.y) for vect in car.corners_pos])
    
    
    #draw sensors
    for sensor in car.sensors_pos:
        pygame.draw.line(screen,GREEN,(car.center_pos.x,car.center_pos.y),(sensor.x,sensor.y))   
    
    for intersect in car.intersects:
        pygame.draw.circle(screen,RED,intersect,5)

    pygame.display.flip()
    clock.tick(50)
    
pygame.quit()

