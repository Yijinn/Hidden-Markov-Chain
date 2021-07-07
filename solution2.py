from __future__ import division
from __future__ import print_function

# Allowed libraries 
import numpy as np
import pandas as pd
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
# from graphviz import Digraph, Graph
# from tabulate import tabulate
import copy
import sys
import os
import datetime
import pickle

def prob(factor, *entry):
	"""
	argument 
	`factor`, a dictionary of domain and probability values,
	`entry`, a list of values, one for each variable in the same order as specified in the factor domain.
	
	Returns p(entry)
	"""

	return factor['table'][entry]     # insert your code here, 1 line    


def join(f1, f2, outcomeSpace):
	"""
	argument 
	`f1`, first factor to be joined.
	`f2`, second factor to be joined.
	`outcomeSpace`, dictionary with the domain of each variable
	
	Returns a new factor with a join of f1 and f2
	"""
	
	# First, we need to determine the domain of the new factor. It will be union of the domain in f1 and f2
	# But it is important to eliminate the repetitions
	common_vars = list(f1['dom']) + list(set(f2['dom']) - set(f1['dom']))
	
	# We will build a table from scratch, starting with an empty list. Later on, we will transform the list into a odict
	table = list()
	
	# Here is where the magic happens. The product iterator will generate all combinations of varible values 
	# as specified in outcomeSpace. Therefore, it will naturally respect observed values
	for entries in product(*[outcomeSpace[node] for node in common_vars]):
		
		# We need to map the entries to the domain of the factors f1 and f2
		entryDict = dict(zip(common_vars, entries))
		f1_entry = (entryDict[var] for var in f1['dom'])
		f2_entry = (entryDict[var] for var in f2['dom'])
		
		# Insert your code here
		p1 = prob(f1, *f1_entry)           # Use the fuction prob to calculate the probability in factor f1 for entry f1_entry 
		p2 = prob(f2, *f2_entry)           # Use the fuction prob to calculate the probability in factor f2 for entry f2_entry 
		
		# Create a new table entry with the multiplication of p1 and p2
		table.append((entries, p1 * p2))
	return {'dom': tuple(common_vars), 'table': odict(table)}


def evidence(var, e, outcomeSpace):
	"""
	argument 
	`var`, a valid variable identifier.
	`e`, the observed value for var.
	`outcomeSpace`, dictionary with the domain of each variable
	
	Returns dictionary with a copy of outcomeSpace with var = e
	"""    
	newOutcomeSpace = outcomeSpace.copy()      # Make a copy of outcomeSpace with a copy to method copy(). 1 line
	newOutcomeSpace[var] = (e,)                # Replace the domain of variable var with a tuple with a single element e. 1 line
	return newOutcomeSpace


def marginalize(f, var, outcomeSpace):
	"""
	argument 
	`f`, factor to be marginalized.
	`var`, variable to be summed out.
	`outcomeSpace`, dictionary with the domain of each variable
	
	Returns a new factor f' with dom(f') = dom(f) - {var}
	"""    
	
	# Let's make a copy of f domain and convert it to a list. We need a list to be able to modify its elements
	new_dom = list(f['dom'])

	
	new_dom.remove(var)            # Remove var from the list new_dom by calling the method remove(). 1 line
	table = list()                 # Create an empty list for table. We will fill in table from scratch. 1 line
	for entries in product(*[outcomeSpace[node] for node in new_dom]):
		s = 0;                     # Initialize the summation variable s. 1 line

		# We need to iterate over all possible outcomes of the variable var
		for val in outcomeSpace[var]:
			# To modify the tuple entries, we will need to convert it to a list
			entriesList = list(entries)
			# We need to insert the value of var in the right position in entriesList
			entriesList.insert(f['dom'].index(var), val)
					  
			p = prob(f, *tuple(entriesList))     # Calculate the probability of factor f for entriesList. 1 line
			s = s + p                            # Sum over all values of var by accumulating the sum in s. 1 line
			
		# Create a new table entry with the multiplication of p1 and p2
		table.append((entries, s))
	return {'dom': tuple(new_dom), 'table': odict(table)}

def normalize(f):
	"""
	argument 
	`f`, factor to be normalized.
	
	Returns a new factor f' as a copy of f with entries that sum up to 1
	""" 
	table = list()
	sum = 0
	for k, p in f['table'].items():
		sum = sum + p
	for k, p in f['table'].items():
		table.append((k, p/sum))
	return {'dom': f['dom'], 'table': odict(table)}

#function from tutorial to print a factor
def printFactor(f):
	"""
	argument
	`f`, a factor to print on screen
	"""
	# Create a empty list that we will fill in with the probability table entries
	table = list()

	# Iterate over all keys and probability values in the table
	for key, item in f['table'].items():
		# Convert the tuple to a list to be able to manipulate it
		k = list(key)
		# Append the probability value to the list with key values
		k.append(item)
		# Append an entire row to the table
		table.append(k)
	# dom is used as table header. We need it converted to list
	dom = list(f['dom'])
	# Append a 'Pr' to indicate the probabity column
	dom.append('Pr')
	print(tabulate(table,headers=dom,tablefmt='orgtbl'))
	
	
def miniForwardOnline(f, transition, outcomeSpace):
	"""
	argument 
	`f`, factor that represents the previous state of the chain.
	`transition`, transition probabilities from time t-1 to t.
	`outcomeSpace`, dictionary with the domain of each variable.
	
	Returns a new factor that represents the current state of the chain.
	"""

	# Make a copy of f so we will not modify the original factor
	fPrevious = f.copy()

	#Set the f_previous domain to be a list with a single variable name appended with '_t-1' to indicate previous time step
	for i in fPrevious.keys():
		fPrevious[i]['dom']=(i+'_t-1',)

	#extract the first factor and then join all factors together
	count=0
	for i in fPrevious.keys():
		if count==0:
			p=fPrevious[i]
			count +=1
		else:
			p=join(p,fPrevious[i],outcomeSpace)

	#Make the join operation between fPrevious and the transition probability table
	fCurrent=join(p,transition,outcomeSpace)

	# Marginalize Variable_t-1
	for i in fPrevious.keys():
		fCurrent=marginalize(fCurrent,i+'_t-1',outcomeSpace)

	#normalization
	fCurrent=normalize(fCurrent)

	return fCurrent

def forwardOnline(f, transition, emission, stateVar, emissionVar, emissionEvi, outcomeSpace):
	"""
	argument 
	`f`, factor that represents the previous state.
	`transition`, transition probabilities from time t-1 to t.
	`emission`, emission probabilities.
	`stateVar`, state (hidden) variable.
	`emissionVar`, emission variable.
	`emissionEvi`, emission observed evidence. If undef, we do only the time update
	`outcomeSpace`, dictionary with the domain of each variable.
	
	Returns a new factor that represents the current state.
	"""

	# Set fCurrent as a copy of f
	fCurrent = f.copy()

	# If emissionEvi == None, we will assume this time step has no observed evidence    
	if emissionEvi != None:
		# Set evidence in the form emissionVar = emissionEvi
		newOutcomeSpace = evidence(emissionVar, emissionEvi, outcomeSpace)
		# Make the join operation between fCurrent and the emission probability table. Use the newOutcomeSpace
		fCurrent = join(fCurrent, emission, newOutcomeSpace)
		# Marginalize emissionVar. Use the newOutcomeSpace
		fCurrent = marginalize(fCurrent, emissionVar, newOutcomeSpace) 
		# Normalize fCurrent, optional step
		fCurrent = normalize(fCurrent)

	return fCurrent

#neighbours graph
n_graph = {
	'r1' : ['r1', 'r2', 'r3'],
	'r2' : ['r2', 'r1', 'r4'],
	'r3' : ['r3', 'r1', 'r7'],
	'r4' : ['r4', 'r2', 'r8'],
	'r5' : ['r5', 'r9', 'r6', 'c3'],
	'r6' : ['r6', 'r5', 'c3'],
	'r7' : ['r7', 'r3', 'c1'],
	'r8' : ['r8', 'r4', 'r9'],
	'r9' : ['r9', 'r8', 'r13'],
	'r10': ['r10', 'c3'],
	'r11': ['r11', 'c3'],
	'r12': ['r12', 'r22', 'outside'],
	'r13': ['r13', 'r9', 'r24'],
	'r14': ['r14', 'r24'],
	'r15': ['r15', 'c3'],
	'r16': ['r16', 'c3'],
	'r17': ['r17', 'c3'],
	'r18': ['r18', 'c3'],
	'r19': ['r19', 'c3'],
	'r20': ['r20', 'c3'],
	'r21': ['r21', 'c3'],
	'r22': ['r22', 'r12', 'r25'],
	'r23': ['r23', 'r24'],
	'r24': ['r24', 'r14', 'r23'],
	'r25': ['r25', 'r22', 'r26', 'c1'],
	'r26': ['r22', 'r27', 'r32', 'c1'],
	'r27': ['r27', 'r26', 'r32'],
	'r28': ['r28', 'c4'],
	'r29': ['r29', 'r30', 'c4'],
	'r30': ['r29', 'r30'],
	'r31': ['r31', 'r32'],
	'r32': ['r31', 'r32', 'r33'],
	'r33': ['r33', 'r32'],
	'r34': ['c2', 'r34'],
	'r35': ['r35', 'c4'],
	'c1' : ['r7', 'c2'],
	'c2' : ['c2', 'r34', 'c1', 'c4'],
	'c3' : ['c3', 'r6'],
	'c4' : ['c4', 'r29', 'r28', 'r35'],
	'o1' : ['o1', 'c4', 'c3'],
	'outside' : ['r12', 'outside'],
}



def load_parameters():
	try:
		pickle_tran = open("transition_matrix.pickle","rb")
		transition_matrix = pickle.load(pickle_tran)
		pickle_tran.close()

		pickle_emi = open("emission_table.pickle","rb")
		emission_table = pickle.load(pickle_emi)
		pickle_emi.close()

		pickle_outcome = open("outcomeSpace.pickle","rb")
		outcomeSpace = pickle.load(pickle_outcome)
		pickle_outcome.close()

		pickle_sensor_room = open("sensor_room_map.pickle","rb")
		sensor_room_map = pickle.load(pickle_sensor_room)
		pickle_sensor_room.close()

		pickle_room_sensor = open("room_sensor_map.pickle","rb")
		room_sensor_map = pickle.load(pickle_room_sensor)
		pickle_room_sensor.close()
		
	except:
		raise("loading parameters error")
		sys.exit()

	return transition_matrix, emission_table, outcomeSpace, room_sensor_map, sensor_room_map


#########################################################################################

#load parameters
transition_matrix, emission_table, outcomeSpace, room_sensor_map, sensor_room_map = load_parameters()

#initialize state
#All rooms are empty except 'outside' area
state = {}
for i in list(n_graph.keys()):
	state[i]={'dom':(i,),'table':odict([((True,),0.01) ,((False,),0.99),])}
state['outside']={'dom':('outside',),'table':odict([((True,),0.99) ,((False,),0.01),])}

previous_state=state.copy()

actions_dict = {'lights1': 'off', 'lights2': 'off', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off', 'lights6': 'off', 'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off', 'lights11': 'off', 'lights12': 'off', 'lights13': 'off', 'lights14': 'off', 'lights15': 'off', 'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 'lights19': 'off', 'lights20': 'off', 'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 'lights25': 'off', 'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off', 'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35':'off'}


def get_action(sensor_data):
	global actions_dict
	global state
	global outcomeSpace
	global n_graph
	global previous_state
	global room_sensor_map
	global sensor_room_map
	global transition_matrix
	global emission_table

	#sensor_data normalization
	for k,v in sensor_data.items():
		if k in sensor_room_map and v:
			if isinstance(v, int):
				sensor_data[k] = True if v > 0 else False
			else:
				sensor_data[k] = True if v == 'motion' else False


	#update each room state
	for room, neighbours in n_graph.items():
		#select the neighbours' states
		neighbour_state = {}
		for neighbour in neighbours:
			neighbour_state[neighbour] = previous_state[neighbour]


		if room in room_sensor_map.keys():
			#B(X_t) = P(x_t|x_t-1) * P(x_t-1) 
			f = miniForwardOnline(neighbour_state, transition_matrix[room], outcomeSpace)
			#B(X_t+1) = P(X_t+1|e_1:t+1)
			state[room]=forwardOnline(f, transition_matrix[room],emission_table[room],room,room_sensor_map[room],sensor_data[room_sensor_map[room]],outcomeSpace)
		else:
			#B(X_t) = P(x_t|x_t-1) * P(x_t-1) 
			state[room]=miniForwardOnline(neighbour_state, transition_matrix[room], outcomeSpace)

		if room[0] == 'r':
			# probability is normalized, therefore simply consider the comparison with 0.5 will get the mpe
			if prob(state[room], True) >0.5:
				actions_dict['lights'+room[1:]] = 'on'
			else:
				actions_dict['lights'+room[1:]] = 'off'

	for i in range(2):
		robot = sensor_data['robot'+str(i+1)]
		#handle None
		if not robot:
			continue

		#make the format of robot the same as spec
		robot = (robot.split(',')[0][2:-1], int(robot.split(',')[1][:-1]))
		
		number = robot[1]
		room = robot[0]
		
		if number > 0:
			state[room]['table'] = odict([((True,), 0.99) ,((False,), 0.01),])
			if room[0] == 'r':
				actions_dict['lights'+room[1:]] = 'on'
		else:
			state[room]['table'] = odict([((False,), 0.99) ,((True,), 0.01),])
			if room[0] == 'r':
				actions_dict['lights'+room[1:]] = 'off'
		

	previous_state=state.copy()

	return actions_dict