import numpy as np
import pandas as pd
import pickle
from itertools import product, combinations
from collections import OrderedDict as odict
import collections


def learn_outcome_space(data):
	outcomeSpace = {}
	for i in data.columns:
		outcomeSpace[i] = tuple(data[i].unique().tolist())
		#add '_t-1', for the belief updating in later
		outcomeSpace[i+'_t-1'] = outcomeSpace[i]
		
	return outcomeSpace

def prob(factor, *entry):
	"""
	argument 
	`factor`, a dictionary of domain and probability values,
	`entry`, a list of values, one for each variable in the same order as specified in the factor domain.
	
	Returns p(entry)
	"""

	return factor['table'][entry]     # insert your code here, 1 line    

def allEqualThisIndex(dict_of_arrays, **fixed_vars):
	"""
	Helper function to create a boolean index vector into a tabular data structure,
	such that we return True only for rows of the table where, e.g.
	column_a=fixed_vars['column_a'] and column_b=fixed_vars['column_b'].
	
	This is a simple task, but it's not *quite* obvious
	for various obscure technical reasons.
	
	It is perhaps best explained by an example.
	
	>>> all_equal_this_index(
	...    {'X': [1, 1, 0], Y: [1, 0, 1]},
	...    X=1,
	...    Y=1
	... )
	[True, False, False]
	"""
	# base index is a boolean vector, everywhere true
	first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
	index = np.ones_like(first_array, dtype=np.bool_)
	for var_name, var_val in fixed_vars.items():
		index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
	return index


def estProbTable_smooth(data, var_name, parent_names, outcomeSpace):
	"""
	Calculate a dictionary probability table by ML given
	`data`, a dictionary or dataframe of observations
	`var_name`, the column of the data to be used for the conditioned variable and
	`parent_names`, a tuple of columns to be used for the parents and
	`outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
	Return a dictionary containing an estimated conditional probability table.
	"""    
	var_outcomes = outcomeSpace[var_name]
	parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
	alpha = 1

	all_parent_combinations = product(*parent_outcomes)
	
	prob_table = odict()
	
	for i, parent_combination in enumerate(all_parent_combinations):
		parent_vars = dict(zip(parent_names, parent_combination))
		parent_index = allEqualThisIndex(data, **parent_vars)
		for var_outcome in var_outcomes:
			var_index = (np.asarray(data[var_name])==var_outcome)
			top = (var_index & parent_index).sum() + alpha
			bot = parent_index.sum()+alpha*len(var_outcomes)
			prob_table[tuple(list(parent_combination)+[var_outcome])] = top/bot

	return {'dom': tuple(list(parent_names)+[var_name]), 'table': prob_table}

def estTransitionTable_smooth(data, var_name, parent_names, outcomeSpace):
	"""
	Calculate a dictionary probability table by ML given
	`data`, a dictionary or dataframe of observations
	`var_name`, the column of the data to be used for the conditioned variable and
	`parent_names`, a tuple of columns to be used for the parents and
	`outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
	Return a dictionary containing an estimated conditional probability table.
	"""    
	var_outcomes = outcomeSpace[var_name]
	parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
	alpha = 1

	all_parent_combinations = product(*parent_outcomes)
	
	prob_table = odict()
	
	for i, parent_combination in enumerate(all_parent_combinations):
		parent_vars = dict(zip(parent_names, parent_combination))
		parent_index = allEqualThisIndex(data, **parent_vars)
		
		#last row has no future transition, so it can't be parent
		parent_index = parent_index[:-1]
		
		for var_outcome in var_outcomes:
			var_index = (np.asarray(data[var_name])==var_outcome)
			#first row has no parent so shift down by one step
			var_index = var_index[1:]
			top = (var_index & parent_index).sum() + alpha
			bot = parent_index.sum()+alpha*len(var_outcomes)
			prob_table[tuple(list(parent_combination)+[var_outcome])] = top/bot
			
	#correctly note down the '_t-1', for the belief updating in future
	parent_names = [i+'_t-1' for i in parent_names]
	return {'dom': tuple(list(parent_names)+[var_name]), 'table': prob_table}

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


def data_process():
	raw_data = pd.read_csv('data.csv')
	#processe data
	for i in range(1, 36):
		raw_data["r"+str(i)] = raw_data["r" + str(i)] > 0


	for i in range(4):
		raw_data['c'+str(i+1)] = raw_data['c'+str(i+1)]>0
		raw_data['reliable_sensor' + str(i+1)] = raw_data['reliable_sensor'+str(i+1)] == 'motion'
		raw_data['unreliable_sensor' + str(i+1)] = raw_data['unreliable_sensor'+str(i+1)] == 'motion'
		raw_data['door_sensor' + str(i+1)] = raw_data['door_sensor'+str(i+1)]>0


	raw_data['o1'] = raw_data['o1'] > 0
	raw_data['outside'] = raw_data['outside'] > 0

	#True : someone in room
	for i in range(1, 5):
		raw_data["door_sensor" + str(i)] = raw_data["door_sensor" + str(i)] > 0

	return raw_data


def learn_parameters(data, outcomeSpace, n_graph, sensor_room_map):
	transition_matrix={}
	for current, previous in n_graph.items():
		transition_matrix[current] = estTransitionTable_smooth(data, current, previous, outcomeSpace)

	emission_table_temp = {}
	for sensor, room in sensor_room_map.items():
		emission_table_temp[sensor] = estProbTable_smooth(data, sensor, room, outcomeSpace)

	# for i in ['door_sensor1', 'door_sensor2', 'door_sensor3', 'door_sensor4']:
	# 	emission_table_temp[i] = normalize(marginalize(emission_table_temp[i], emission_table_temp[i]['dom'][0], outcomeSpace))

	#index by room name
	emission_table={}
	for k in emission_table_temp.keys():
		emission_table[emission_table_temp[k]['dom'][0]]=emission_table_temp[k]

	return transition_matrix, emission_table

def update_map(emission_table):
	room_sensor_map = {}
	sensor_room_map = {}
	for i in emission_table:
		room = i
		sensor = emission_table[i]['dom'][1]

		room_sensor_map[room] = sensor
		sensor_room_map[sensor] = room
	return room_sensor_map, sensor_room_map


def output_parameters(transition_matrix, emission_table, outcomeSpace, room_sensor_map, sensor_room_map):
	pickle_out = open("outcomeSpace.pickle","wb")
	pickle.dump(outcomeSpace, pickle_out)
	pickle_out.close()

	pickle_tran = open("transition_matrix.pickle", "wb")
	pickle.dump(transition_matrix, pickle_tran)
	pickle_tran.close()

	pickle_emi = open("emission_table.pickle", "wb")
	pickle.dump(emission_table, pickle_emi)
	pickle_emi.close()

	pickle_sensor_room = open("sensor_room_map.pickle","wb")
	sensor_room_map = pickle.dump(sensor_room_map, pickle_sensor_room)
	pickle_sensor_room.close()

	pickle_room_sensor = open("room_sensor_map.pickle","wb")
	room_sensor_map = pickle.dump(room_sensor_map, pickle_room_sensor)
	pickle_room_sensor.close()

	return

def train():
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

	sensor_room_map = {
		'unreliable_sensor1': ['o1'],
		'unreliable_sensor2': ['c3'],
		'unreliable_sensor3': ['r1'],
		'unreliable_sensor4': ['r24'],
		'reliable_sensor1': ['r16'],
		'reliable_sensor2': ['r5'],
		'reliable_sensor3': ['r25'],
		'reliable_sensor4': ['r31'],
		# 'door_sensor1': ['r8', 'r9'],
		# 'door_sensor2': ['c1', 'c2'],
		# 'door_sensor3': ['r26', 'r27'],
		# 'door_sensor4': ['r35', 'c4'],
	}

	data = data_process()
	outcomeSpace = learn_outcome_space(data)
	transition_matrix, emission_table = learn_parameters(data, outcomeSpace, n_graph, sensor_room_map)

	room_sensor_map, sensor_room_map = update_map(emission_table)

	output_parameters(transition_matrix, emission_table, outcomeSpace, room_sensor_map, sensor_room_map)

	return


train()