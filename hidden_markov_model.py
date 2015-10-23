import numpy as np
from collections import defaultdict, Counter

def learn_transition_matrix(training_sequences, speaker_sequences, number_of_states = 43, order = 1):
	# Learns the first order transistion matrix from a training corpus of sequences of labels
	# assumes that the labels have been transformed to integers from 0 to number_of_states-1
	start_prob = np.zeros(number_of_states)
	X_1 = defaultdict(Counter)
	for sequence, speakers in zip(training_sequences, speaker_sequences):
		assert len(sequence) == len(speakers)
		from_tag = (None,) * order
		from_tag = from_tag[1:] + (sequence[0],)
		# utterers = (None,) * order
		# utterers = utterers[1:] + (speakers[0],)
		start_prob[from_tag] += 1
		for i in range(1,len(sequence)):
			to_tag = sequence[i]
			# speaker_tuple = tuple([speakers[i] == utt for utt in utterers])
			# X_1[(from_tag, speaker_tuple)][to_tag] += 1
			X_1[from_tag][to_tag] += 1
			from_tag = from_tag[1:] + (to_tag,)
			# utterers = utterers[1:] + (speakers[i],)
	# normalize to get actual probabilities
	start_prob = start_prob/sum(start_prob)
	for counter in X_1.itervalues():
		total = sum(counter.values())
		for key in counter.iterkeys():
			counter[key] /= float(total)
	return start_prob, X_1

def viterbi_decoder(sequence,speakers, start_prob, transition_matrix, emmision_probs, order = 1):
	assert len(sequence) == len(speakers)
	number_of_states = len(start_prob)
	

	T_1 = np.zeros((number_of_states, len(sequence)))
	T_2 = np.zeros((number_of_states, len(sequence)))

	T_1[:,0] = np.multiply(emmision_probs[:,0], start_prob)

	# utterers = (None,) * order
	# utterers = utterers[1:] + (speakers[0],)
	n_Nones = order -1
	for i in range(1,len(sequence)):
		for j in range(number_of_states):
			# speaker_tuple = tuple([speakers[i] == utt for utt in utterers])
			state_lists = [state for state in transition_matrix.iterkeys() if state[:n_Nones] == (None,)*n_Nones]# and state[1] == speaker_tuple]

			maxk = None
			maxval = None
			for state in state_lists:
				val = T_1[state[-1], i-1] *transition_matrix[state][j] *emmision_probs[j,i]
				if val > maxval:
					maxval = val
					maxk = state[-1]
			T_1[j,i] = maxval
			T_2[j,i] = maxk
		# utterers = utterers[1:] + (speakers[i],)
		if n_Nones > 0:
			n_Nones -= 1
	most_likely_hidden = np.zeros(len(sequence))
	most_likely_hidden[-1] = np.argmax(T_1[:,-1])
	for i in range(1,len(sequence)):
		most_likely_hidden[-i-1] = T_2[most_likely_hidden[-i],-i]

	return most_likely_hidden

def evaluate_sequence(predicted, true):
	assert len(predicted) == len(true)
	return sum(np.array([int(predicted[i]) == int(true[i]) for i in range(len(true))]))

if __name__ == "__main__":
	training_sequences = [[0,1,2,1,1,0],[1,2,0,1]]
	speaker_sequences =  [["a","a","b","a","b","b"],["b","a","b","b"]]
	start_prob, transition_matrix = learn_transition_matrix(training_sequences, speaker_sequences,number_of_states = 3, order = 2)
	print viterbi_decoder(["Hallo", "dit", "is" , "een", "test"],["a","a","b","a","a"], start_prob, transition_matrix, None, order = 2)