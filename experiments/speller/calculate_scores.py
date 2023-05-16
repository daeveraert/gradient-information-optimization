import sys

# Preprocess hypotheses and split into words
with open(sys.argv[1], 'r') as fp:
	hypotheses = [each.strip().split(" ") for each in fp.readlines()]

# Preprocess targets and split into words
with open(sys.argv[2], 'r') as fp:
	targets = [each.strip().split(" ") for each in fp.readlines()]

# Preprocess sources and split into words
with open(sys.argv[3], 'r') as fp:
	sources = [each.strip().split(" ") for each in fp.readlines()]

acc_numerator = 0
acc_denominator = 0
corr_numerator = 0
corr_denominator = 0
len_uneven = 0
for i in range(len(hypotheses)):
	hypothesis = hypotheses[i]
	target = targets[i]
	source = sources[i]

	# Skip mismatches
	if len(hypothesis) != len(target):
		len_uneven += 1
		continue

	# Compute word-level correction rate
	for x in range(len(target)):
		if len(target) == len(source) and target[x] != source[x]:
			corr_denominator += 1
			if source[x] != hypothesis[x]:
				corr_numerator += 1

	# Compute word-level accuracy
	for x in range(len(target)):
		if target[x] == hypothesis[x]:
			acc_numerator += 1
		acc_denominator += 1

print("Mismatch: " + str(len_uneven))
print("Word-Level Accuracy: " + str(acc_numerator / acc_denominator))
print("Correction Rate: " + str(corr_numerator / corr_denominator))


