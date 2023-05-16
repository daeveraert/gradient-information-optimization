import sys

dct = {}
bad_data_counter = 0
with open(sys.argv[1], "r") as fp:
    for each in fp:
        sepped = each.strip().split("\t")
        # Expect length of 2
        if len(sepped) != 2:
            bad_data_counter += 1
        else:
            dct[sepped[0].strip()] = sepped[1]  # Create dictionary of input-output pairs to later look up and match

print("Weird parses: " + str(bad_data_counter))

lines = []
not_in_dct = 0
with open(sys.argv[2], "r") as fp:
    for each in fp:
        cleaned = each.strip()
        # Expect all data to be in the dictionary
        if cleaned not in dct:
            not_in_dct += 1
        else:
            lines += [cleaned + '\t' + dct[cleaned] + "\n"]  # Write input \t output

print("Not in Dict: " + str(not_in_dct))

with open(sys.argv[3], "w") as op:
    op.writelines(lines)


