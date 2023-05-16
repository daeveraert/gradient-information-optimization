import sys
import csv

bad_data_counter = 0
written = 0

with open(sys.argv[1], "r", newline='') as fp, open(sys.argv[2], "w") as op:
    csv_fp = csv.reader(fp, delimiter=',')
    for i, each in enumerate(csv_fp):
        # Skip header
        if i == 0:
            continue

        # Expect length of 3; determine how many are bad data
        if len(each) < 3:
            bad_data_counter += 1

        # Filter if the topK ID is above the specified filter value
        if int(each[2]) > int(sys.argv[3]):
            continue
        else:
            written += 1
            op.write(each[0].strip() + "\n")

print("Bad Data: " + str(bad_data_counter))
print("Total Written: " + str(written))
