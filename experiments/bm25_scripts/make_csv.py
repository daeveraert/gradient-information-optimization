import csv
import sys

with open(sys.argv[1], "r") as fp, open(sys.argv[2], "w", newline='') as op:
    csv_fp = fp
    csv_op = csv.writer(op, delimiter=',')
    csv_op.writerow(["text","id"])  # Give column names
    total = 0
    for i, each in enumerate(csv_fp):
        csv_op.writerow([each.strip()] + [i])  # Write value and ID in CSV format
        total += 1
