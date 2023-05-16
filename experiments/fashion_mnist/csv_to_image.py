import sys

to_write = ""

with open(sys.argv[1], "r") as fp, open(sys.argv[2], "w") as op:
    for i, each in enumerate(fp):
        # Skip header
        if i == 0:
            continue
        sepped = each.strip().split(",")  # Split into label and image pixels
        to_write += "[" + ", ".join(sepped[1:]) + "]\t" + sepped[0] + "\n"  # Make into array of image \t label format
    op.write(to_write)
