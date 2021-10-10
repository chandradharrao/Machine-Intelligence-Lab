censor = ("pprint","print")
import sys
with open("tmp.py","w") as tmp:
    with open(f"{sys.argv[1]}.py","r") as f:
        for line in f:
            if any(s in line for s in censor):
                continue
            else:
                tmp.write(line)
        f.close()
    tmp.close()
print("Done!")