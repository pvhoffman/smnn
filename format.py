import csv

def main():
    with open('training.csv', 'r') as inf:
        with open('training-x.out', 'w') as oufx:
            oufy = open('training-y.out', 'w')
            # skip the header
            inf.readline()

            for line in inf:
                vs = line.split(",")
                if len(vs) == 33:
                    w = ", ".join(vs[1:31])
                    w = w + "\n"
                    x = "0\n"
                    t = vs[32]
                    if(t.startswith('s')):
                        x = "1\n"
                    oufx.write(w);
                    oufy.write(x)



if __name__ == '__main__':
    main()

