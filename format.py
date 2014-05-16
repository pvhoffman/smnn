import csv

def format_test():
    with open('test.csv', 'r') as inf:
        with open('test-x.out', 'w') as oufx:
            # skip the header
            inf.readline()
            for line in inf:
                vs = line.split(",")
                if len(vs) == 31:
                    w = ", ".join(vs[1:])
                    oufx.write(w);


def format_train():
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


def main():
    format_train()
    format_test()

if __name__ == '__main__':
    main()

