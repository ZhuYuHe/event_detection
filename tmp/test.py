from bin.evaluate import computeF
import codecs

def computePR(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] == 1:
            correct += 1
    P = correct / y_pred.count(1)
    R = correct / y_true.count(1)
    F = computeF(P,R)
    return P,R,F

def main():
    y_true = []
    y_pred = []
    with codecs.open('tmp/eval_y_self_attention.txt', 'r', 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '__label__非事件':
                y_pred.append(0)
            else:
                y_pred.append(1)
    with codecs.open('tmp/true_label.txt', 'r', 'utf8') as f:
        for line in f:
            line = line.strip()
            y_true.append(int(line))
    P, R, F = computePR(y_true, y_pred)
    print('Precision: {}'.format(P))
    print('Recall: {}'.format(R))
    print('F: {}'.format(F))

if __name__ == '__main__':
    main()
