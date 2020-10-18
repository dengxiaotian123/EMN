#coding=utf-8
def ComputeR10_1(scores,labels,count = 10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:#如果最大的就是第一个那就判断正确
                correct = correct + 1
    score=float(correct)/ total
    print('R10_1',score )


def ComputeR2_1(scores,labels,count = 2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    score=float(correct) / total
    print('R2_1',score )

def ComputeR10_2(scores,labels,hit=2,count = 10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            pos_score=scores[i]
            total = total+1
            sublist = scores[i:i+count]
            curr = sorted(sublist, reverse=True) #from large to small

            if curr[hit-1] <= pos_score:
                correct = correct + 1
    score=float(correct) / total
    print('R10_2',score )

def ComputeR10_5(scores,labels,hit=5,count = 10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            pos_score=scores[i]
            total = total+1
            sublist = scores[i:i+count]
            curr = sorted(sublist, reverse=True) #from large to small

            if curr[hit-1] <= pos_score:
                correct = correct + 1
    score=float(correct) / total
    print('R10_5',score )
    return score
