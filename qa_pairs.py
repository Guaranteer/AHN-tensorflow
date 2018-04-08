import json

def answer_set(filename_list):

    ans_set = dict()

    for filename in filename_list:
        with open(filename,'r') as fr:
            data = json.load(fr)

        for key,items in data.items():
            for i,item in enumerate(items):
                print(item)
                ty = item[0]
                ans = item[2]
                if ty in ans_set:
                    if ans not in ans_set[ty]:
                        ans_set[ty].append(ans)
                else:
                    ans_set[ty] = [ans]
    print(ans_set)
    with open('answer.json','w') as fw:
        json.dump(ans_set,fw)

def clear(filename_list):
    clear_data = dict()

    for filename in filename_list:
        with open(filename,'r') as fr:
            data = json.load(fr)

        for key,items in data.items():
            if len(items) > 0:
                clear_data[key] = items
    with open('question.json','w') as fw:
        json.dump(clear_data,fw)

def count():
    with open('answer.json','r') as f1:
        data1 = json.load(f1)
    with open('question.json','r') as f2:
        data2 = json.load(f2)

    type = dict()

    c = 0
    for key,items in data1.items():
        c += len(items)
    print(c)

    c = 0
    for key,items in data2.items():
        c += len(items)
        for item in items:
            if item[0] in type:
                type[item[0]] += 1
            else:
                type[item[0]] = 1
    print(c)
    print(type)

def count2():
    with open('captions/train.json','r') as f:
        data = json.load(f)

    max = 0
    s = 0
    for key,items in data.items():

        c = len(items["timestamps"])
        s += c
        if c > max:
            print(c,key,items)
            max = c
    print(max)
    print(s/len(data.keys()))


def mulnum():
    a = [0.4214,	0.8643,	0.5743,	0.2462,	0.2847
]
    print(a[0]*0.338 +a[1] *0.07+a[2]*0.172 +a[3]*0.278 +a[4]*0.141)
file_list = ['qg/val_2.json','qg/val_1.json','qg/train.json','qg/val_2_static.json','qg/val_1_static.json','qg/train_static.json']
# answer_set(file_list)
mulnum()


