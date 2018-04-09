import json
import question_generator as qg
import os

def qg_wraper(filename,outputpath):

    args = qg.parseArgs()
    with open(filename,'r') as fr:
        data = json.load(fr)

    qg_data = dict()
    keys = data.keys()
    count = 0
    for key in keys:
        print(count)
        count += 1
        sents = data[key]["sentences"]
        times = data[key]["timestamps"]
        qa_list = list()
        for i in range(len(sents)):
            sent = sents[i]
            time = times[i]
            questions = qg.runSentence(parserFolder=args.parser_path, sentence=sent)
            for j in range(len(questions)):
                questions[j].append(time)
            qa_list.extend(questions)
        qg_data[key] = qa_list

    print(qg_data)

    with open(outputpath,'w') as fw:
        json.dump(qg_data,fw)

qg_wraper('captions/val_2.json','test.json')
qg_wraper('captions/val_1.json','test2.json')
qg_wraper('captions/train.json','test3.json')