import pandas as pd
import json

def read(file):
    return pd.read_csv(file)


######## formating json ########
def format_json(sum_file, qaps_file, op_file):

    sum = read("data/"+ sum_file)
    qaps = read("data/" + qaps_file)
    sum.sort_values(by=['document_id'])
    qaps.sort_values(by=['document_id'])
    qaps = qaps.to_numpy()
    op = {}
    data = []
    i = 0
    for index, content in sum.iterrows():
        temp_data = {}
        temp_data["title"] = content["document_id"]

        paragraphs = []

        temp_para = {}
        temp_para["context"] = content["summary"]

        qas = []

        match_flag = False
        for ques in range(i, len(qaps)):     
            if qaps[ques][0] == content["document_id"]:
                match_flag = True
                temp_qas = {}
                answer = []

                temp_answer = {}
                temp_answer["answer_start"] = 0
                temp_answer["text"] = qaps[ques][3]

                answer.append(temp_answer)
                temp_qas["answers"] = answer
                temp_qas["question"] =qaps[ques][2]
                temp_qas["id"] = content["document_id"]
                qas.append(temp_qas) 

            if match_flag and qaps[ques][0]!= content["document_id"]:
                break
            i+=1
        temp_para["qas"] = qas

        paragraphs.append(temp_para)

        temp_data["paragraphs"] = paragraphs
        data.append(temp_data)
        # uncomment this and replace 100 with number of samples required.
        # if len(data) >= 100:
        #     break

    op["data"] = data
    with open(op_file, 'w') as f:
        json.dump(op, f)
######## formating json done #######


format_json("train_sum.csv", "train_qaps.csv", 'data/json/train.json')
format_json("dev_sum.csv", "dev_qaps.csv", 'data/json/dev.json')
format_json("test_sum.csv", "test_qaps.csv", 'data/json/test.json')
