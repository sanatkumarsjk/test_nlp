import pandas as pd

def read(file):
    return pd.read_csv(file)

def write(file, data):
    data.to_csv("data/"+file,index=False)

def split_data():
    sum = read("data/summaries.csv")
    qaps = read("data/qaps.csv")

    train_sum = sum[sum['set'] == 'train']
    write('train_sum.csv', train_sum)

    dev_sum = sum[sum['set'] == 'valid']
    write('dev_sum.csv', dev_sum)

    test_sum = sum[sum['set'] == 'test']
    write('test_sum.csv', test_sum)


    train_qaps = qaps[qaps['set'] == 'train']
    write('train_qaps.csv', train_qaps)

    dev_qaps = qaps[qaps['set'] == 'valid']
    write('dev_qaps.csv', dev_qaps)

    test_qaps = qaps[qaps['set'] == 'test']
    write('test_qaps.csv', test_qaps)

print("Spliting data to train, dev and test")
split_data()
print("Spliting data to train, dev and test completed")
