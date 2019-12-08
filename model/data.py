import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string




def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, args):
        nltk.download('punkt')
        nltk.download('stopwords')
        path = '.data/squad'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'

        print("preprocessing data files...")
        if not os.path.exists(f'{path}/{args.train_file}l'):
            self.preprocess_file(f'{path}/{args.train_file}')
        if not os.path.exists(f'{path}/{args.dev_file}l'):
            self.preprocess_file(f'{path}/{args.dev_file}')

        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       'f_idx': ('f_idx', self.RAW),
                       'se_idx': ('se_idx', self.LABEL),
                       't_idx': ('t_idx', self.LABEL),
                       'fo_idx': ('fo_idx', self.LABEL),
                       'fi_idx': ('fi_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('f_idx', self.LABEL), ('se_idx', self.LABEL),('t_idx', self.LABEL), ('fo_idx', self.LABEL),('fi_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.dev = data.TabularDataset.splits(
                path=path,
                train=f'{args.train_file}l',
                validation=f'{args.dev_file}l',
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        #cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))

        print("building iterators...")
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_sizes=[args.train_batch_size, args.dev_batch_size],
                                       device=device,
                                       sort_key=lambda x: len(x.c_word))

    def preprocess_file(self, path):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            counter = 0                                                            #jk
            total = 0
            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context.lower())
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            total += 1
                            answer = ans['text']

                            ############## jk
                            word_tokens = word_tokenize(answer.lower())
                            # filtered_sentence = [w for w in word_tokens if not w in stop_words]
                            word_tokens = list(filter(lambda token: token not in string.punctuation, word_tokens))

                            filtered_sentence = []
                            stop_words = set(stopwords.words('english'))
                            for w in word_tokens:
                                if w not in stop_words:
                                    filtered_sentence.append(w)

                            # distributing probability to all gt values
                            prob = 1/filtered_sentence
                            gt = [0]*len(tokens)
                            for i in filtered_sentence:
                                try:
                                    gt[tokens.index(i)] = prob
                                except:
                                   # print(i)
                                    pass
                            # if len(gt) == 2:
                            #    # print("bogus data found")
                            #     gt += [0,0]
                            # else:
                            #     pass
                            #    # print("normal data found-----------------------------")

                            ##################jk


                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break
                            if len(gt) > 1:
                                dump.append(dict([('id', id),
                                                ('context', context),
                                                ('question', question),
                                                ('answer', answer),
                                                ('f_idx', gt),                   #jk
                                                ('se_idx', gt[0] ),                   #jk
                                                ('t_idx', gt[1] ),                    #jk
                                                ('fo_idx', gt[0] ),                   #jk
                                                ('fi_idx', gt[0] )]))                 #jk
                            else:
                                #print("skipping", gt)                                #jk
                                counter += 1                                         #jk

        print(counter, "QAs skipped because gt < 5:", total)                                 #jk





        with open(f'{path}l', 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)
