import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tools.text import clean_text

def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default='C:/data/niosh_ifund/')
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--train_file', type=str, default='train.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    # Importing the raw data
    train = pd.read_csv(args.data_dir + args.train_file,
                        usecols=['text', 'event'])
    test = pd.read_csv(args.data_dir + args.test_file,
                       usecols=['text'])
    
    # Adding a random identifier for the BERT scripts
    num_train = train.shape[0]
    num_test = test.shape[0]
    num_records = num_train + num_test
    ids = np.array([''.join(['record', str(num)]) 
                    for num in list(range(num_records))])
    np.random.shuffle(ids)
    train['id'] = ids[0:num_train]
    test['id'] = ids[num_train:]
    
    # Lowercasing and adding spaces around common abbreviations;
    # only fixes a few things
    train.text = pd.Series(clean_text(train.text))
    test.text = pd.Series(clean_text(test.text))
    
    # Clipping the docs to the max length
    train_lengths = np.array([len(doc.split()) for doc in
                            pd.concat([train.text, test.text])])
    test_lengths = np.array([len(doc.split()) for doc in
                            pd.concat([train.text, test.text])])
    clip_to = np.min([np.max(train_lengths), np.max(test_lengths)])
    train.text = pd.Series([' '.join(doc.split()[:clip_to]) 
                            for doc in train.text])
    test.text = pd.Series([' '.join(doc.split()[:clip_to])
                          for doc in test.text])
    pd.Series(clip_to).to_csv(args.data_dir + 'clip_to.csv',
                              header=False, index=False)
    
    # Making a lookup dictionary for the event codes
    code_df = pd.read_csv(args.data_dir + 'code_descriptions.csv')
    codes = code_df.event.values
    code_dict = dict(zip(codes, np.arange(len(codes))))
    train.event = [code_dict[code] for code in train.event]
            
    # Saving the code dict to disk
    code_df = pd.DataFrame.from_dict(code_dict, orient='index')
    code_df['event_code'] = code_df.index
    code_df.columns = ['value', 'event_code']
    code_df.to_csv(args.data_dir + 'code_dict.csv', index=False)
    
    # Rearranging the columns for BERT
    train['filler'] = np.repeat('a', train.shape[0])
    test['filler'] = np.repeat('a', test.shape[0])
    train = train[['id', 'event', 'filler', 'text']]
    test = test[['id', 'text']]
    
    # Shuffling the rows
    train = train.sample(frac=1)
    
    # Writing the regular splits to disk
    train.to_csv(args.data_dir + 'train.tsv', sep='\t', 
                 index=False, header=False)
    test.to_csv(args.data_dir + 'test.tsv', sep='\t', 
                index=False, header=True)
