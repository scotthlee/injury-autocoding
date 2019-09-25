import joblib, argparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default='C:/data/niosh_ifund/')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--test_file', type=str, default='test.tsv')
    parser.add_argument('--text_only', type=bool, default=True)
    parser.add_argument('--train_blender', type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    # Importing the event code dictionary to convert the BERT indices
    code_df = pd.read_csv(args.data_dir + 'code_dict.csv')
    code_dict = dict(zip(code_df.value, code_df.event_code))
    
    # Importing the scores from the 4 BERT runs
    if args.mode == 'validate':
        run_folder = 'val_runs'
    elif args.mode == 'test':
        run_folder = 'test_runs'
    run1_probs = np.array(pd.read_csv(args.data_dir + run_folder + 
                                      '/run_1/test_results.tsv', sep='\t', 
                                      header=None))
    run2_probs = np.array(pd.read_csv(args.data_dir + run_folder + 
                                      '/run_2/test_results.tsv', sep='\t', 
                                      header=None))
    run3_probs = np.array(pd.read_csv(args.data_dir + run_folder + 
                                      '/run_3/test_results.tsv', sep='\t', 
                                      header=None))
    run4_probs = np.array(pd.read_csv(args.data_dir + run_folder + 
                                      '/run_4/test_results.tsv', sep='\t', 
                                      header=None))
    prob_list = [run1_probs, run2_probs, run3_probs, run4_probs]
    
    # Grouping the probabilities for regular averaging
    avg_probs = np.mean(prob_list, axis=0)
    avg_guesses = np.array([code_dict[code] 
                            for code in np.argmax(avg_probs, axis=1)])
    
    # Grouping the probabilities for blending
    wide_probs = np.concatenate(prob_list, axis=1)
    
    # Producing guesses when only the input text is available
    if args.text_only:
        # Loading the blender model
        # lgr = joblib.load(args.data_dir + 'blender.joblib')
        # blend_guesses = lgr.predict(wide_probs)
        # blend_probs = np.max(lgr.predict_proba(wide_probs), axis=1)
        # print(blend_probs[0])
        
        # Exporting the guesses to disk
        ids = pd.read_csv(args.data_dir + args.test_file,
                          sep='\t')['id']
        guess_df = pd.DataFrame(pd.concat([ids,
                                           pd.Series(avg_guesses),
                                           pd.Series(np.max(avg_probs,
                                                            axis=1))],
                                          axis=1))
        guess_df.columns = ['id', 'avg_guess', 'avg_prob']
        guess_df.to_csv(args.data_dir + 'guesses.csv', 
                        header=True, index=False)
    
    # Producing guesses and scores when the labels are also available
    else:
        # Getting the guesses from the blending model
        if args.train_blender:
            targets = pd.read_csv(args.data_dir + args.test_file)['event']
            lgr = LogisticRegression()
            lgr.fit(wide_probs, targets)
            joblib.dump(lgr, args.data_dir + 'blender.joblib')
        else:
            lgr = joblib.load(args.data_dir + 'blender.joblib')
        blend_guesses = lgr.predict(wide_probs)
        
        # Importing the test records and getting the various scores
        test_records = pd.read_csv(args.data_dir + args.test_file)
        targets = np.array(test_records.event)
        avg_f1 = f1_score(targets, avg_guesses, average='weighted')
        blend_f1 = f1_score(targets, blend_guesses, average='weighted')
        print('')
        print('Weighted macro f1 on the test set is ' + str(avg_f1) +
              ' with averaging and ' + str(blend_f1) + ' with blending.' )
    
        # Writing results to disk
        results = pd.DataFrame(pd.concat([test_records.id,
                                          test_records.text, 
                                          test_records.event,
                                          pd.Series(avg_guesses),
                                          pd.Series(blend_guesses)], axis=1))
        results.columns = ['id', 'text', 'event', 'avg_guess', 'blend_guess']
        results.to_csv(args.data_dir + 'results.csv', header=True, index=False)
