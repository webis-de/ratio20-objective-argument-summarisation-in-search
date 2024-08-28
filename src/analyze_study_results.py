import pandas as pd
import itertools
import numpy as np
import collections
import krippendorff
from icecream import ic
from ast import literal_eval
import scipy.stats

def analyze_summarization():
    df = pd.read_csv('../data/snippets_study_results.csv')
    # only keep users 4,5,6,7,8
    df = df[df['user_id'] > 3]
    # check if each user has exactly 99 rows
    ic(df.groupby('user_id').count())
    # keep only user with exactly 99 rows
    df = df.groupby('user_id').filter(lambda x: len(x) == 99)
    # decompress result column
    df['result'] = df['result'].apply(literal_eval)
    df['rank1'] = df['result'].apply(lambda x: x['otherErrorQuestion1'])
    df['rank2'] = df['result'].apply(lambda x: x['otherErrorQuestion2'])
    df['rank3'] = df['result'].apply(lambda x: x['otherErrorQuestion3'])
    df['argumentA'] = df['post_id'].apply(lambda x: x.split('_')[1])
    df['argumentB'] = df['post_id'].apply(lambda x: x.split('_')[2])
    df['argumentC'] = df['post_id'].apply(lambda x: x.split('_')[3])

    results = {'argsme': [], 'llama': [], 'bart': []}
    for i, row in df.iterrows():
        if row['rank1'] == 'other1':
            results[row['argumentA']].append(1)
        elif row['rank1'] == 'other2':
            results[row['argumentB']].append(1)
        elif row['rank1'] == 'other3':
            results[row['argumentC']].append(1)
        if row['rank2'] == 'other4':
            results[row['argumentA']].append(2)
        elif row['rank2'] == 'other5':
            results[row['argumentB']].append(2)
        elif row['rank2'] == 'other6':
            results[row['argumentC']].append(2)
        if row['rank3'] == 'other7':
            results[row['argumentA']].append(3)
        elif row['rank3'] == 'other8':
            results[row['argumentB']].append(3)
        elif row['rank3'] == 'other9':
            results[row['argumentC']].append(3)

    for k, v in results.items():
        ic(k, sum(v)/len(v))
        ic(sorted(collections.Counter(v).most_common(), key=lambda x: x[0]))

    ic.disable()

    # calculate inter-rater agreement as rank correlation
    # kendall's tau
    rank1_annotations = [df[df['user_id']==x].sort_values('post_id')['rank1'].values for x in df.user_id.unique()]
    rank1_annotations = np.array([[1 if x=='other1' else 2 if x=='other2' else 3 for x in y] for y in rank1_annotations])
    rank2_annotations = [df[df['user_id']==x].sort_values('post_id')['rank2'].values for x in df.user_id.unique()]
    rank2_annotations = np.array([[1 if x=='other4' else 2 if x=='other5' else 3 for x in y] for y in rank2_annotations])
    rank3_annotations = [df[df['user_id']==x].sort_values('post_id')['rank3'].values for x in df.user_id.unique()]
    rank3_annotations = np.array([[1 if x=='other7' else 2 if x=='other8' else 3 for x in y] for y in rank3_annotations])
    # for each user 
    # group rank 1, 2, 3 together into tuples
    tuples  = []
    for i in range(len(rank1_annotations)):
        tuples.append([])
        tuples[i] = [[x, y, z] for x, y, z in zip(rank1_annotations[i], rank2_annotations[i], rank3_annotations[i])]


    ic(rank1_annotations)
    ic(scipy.stats.kendalltau(x=tuples[0][0], y=tuples[1][0]))

    # compute all combinations of annotators
    combinations = list(itertools.combinations(range(len(tuples)), 2))
    ic(combinations)

    total = 0
    for c in combinations:
        for i in range(len(tuples[0])):
            kt = scipy.stats.kendalltau(x=tuples[c[0]][i], y=tuples[c[1]][i], method='exact')
            ic(tuples[c[0]][i])
            ic(tuples[c[1]][i])
            ic(kt)
            ic(kt[0])
            total += kt[0]
    ic.enable()
    ic(total/(len(tuples[0]) * len(combinations)))


def analyze_neutralization():
    df = pd.read_csv('../data/neutralization_study_results.csv')
    # only keep users 4,5,6,7,8
    df = df[df['user_id'] > 3]
    # check if each user has exactly 100 rows
    ic(df.groupby('user_id').count())
    # keep only user with exactly 100 rows
    df = df.groupby('user_id').filter(lambda x: len(x) == 100)
    # decompress result column
    df['result'] = df['result'].apply(literal_eval)
    df['rank1'] = df['result'].apply(lambda x: x['otherErrorQuestion1'])
    df['rank2'] = df['result'].apply(lambda x: x['otherErrorQuestion2'])
    df['rank3'] = df['result'].apply(lambda x: x['otherErrorQuestion3'])
    df['post_id'] = df['post_id'].apply(lambda x: x.replace('llama_ppo', 'llama-ppo'))
    df['argumentA'] = df['post_id'].apply(lambda x: x.split('_')[1])
    df['argumentB'] = df['post_id'].apply(lambda x: x.split('_')[2])
    df['argumentC'] = df['post_id'].apply(lambda x: x.split('_')[3])

    results = {'source': [], 'llama-ppo': [], 'llama': []}
    for i, row in df.iterrows():
        if row['rank1'] == 'other1':
            results[row['argumentA']].append(1)
        elif row['rank1'] == 'other2':
            results[row['argumentB']].append(1)
        elif row['rank1'] == 'other3':
            results[row['argumentC']].append(1)
        if row['rank2'] == 'other4':
            results[row['argumentA']].append(2)
        elif row['rank2'] == 'other5':
            results[row['argumentB']].append(2)
        elif row['rank2'] == 'other6':
            results[row['argumentC']].append(2)
        if row['rank3'] == 'other7':
            results[row['argumentA']].append(3)
        elif row['rank3'] == 'other8':
            results[row['argumentB']].append(3)
        elif row['rank3'] == 'other9':
            results[row['argumentC']].append(3)

    for k, v in results.items():
        ic(k, sum(v)/len(v))
        ic(sorted(collections.Counter(v).most_common(), key=lambda x: x[0]))

    ic.disable()

    # calculate inter-rater agreement as rank correlation
    # kendall's tau
    rank1_annotations = [df[df['user_id']==x].sort_values('post_id')['rank1'].values for x in df.user_id.unique()]
    rank1_annotations = np.array([[1 if x=='other1' else 2 if x=='other2' else 3 for x in y] for y in rank1_annotations])
    rank2_annotations = [df[df['user_id']==x].sort_values('post_id')['rank2'].values for x in df.user_id.unique()]
    rank2_annotations = np.array([[1 if x=='other4' else 2 if x=='other5' else 3 for x in y] for y in rank2_annotations])
    rank3_annotations = [df[df['user_id']==x].sort_values('post_id')['rank3'].values for x in df.user_id.unique()]
    rank3_annotations = np.array([[1 if x=='other7' else 2 if x=='other8' else 3 for x in y] for y in rank3_annotations])
    # for each user 
    # group rank 1, 2, 3 together into tuples
    tuples  = []
    for i in range(len(rank1_annotations)):
        tuples.append([])
        tuples[i] = [[x, y, z] for x, y, z in zip(rank1_annotations[i], rank2_annotations[i], rank3_annotations[i])]


    ic(rank1_annotations)
    ic(scipy.stats.kendalltau(x=tuples[0][0], y=tuples[1][0]))

    # compute all combinations of annotators
    combinations = list(itertools.combinations(range(len(tuples)), 2))

    total = 0
    for c in combinations:
        for i in range(len(tuples[0])):
            kt = scipy.stats.kendalltau(x=tuples[c[0]][i], y=tuples[c[1]][i])
            ic(tuples[0][i])
            ic(tuples[1][i])
            ic(kt)
            ic(kt[0])
            total += kt[0]

    ic.enable()
    ic(total/(len(tuples[0]) * len(combinations)))


def analyze_search():
    df = pd.read_csv('../data/search_study_results.csv')
    # only keep users 4,5,6,7,8
    df = df[df['user_id'].isin([3,4,5,6,7])]
    # check if each user has exactly 100 rows
    ic(df.groupby('user_id').count())
    # keep only user with exactly 100 rows
    df = df.groupby('user_id').filter(lambda x: len(x) == 99)
    # decompress result column
    df['result'] = df['result'].apply(literal_eval)
    df['selection'] = df['result'].apply(lambda x: x['radio-card'])
    df['argumentA'] = df['post_id'].apply(lambda x: x.split('_')[1])
    df['argumentB'] = df['post_id'].apply(lambda x: x.split('_')[2])

    results = []
    for i, row in df.iterrows():
        if row['selection'] == '1':
            results.append(row['argumentB'])
        else:
            results.append(row['argumentA'])

    df['selection'] = results
    ic(df['selection'].value_counts())

    #ic.disable()
    df['selection'] = df['selection'].apply(lambda x: 1 if x=='bart' else 2)
    alpha = krippendorff.alpha(reliability_data=[df[df['user_id']==x].sort_values('post_id')['selection'].values for x in df.user_id.unique()], level_of_measurement='nominal')
    ic(alpha)


if __name__ == '__main__':
    #analyze_summarization()
    #analyze_neutralization()
    analyze_search()
