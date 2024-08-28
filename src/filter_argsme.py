import pandas as pd

def add_prediction(df, predictions):
    # set column names of predictions to post_id and prediction
    predictions.columns = ['post_id', 'prediction']
    df['prediction'] = predictions['prediction'].values.tolist()
    df['prediction'] = df['prediction'].apply(lambda x: [x.split(",")[0][1:], x.split(",")[1][1:-1]])
    df['appropriate'] = df['prediction'].apply(lambda x: float(x[0]))
    df['inappropriate'] = df['prediction'].apply(lambda x: float(x[1]))
    return df

if __name__ == '__main__':
    df = pd.read_csv('../data/argsme/inappropriate_arguments_sample.csv')

    print(df.shape)
    df = add_prediction(df, pd.read_csv(
        "../data/argsme/ensemble_predictions_argsme.txt", sep="\t", header=None))

    selected = []
    issue_count = {}
    for i, row in df.iterrows():
        if row['inappropriate'] >= 0.5:
            if row['query'] not in issue_count:
                issue_count[row['query']] = {}
            if row['stance'] not in issue_count[row['query']]:
                issue_count[row['query']][row['stance']] = 0
            if issue_count[row['query']][row['stance']] < 5:
                selected.append(i)
                issue_count[row['query']][row['stance']] += 1

    df = df.iloc[selected]
    print(df.shape)
    print(df.head())
    print(issue_count)
    df.to_csv('../data/argsme/inappropriate_arguments_sample_filtered.csv', index=False)
