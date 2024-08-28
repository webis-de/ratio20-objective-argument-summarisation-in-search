import requests
import re
import pandas as pd

TOP_10_TOPICS = [
    'climate change',
    'feminism',
    'abortion',
    'trump',
    'brexit',
    'death penalty',
    'google',
    'vegan',
    'nuclear energy',
    'donald trump'
]


def get_word_count(text):
    return len(re.findall(r'\w+', text))


def query_argsme(issue):
    if ' ' in issue:
        issue = issue.replace(' ', '%20')
    url = 'https://www.args.me/api/v2/arguments?query={}&pageSize=300&format=json'.format(
        issue)
    response = requests.get(url)
    return response.json()


def filter_response(response):
    query = [response['query']['text']] * len(response['arguments'])
    argument = [arg['premises'][0]['text'] for arg in response['arguments']]
    stance = [arg['stance'] for arg in response['arguments']]
    snippet = [''] * len(response['arguments'])
    for i, arg in enumerate(response['arguments']):
        for anno in arg['premises'][0]['annotations']:
            if anno['type'] == 'me.args.argument.Snippet':
                if snippet[i] == '':
                    snippet[i] = argument[i][anno['start']:anno['end']]
                else:
                    snippet[i] = snippet[i] + ' ... ' + argument[i][anno['start']:anno['end']]
    return query, argument, stance, snippet


if __name__ == '__main__':
    query, argument, stance, snippet = [], [], [], []
    for topic in TOP_10_TOPICS:
        response = query_argsme(topic)
        tmp_query, tmp_argument, tmp_stance, tmp_snippet = filter_response(response)
        count = 0
        for tq, ta, ts, tsn in zip(tmp_query, tmp_argument, tmp_stance, tmp_snippet):
            tmp_word_count = get_word_count(ta)
            if tmp_word_count >= 100 and tmp_word_count <= 500:
                count += 1
                query.append(tq)
                argument.append(ta)
                stance.append(ts)
                snippet.append(tsn)
    df = pd.DataFrame({'query': query,
                       'argument': argument,
                       'stance': stance,
                       'snippet': snippet})
    df['id'] = df.index
    df.to_csv('../data/argsme/inappropriate_arguments_sample.csv', index=False)
