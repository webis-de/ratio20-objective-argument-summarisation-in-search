import pandas as pd
import random

def prepare_neutralization():
    app_df = pd.read_csv('../data/inappropriate_arguments_sample_100_appropriateness.csv')
    llam_ndf = pd.read_csv('../data/results-by-corpus/appropriateness/neutralization/llama.csv', sep="\t", header=None)
    llam_ndf.columns = ['id', 'llama_neutralization']
    llama_df_ppo_10a_00ss = pd.read_csv('../data/results-by-corpus/appropriateness/neutralization/llama_ppo_rewrite_appropriateness_llama-7b-harmonic-mean-10a-00ss.csv', sep="\t", header=None)
    llama_df_ppo_10a_00ss.columns = ['id', 'llama_neutralization']

    app_df['id'] = [i for i in range(1, len(app_df) + 1)]
    app_df['source'] = app_df['argument'].tolist()
    app_df['rewrite_a'] = llam_ndf['llama_neutralization']
    app_df['rewrite_b'] = llama_df_ppo_10a_00ss['llama_neutralization']
    app_df['batch'] = [2] * len(app_df)

    decodable_ids = []
    shuffled_sources = []
    shuffled_rewrites_a = []
    shuffled_rewrites_b = []

    # get a random order of [0,1,2]
    for i, row in app_df.iterrows():
        rand_order = random.sample([row.source, row.rewrite_a, row.rewrite_b], 3)
        shuffled_sources.append(rand_order[0])
        shuffled_rewrites_a.append(rand_order[1])
        shuffled_rewrites_b.append(rand_order[2])
        # get position of source in the random rand_order
        pos_source = rand_order.index(row.source)
        pos_rewrite_a = rand_order.index(row.rewrite_a)
        pos_rewrite_b = rand_order.index(row.rewrite_b)
        tmp_order = {'source': pos_source, 'llama': pos_rewrite_a, 'llama_ppo': pos_rewrite_b}
        # get keys ordered by value
        tmp_order = sorted(tmp_order, key=tmp_order.get)

        decodable_ids.append(str(row.id) + '_' + tmp_order[0] + '_' + tmp_order[1] + '_' + tmp_order[2])

    app_df['source'] = shuffled_sources
    app_df['rewrite_a'] = shuffled_rewrites_a
    app_df['rewrite_b'] = shuffled_rewrites_b
    app_df['id'] = decodable_ids

    app_df = app_df[['id', 'source', 'rewrite_a', 'rewrite_b', 'issue', 'batch']]
    app_df.to_csv('../data/neutralization_study_data.csv', index=False)


def prepare_snippets():
    args_df = pd.read_csv("../data/inappropriate_arguments_sample_100_argsme.csv")
    llama_df = pd.read_csv("../data/results-by-corpus/argsme/summarization/llama.csv", sep="\t", header=None)
    llama_df.columns = ['id', 'llama_gist']
    bart_df = pd.read_csv("../data/results-by-corpus/argsme/summarization/bart.csv")

    args_df['id'] = [i for i in range(1, len(args_df) + 1)]
    args_df['issue'] = args_df['query'].tolist()
    args_df['issue'] = [x.capitalize() for x in args_df['issue'].tolist()]
    args_df['source'] = args_df['argument'].tolist()
    args_df['rewrite_a'] = args_df['snippet'].tolist()
    args_df['rewrite_b'] = bart_df['bart_gist']
    args_df['rewrite_c'] = llama_df['llama_gist']
    args_df['batch'] = [1] * len(args_df)

    decodable_ids = []
    shuffled_rewrites_a = []
    shuffled_rewrites_b = []
    shuffled_rewrites_c = []

    # get a random order of [0,1,2]
    for i, row in args_df.iterrows():
        rand_order = random.sample([row.rewrite_a, row.rewrite_b, row.rewrite_c], 3)
        shuffled_rewrites_a.append(rand_order[0])
        shuffled_rewrites_b.append(rand_order[1])
        shuffled_rewrites_c.append(rand_order[2])
        # get position of source in the random rand_order
        pos_rewrite_a = rand_order.index(row.rewrite_a)
        pos_rewrite_b = rand_order.index(row.rewrite_b)
        pos_rewrite_c = rand_order.index(row.rewrite_c)
        tmp_order = {'argsme': pos_rewrite_a, 'llama': pos_rewrite_b, 'bart': pos_rewrite_c}
        # get keys ordered by value
        tmp_order = sorted(tmp_order, key=tmp_order.get)

        decodable_ids.append(str(row.id) + '_' + tmp_order[0] + '_' + tmp_order[1] + '_' + tmp_order[2])

    args_df['rewrite_a'] = shuffled_rewrites_a
    args_df['rewrite_b'] = shuffled_rewrites_b
    args_df['rewrite_c'] = shuffled_rewrites_c
    args_df['id'] = decodable_ids

    args_df = args_df[['id', 'source', 'rewrite_a', 'rewrite_b', 'rewrite_c', 'issue', 'batch']]
    args_df.to_csv('../data/snippets_study_data.csv', index=False)


def prepare_snippets_2():
    search_queries = {
        "Climate change": "Pros and cons of climate change",
        "Trump": "Advantages and disadvantages of Trump's policies",
        "Donald": "Arguments for and against Donald Trump",
        "Feminism": "Debate over the benefits and drawbacks of feminism",
        "Death penalty": "Arguments in favor and against the death penalty",
        "Abortion": "Controversies and viewpoints on abortion",
        "Brexit": "Supporting and opposing arguments for Brexit",
        "Google": "Pros and cons of using Google",
        "Vegan": "Debate on the benefits and downsides of a vegan diet",
        "Nuclear energy": "For and against arguments about nuclear energy",
        "Donald trump": "Opinions in support and against Donald Trump"
    }

    app_df = pd.read_csv('../data/inappropriate_arguments_sample_100_argsme.csv')
    bart_df = pd.read_csv('../data/results-by-corpus/argsme/summarization/bart.csv')
    llama_df_ppo_10a_00ss = pd.read_csv('../data/results-by-corpus/argsme/both/bart_summarized_and_neutralized.csv', sep="\\t", header=None)
    llama_df_ppo_10a_00ss.columns = ['id', 'llama_neutralization']

    app_df['id'] = [i for i in range(1, len(app_df) + 1)]
    app_df['issue'] = app_df['query'].tolist()
    app_df['issue'] = [x.capitalize() for x in app_df['issue'].tolist()]
    app_df['source'] = app_df['argument'].tolist()
    app_df['snippet_a'] = bart_df['bart_gist']
    app_df['snippet_b'] = llama_df_ppo_10a_00ss['llama_neutralization']
    app_df['batch'] = [2] * len(app_df)

    decodable_ids = []
    shuffled_snippets_a = []
    shuffled_snippets_b = []

    # get a random order of [0,1,2]
    for i, row in app_df.iterrows():
        rand_order = random.sample([row.snippet_a, row.snippet_b], 2)
        shuffled_snippets_a.append(rand_order[0])
        shuffled_snippets_b.append(rand_order[1])
        # get position of source in the random rand_order
        pos_snippet_a = rand_order.index(row.snippet_a)
        pos_snippet_b = rand_order.index(row.snippet_b)
        tmp_order = {'llama': pos_snippet_a, 'bart': pos_snippet_b}
        # get keys ordered by value
        tmp_order = sorted(tmp_order, key=tmp_order.get)

        decodable_ids.append(str(row.id) + '_' + tmp_order[0] + '_' + tmp_order[1])

    app_df['issue'] = [search_queries[x] for x in app_df['issue'].tolist()]
    app_df['snippet_a'] = shuffled_snippets_a
    app_df['snippet_b'] = shuffled_snippets_b
    app_df['id'] = decodable_ids

    app_df = app_df[['id', 'source', 'snippet_a', 'snippet_b', 'issue', 'batch']]
    app_df.to_csv('../data/search_study_data.csv', index=False)


if __name__ == '__main__':
    #prepare_neutralization()
    #prepare_snippets()
    prepare_snippets_2()
