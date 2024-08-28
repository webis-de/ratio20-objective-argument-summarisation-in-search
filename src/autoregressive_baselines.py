import pandas as pd
import torch
from peft import PeftModel, PeftConfig
import os
import spacy
import re
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MODEL_LIST = [
    "EleutherAI/gpt-j-6B",
    "bigscience/bloom-7b1",
    "huggyllama/llama-7b",
    "nlpcloud/instruct-gpt-j-fp16",
    "bigscience/bloomz-7b1",
    "larrylawl/alpaca-7b"
]


GEN_ARGS = {
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 0,
    "temperature": 1.0,
    "num_return_sequences": 1,
}

PROMPT_PATTERN = '''Here is some text: {{{}}}. Here is a rewrite of the text that is more appropriate and makes only minimal changes: {{{}}}.'''

INSTRUCT_PRE = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'''
INSTRUCT_PATTERN_REWRITE = '''### Instruction:\nRewrite the following argument on the topic of "{}" to be more appropriate and make only minimal changes to the original argument.\n\n### Input:\n{}\n\n### Response:\n{}\n\n'''
INSTRUCT_PATTERN_EXTRACT = '''### Instruction:\nThe following is an argument on the topic of "{}". Extract a coherent gist from it that is exactly two sentences long.\n\n### Input:\n{}\n\n### Response:\n{}\n\n'''
INSTRUCT_PATTERN_REWRITE_EXTRACT = '''### Instruction:\nRewrite the following argument on the topic of "{}" to be more appropriate and make only minimal changes to the original argument. Subsequently, extract a coherent gist from it that is exactly two sentences long.\n\n### Input:\n{}\n\n### Response:\n{}\n\n'''
INSTRUCT_PATTERN_EXTRACT_REWRITE = '''### Instruction:\nThe following is an argument on the topic of "{}". Extract a coherent gist from it that is exactly two sentences long. Subsequently, rewrite the gist to be more appropriate and make only minimal changes to the original gist.\n\n### Input:\n{}\n\n### Response:\n{}\n\n'''

FEW_SHOT_EXAMPLES_SUB = [
    (''''Towed three times and impounded for 30 days each time? Man, you're just not getting the message, are you? If you are in California, you bet the police can forfeit your vehicle and it doesn't take three times to make it a charm. Technically, your vehicle could be subject to forfeiture proceedings after your first suspended license beef. Someone like you is exactly the reason the legislature designed that law, because your privilege to drive has been taken away from you and yet you obviously continue to drive. People like you are involved in an exponentially higher than average number of traffic accidents so the legislature figured maybe people like you should have your vehicles forfeited to the state if you just didn't go along with the game plan. Voila - I give you California Vehicle Code section 14607.6...and a link to it below. It would also be worth your time to review 14607.4, whether or not you live in California. You really need to stop driving. Really.''',
     '''If you are in California, the police can forfeit your vehicle after fewer than three incidents. Technically, your vehicle could be subject to forfeiture proceedings after your first suspended license. The legislature designed that law with people in mind that don't take suspended licenses seriously and continue to drive anyway. Such people tend to be involved in an exponentially higher-than-average number of traffic accidents, so the legislature figured that they should have their vehicles taken into custody if they don't take the suspension seriously. Voila - I give you California Vehicle Code section 14607.6 and a link to it below. You might also want to take time to review 14607.4, whether or not you live in California. After three suspensions, you should consider stopping driving.'''),
    (''''Yes i am completely for it. People are arguing that it is barbaric and inhumane but who can stand up and say that some perv who has raped and killed a child still has human rights and the right to live. We would put down a dangerous dog why not do the same to some of the scum that lives in our country. The justice system in britain at the moment is hopeless. Far to many people are gettin away with all sorts and something needs to be done!!''',
     '''Yes, I am for it. People argue that it is barbaric and inhumane, but who can stand up and say that a person conducting child abuse still has the right to live? We would put down a dangerous dog; why not act in a similar way with criminals living in our country? I don't think very highly of the justice system in Britain at the moment. Far too many people are getting away with all sorts of things, and something needs to be done!'''),
    (''''Porn is Wrong. mainly because they are Not Doing it Right. it should be Hi Def. in three years, it will be in 3-D.''',
     '''Porn is wrong because they are not doing it right. But jokes aside, it should be in higher resolution. In three years, it will be in 3D.'''),
    (''''There should be no argument in this really...whatever way yu see a fetus...its still a living form that has been created in a very intimate way... you shouldn't be changing what mothernature or God or fate or whatever has decided for you...and if you didn;t wannna get preggo in the first place...don't have sex or use protection. Yeh there are some women that get raped and it's very unfortunate but they should give the child up for adoption. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of it's life?''',
     '''In point of view, whatever way you see a fetus it's still a living form that has been created in a very intimate way. You shouldn't be changing what mother nature or God or fate or whatever has decided for you and if you didn't wanna get pregnant in the first place, don't have sex or use protection. I see the issue that there there are some women that get raped, and it's very unfortunate, but I think they should give the child up for adoption in such cases. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of its life?'''),
    (''''Believing "Evolution" as in Darwinism and the like, is like believing the puzzle can be solved by pouring the pieces out because two pieces kind of stuck together.''',
     '''Believing in "Evolution," as in Darwinism and the like, is like believing the puzzle can be solved by pouring the pieces out because two pieces kind of stuck together. The probability of such things happening tends to be very low; thus, the "evolution" theory seems to be very unlikely, in my opinion.'''),
    (''''The conversation can not be used as evidence in a court of law. I don't know what the lady hoped to gain from recording the conversation other than to create more drama. Some people are hooked on drama and they actually do what they can to create it. Run as far away and as fast as you can from these types. They will suck you dry.''',
     '''The conversation cannot be used as evidence in a court of law. I don't know what the lady hoped to achieve by recording the conversation except that she wanted to create even more drama. Some people like drama and try to create it. I would suggest that you do not surround yourself with such people because they only cost you energy.'''),
    (''''i would turn in my wife because its wrong to kill someone. it could have been an accident but it was still wrong and besides the police are going to find out who killed that person but i don't want her to leave me for a long period of time so i would tell but then again i wouldn't.''',
     '''On the one hand, I consider it to be the right thing to turn in my wife because it's wrong to kill someone. It could have been an accident, but it was still wrong, and besides, the police are going to find out who killed that person. On the other hand, I don't want her to leave me for a long period of time, so I'm a bit torn in this regard.'''),
    (''''it dose not show kids expressions and unforms dose not show is it''',
     '''School uniforms do not let kids express themselves, and it doesn't let them show who they are.'''),
    (''''Firebug, WebDeveloper, TabMix, FaviconizeTab, GreaseMonkey, IETab (to use when you visit microsot.com). Just some reason why i prefer Firefox''',
     '''Firefox has many great tools and plugins, such as Firebug, WebDeveloper, TabMix, FaviconizeTab, GreaseMonkey, and IETab (to use when you visit microsot.com). Those tools are just some of the reasons why I prefer Firefox.''')
]

FEW_SHOT_EXAMPLES_CORE = [
    ('''Hitler invaded Poland in 1932 and the world turned against Germany. In fact, there are dozens if cases in the last 100 years where countries have invaded other nations and the world has caused uproar and rose up against it. Yet some dumb Texan does it and gets away with it. Try him for war crimes, along with Tony Blair and have them both executed or imprisoned.''',
     '''Hitler invaded Poland in 1932, and the world turned against Germany. In fact, there are dozens of cases in the last 100 years where countries have invaded other nations, and the world has caused uproar and rose up against it. Yet an American does it and gets away with it. He should be prosecuted for war crimes, along with Tony Blair.'''),
    (''''There should be no argument in this really...whatever way yu see a fetus...its still a living form that has been created in a very intimate way... you shouldn't be changing what mothernature or God or fate or whatever has decided for you...and if you didn;t wannna get preggo in the first place...don't have sex or use protection. Yeh there are some women that get raped and it's very unfortunate but they should give the child up for adoption. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of it's life?''',
     '''In point of view, whatever way you see a fetus it's still a living form that has been created in a very intimate way. You shouldn't be changing what mother nature or God or fate or whatever has decided for you and if you didn't wanna get pregnant in the first place, don't have sex or use protection. I see the issue that there there are some women that get raped, and it's very unfortunate, but I think they should give the child up for adoption in such cases. It's not the child's fault that it was created. So why should the goring being have to pay the ultimate price of its life?'''),
    ('''We will be able to ban water bottles until we get out of this recession!''',
     '''Banning water bottles is a costly or unpopular policy that can only be implemented when the economy is doing well.'''),
    ('''tv because only tv can bring u live news at books u can't find also it's educational''',
     '''TV is better than books because TV can bring you live news that you cannot find in books. Also, it's educational.''')
]

FEW_SHOT_EXAMPLE_ROOT = [
    ('''Fair trade capitalist. Extreme right, I hate any form of marxism. I believe a man should be judge by who he is not his hide. I believe that the solution for the problems of this world is found in the bible. Public education is a waste of money, due to the literacy rate and no basic understanding of economics. Having this as a basis, the world will end one day, not worried about it, because of the laws of nature, namely entropy. The hatred for Israel and Christianity will continue to grow. Nothing will change we have always had storms and always will. We will always have wars, because man is basically selfish''',
     '''I advocate for fair trade and free markets. I strongly reject any form of Marxism. I value individual character over group identity. I find guidance and hope in the bible. I question the effectiveness and efficiency of public education, given the low literacy rate and lack of economic literacy. Based on these views, I accept that the world will end one day, according to the laws of nature, such as entropy. I also expect that Israel and Christianity will face more hatred. I do not think that the world will change much. There have always been storms and wars, and there always will be, because human nature is flawed and selfish.''')
]

FEW_SHOT_EXAMPLES = [
    ('''Coming from a casual internet user, I prefer IE because it has a sleeker design, and I find the bookmark/history thing easier. I'm using firefox though, cuz I don't want to transfer my bookmarks again after my brother made it my default browser.''',
     '''Although I use Firefox as my default browser because I don't want to transfer my bookmarks, as a casual Internet user I prefer IE because it has a sleeker design and an easier bookmarking function.'''),
    ('''Porn is wrong when it is not done in moderation. Porn addicts turn out to have intimacy issues in their relationships and mistreat and view women in a negative manner. Also people who are addicted to porn, expect all women to look like porn stars and act like porn stars, when in reality that is not what sex is about... But it tends to screw up their reality.''',
     '''I believe porn is wrong if not consumed in moderation, as many porn addicts tend to have intimacy issues in their relationships and are prone to mistreat and view women in a negative manner. In addition, people who are addicted to porn have a screwed up view of sex and expect all women to look and act like porn stars, when in reality that is not the case.I believe porn is wrong if not consumed in moderation, as many porn addicts tend to have intimacy issues in their relationships and are prone to mistreat and view women in a negative manner. In addition, people who are addicted to porn have a screwed up view of sex and expect all women to look and act like porn stars, when in reality that is not the case.'''),
    ('''THE SCHOOL UNIFORM IS A VERY GOOOOOOOOOD IDEA , WHY ?? becouse the school uniform makes pupils concentrated on their education than on their clothes and I believe that school uniform instills discipline among pupils it makes pupils with diferent material statuses more equal :)''', '''School uniforms are a good idea because they make students focus more on their education than on their clothes, which I think leads to more discipline and makes students of different material status more equal.''')
]



# check for length of at least 200 words
def get_word_count(text):
    return len(re.findall(r'\w+', text))


class AutoRegressivePredictor:
    def __init__(self, model_name, gen_args, peft_model_name=None):
        if peft_model_name is not None:
            peft_config = PeftConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(model, model_name).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained('/bigwork/nhwpziet/appropriateness-style-transfer/data/models/instruction-finetuning/llama-7b-instruct')

        self.gen_args = gen_args
        self.num_written = 0

    def write_to_file(self, file_path, predictions):
        with open(file_path, 'a') as f:
            for prediction in predictions:
                prediction = prediction.replace("\n", " ")
                prediction = prediction.replace("\t", " ")
                f.write(f"{self.num_written}\t{prediction}\n")
                self.num_written += 1

    def predict_ds(self, ds, to_file=False, file_path=None, overwrite=False):
        if to_file and not overwrite:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                self.num_written = len(lines)
            ds = ds[self.num_written:]
        if to_file and overwrite:
            self.num_written = 0
            with open(file_path, 'w') as f:
                f.write("")
        for sample in tqdm(ds, total=len(ds)):
            sample_predictions = self.predict_sample(sample)
            if to_file:
                self.write_to_file(file_path, sample_predictions)

    def predict_batch(self, sample):
        with torch.no_grad():
            input_ids = self.tokenizer(sample, return_tensors="pt").input_ids.to("cuda")
            output = self.model.generate(
                input_ids,
                **self.gen_args,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)


class AppropriatenessPredictorFromLM(AutoRegressivePredictor):
    def __init__(self, model_name, gen_args):
        super().__init__(model_name, gen_args)

    def predict_sample(self, sample):
        with torch.no_grad():
            prompt, post_text = sample
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            post_text_input_ids = self.tokenizer(post_text, return_tensors="pt").input_ids.to("cuda")
            #i = len(FEW_SHOT_EXAMPLES_SUB) - 1
            #while len(prompt_input_ids[0]) + int(len(post_text_input_ids[0]) * 2) > 2048:
            #    prompt = prompt[:-len(INSTRUCT_PATTERN[:-4].format(post_text))]
            #    prompt = prompt[:-len(INSTRUCT_PATTERN.format(FEW_SHOT_EXAMPLES_SUB[i][0], FEW_SHOT_EXAMPLES_SUB[i][1]))]
            #    prompt = prompt + INSTRUCT_PATTERN[:-4].format(post_text)
            #    prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            #    i -= 1
            self.gen_args["max_new_tokens"] = min([2048-len(prompt_input_ids[0]), int(len(post_text_input_ids[0]) * 2)])
            self.gen_args["min_new_tokens"] = min([2048-len(prompt_input_ids[0]), int(len(post_text_input_ids[0]) * 0.5)])
            outputs = self.model.generate(
                prompt_input_ids,
                **self.gen_args,
            )
            decoded_outputs = []
            for output in outputs:
                decoded_outputs.append(self.tokenizer.decode(output[len(prompt_input_ids[0]):], skip_special_tokens = True).split('''}''')[0])
            return decoded_outputs


class AppropriatenessPredictorFromInstructLM(AutoRegressivePredictor):
    def __init__(self, model_name, gen_args, task):
        super().__init__(model_name, gen_args)
        self.task = task

        if 'extract' in task:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.add_pipe("sentencizer")

    def predict_sample(self, sample):
        with torch.no_grad():
            prompt, post_text = sample
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            if 'extract' in self.task:
                # split post_text into sentences
                post_text_sents = [sent.text for sent in self.nlp(post_text).sents]
                # get number of tokens in each sentences
                post_text_sents_num_tokens = [len(self.tokenizer(sent, return_tensors="pt").input_ids[0]) for sent in post_text_sents]
                # sum up the number  of tokens in the two longest sentences
                post_text_sents_num_tokens.sort(reverse=True)
                post_text_sents_num_tokens = post_text_sents_num_tokens[:2]
                post_text_sents_num_tokens_sum = sum(post_text_sents_num_tokens)
                self.gen_args["max_new_tokens"] = post_text_sents_num_tokens_sum
            else:
                self.gen_args["max_new_tokens"] = 2048-len(prompt_input_ids[0])
            outputs = self.model.generate(
                prompt_input_ids,
                **self.gen_args,
            )
            decoded_outputs = []
            for output in outputs:
                tmp_out = self.tokenizer.decode(output[len(prompt_input_ids[0]):], skip_special_tokens = True).strip()
                if 'extract' in self.task:
                    tmp_out = ' '.join([sent.text for sent in self.nlp(tmp_out).sents][:2])
                decoded_outputs.append(tmp_out)

            return decoded_outputs


class AppropriatenessPredictorFromPeftInstructLM(AutoRegressivePredictor):
    def __init__(self, model_name, gen_args, task):
        super().__init__(model_name, gen_args, peft_model_name=model_name)
        self.task = task

        if 'extract' in task:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.add_pipe("sentencizer")

    def predict_sample(self, sample):
        with torch.no_grad():
            prompt, post_text = sample
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            post_text_input_ids = self.tokenizer(post_text, return_tensors="pt").input_ids.to("cuda")

            if 'extract' in self.task:
                # split post_text into sentences
                post_text_sents = [sent.text for sent in self.nlp(post_text).sents]
                # get number of tokens in each sentences
                post_text_sents_num_tokens = [len(self.tokenizer(sent, return_tensors="pt").input_ids[0]) for sent in post_text_sents]
                # sum up the number  of tokens in the two longest sentences
                post_text_sents_num_tokens.sort(reverse=True)
                post_text_sents_num_tokens = post_text_sents_num_tokens[:2]
                post_text_sents_num_tokens_sum = sum(post_text_sents_num_tokens)
                self.gen_args["max_new_tokens"] = post_text_sents_num_tokens_sum
            else:
                self.gen_args["max_new_tokens"] = int(len(post_text_input_ids[0]) * 1.25)
                self.gen_args["min_new_tokens"] = int(len(post_text_input_ids[0]) * 0.25)
                print(self.gen_args["max_new_tokens"], self.gen_args["min_new_tokens"])
            outputs = self.model.generate(
                input_ids=prompt_input_ids,
                **self.gen_args,
            )
            decoded_outputs = []
            for output in outputs:
                tmp_out = self.tokenizer.decode(output[len(prompt_input_ids[0]):], skip_special_tokens = True).strip()
                if 'extract' in self.task:
                    tmp_out = ' '.join([sent.text for sent in self.nlp(tmp_out).sents][:2])
                # split of unfinished sentences
                if tmp_out.strip()[-1] not in ['.', '!', '?'] and '.' in tmp_out:
                    tmp_out = ".".join(tmp_out.split('.')[:-1]) + '.'
                decoded_outputs.append(tmp_out)

            return decoded_outputs


def get_baseline_preds():
    args = parse_args()
    if args.dataset == 'argsme':
        ds_path = '../data/inappropriate_arguments_sample_100_argsme.csv'
        df = pd.read_csv(ds_path)
        df['issue'] = df['query']
    if args.dataset == 'argsme_bart':
        ds_path = '../data/results-by-corpus/argsme/summarization/bart.csv'
        df = pd.read_csv(ds_path)
        df['issue'] = df['query']
        df['argument'] = df['bart_gist']
    if args.dataset == 'appropriateness_bart':
        ds_path = '../data/results-by-corpus/appropriateness/summarization/bart.csv'
        df = pd.read_csv(ds_path)
        df['argument'] = df['supervised_summary']
    if args.dataset == 'appropriateness':
        ds_path = '../data/inappropriate_arguments_sample_100_appropriateness.csv'
        df = pd.read_csv(ds_path)

    if args.task1 is not None:
        if args.task1 == 'rewrite':
            INSTRUCT_PATTERN = INSTRUCT_PATTERN_REWRITE
        elif args.task1 == 'extract':
            INSTRUCT_PATTERN = INSTRUCT_PATTERN_EXTRACT
        elif args.task1 == 'rewrite_extract':
            INSTRUCT_PATTERN = INSTRUCT_PATTERN_REWRITE_EXTRACT
        elif args.task1 == 'extract_rewrite':
            INSTRUCT_PATTERN = INSTRUCT_PATTERN_EXTRACT_REWRITE
        df['prompt'] = df[['issue', 'argument']].apply(lambda x: INSTRUCT_PRE + INSTRUCT_PATTERN[:-4].format(x[0][:-1], x[1]), axis=1)

        if 'ppo' in  args.model_name:
            model = AppropriatenessPredictorFromPeftInstructLM(args.model_name, GEN_ARGS, args.task1)
            save_path = '../data/llama_ppo_{}_{}_{}.csv'.format(args.task1, args.dataset, args.model_name.split('/')[-2])
        else:
            model = AppropriatenessPredictorFromInstructLM(args.model_name, GEN_ARGS, args.task1)
            save_path = '../data/llama_{}_{}.csv'.format(args.task1, args.dataset)
        samples = list(zip(df['prompt'], df['argument']))
        model.predict_ds(samples, to_file=True, file_path=save_path, overwrite=True)

    if args.task2 is not None:
        if args.task2 == 'rewrite':
            prev_save_path = '../data/llama_{}_{}.csv'.format('extract', args.dataset)
            save_path = '../data/llama_{}_then_{}_{}.csv'.format('extract', args.task2,  args.dataset)
            INSTRUCT_PATTERN = INSTRUCT_PATTERN_REWRITE
        elif args.task2 == 'extract':
            prev_save_path = '../data/llama_{}_{}.csv'.format('rewrite', args.dataset)
            save_path = '../data/llama_{}_then_{}_{}.csv'.format('rewrite', args.task2,  args.dataset)
            INSTRUCT_PATTERN = INSTRUCT_PATTERN_EXTRACT
        print(prev_save_path)
        pred_df = pd.read_csv(prev_save_path, header=None, sep='\t', names=['0', '1'])
        print(pred_df.head())
        df['argument'] = pred_df['1']
        df['prompt'] = df[['issue', 'argument']].apply(lambda x: INSTRUCT_PRE + INSTRUCT_PATTERN[:-4].format(x[0][:-1], x[1]), axis=1)

        if 'ppo' in  args.model_name:
            model = AppropriatenessPredictorFromPeftInstructLM(args.model_name, GEN_ARGS, args.task2)
        else:
            model = AppropriatenessPredictorFromInstructLM(args.model_name, GEN_ARGS, args.task2)
        samples = list(zip(df['prompt'], df['argument']))
        model.predict_ds(samples, to_file=True, file_path=save_path, overwrite=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--task1', type=str, default=None)
    parser.add_argument('--task2', type=str, default=None)
    parser.add_argument('--dataset', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    get_baseline_preds()
