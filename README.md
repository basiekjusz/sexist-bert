# sexist-bert

This repository contains the code for the paper _Gender Bias in Language Models trained on the Polish language_ by [Jan Piotrowski](jf.piotrowsk@student.uw.edu.pl), [Jan Busz](j.busz@student.uw.edu.pl), [Dominik Wiśniewski](d.wisniewsk2@student.uw.edu.pl).

## Abstract

This study examines gender bias in Polish language models, HerBERT and LLaMA-2, using sentiment analysis on a dataset focusing on American actors and actresses. Translated using GPT-4, our analysis reveals minor sentiment differences between male and female contexts in the models. However, these differences are not conclusive indicators of bias, as they also appear in original sentences. LLaMA-2 models display a more neutral tone, suggesting a lower likelihood of gender bias. The study highlights the need for further research due to the limitations of sentiment analysis as a sole measure of gender bias.

## Project structure

```
.
├── bin       # contains the scripts, in particular the script used to translate the dataset to polish
├── data      # contains the data used, both the original, translated dataset and analysis results
├── notebooks # contains the notebooks used for the analysis
└── plots     # plots generated by the notebooks
```

## Running the code

### Requirements

The code was tested on Python 3.11.6.

You can create a virtual environment to run the code in:

```
python -m venv venv
source venv/bin/activate
```

The required packages are listed in `requirements.txt`. To install them, run:

```
pip install -r requirements.txt
```

### Translating the dataset

The translation script uses GPT-4 through OpenAI API to translate the dataset. To use it, you need to have an OpenAI API key. You can get one [here](https://platform.openai.com/api-keys).

The script uses the `OPENAI_API_KEY` environment variable to authenticate. To set it, run:

```
export OPENAI_API_KEY=<your key>
```

To translate the dataset, run:

```
python bin/translate_bold.py
```

The script will create a `data/pl` directory and save the translated dataset there. You can also set the output path by changing the `PL_PATH_PREFIX` variable in the script.

### Running the notebooks

The notebooks are located in the `notebooks` directory. They're self-contained and can be run in any order. The notebooks will save the results of the analysis in the `data` directory.