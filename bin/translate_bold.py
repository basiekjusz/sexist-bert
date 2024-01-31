# The purpose of this script is to translate the BOLD dataset from english to polish.
# It's done using the GPT-4 API from OpenAI.
# To run the script you need to set the OPENAI_API_KEY environment variable.
# The script is run from the root directory of the project and outputs the translated
# dataset to the directory set by `PL_PATH_PREFIX` variable.

from openai import OpenAI
import json

EN_PATH_PREFIX = "data/en/"
PL_PATH_PREFIX = "data/pl/"
TYPES = [("prompts", "prompt")]
PATH_TEMPLATES = [
    "{}/gender_{}.json",
]


def _translate_related_wiki_prompt(client: OpenAI, wiki: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are a translator that translates text from english to polish.
                Translate the following two texts to ensure their correspondence.
                Translate both texts to the same target language and verify that the translated
                versions still maintain a coherent and corresponding relationship.
                Make sure that that the shorter version is exactly the same as the longer version.
                Return ONLY the translated texts.""",
            },
            {"role": "user", "content": wiki + "\n" + prompt},
        ],
    )

    response = response.choices[0].message.content

    with open("data/debug.log", "a") as f:
        input = "[INPUT]: " + wiki + " | " + prompt
        output = "[OUTPUT]: " + response
        f.write("\n" + input + output)

    wiki_response = response.split("\n")[0]
    prompt_response = (
        response.split("\n")[1] if len(response.split("\n")) > 1 else wiki_response
    )

    if len(response.split("\n")) != 2:
        with open("data/debug.log", "a") as f:
            f.write("ERROR")

    return wiki_response, prompt_response


def main():
    client = OpenAI()

    for path_template in PATH_TEMPLATES:
        read_path_wiki = EN_PATH_PREFIX + path_template.format("wikipedia", "wiki")
        read_path_prompts = EN_PATH_PREFIX + path_template.format("prompts", "prompt")

        with open(read_path_wiki, "r") as f:
            wiki_data = json.load(f)
        with open(read_path_prompts, "r") as f:
            prompt_data = json.load(f)

        for wiki_class_name, prompt_class_name in zip(
            wiki_data.keys(), prompt_data.keys()
        ):
            assert wiki_class_name == prompt_class_name

            wiki_class = wiki_data[wiki_class_name]
            prompt_class = prompt_data[prompt_class_name]

            for wiki_example, prompt_example in zip(
                wiki_class.keys(), prompt_class.keys()
            ):
                assert wiki_example == prompt_example

                wiki_itemset = wiki_class[wiki_example]
                prompt_itemset = prompt_class[prompt_example]

                assert len(wiki_itemset) == len(prompt_itemset)

                translated_wiki_itemset = []
                translated_prompt_itemset = []
                for wiki_item, prompt_item in zip(wiki_itemset, prompt_itemset):
                    translated_wiki, translated_prompt = _translate_related_wiki_prompt(
                        client, wiki_item, prompt_item
                    )

                    translated_wiki_itemset.append(translated_wiki)
                    translated_prompt_itemset.append(translated_prompt)

                wiki_class[wiki_example] = translated_wiki_itemset
                prompt_class[prompt_example] = translated_prompt_itemset

        save_path_wiki = PL_PATH_PREFIX + path_template.format("wikipedia", "wiki")
        save_path_prompts = PL_PATH_PREFIX + path_template.format("prompts", "prompt")

        with open(save_path_wiki, "w") as f:
            json.dump(wiki_data, f)
        with open(save_path_prompts, "w") as f:
            json.dump(prompt_data, f)


if __name__ == "__main__":
    main()
