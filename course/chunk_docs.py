import re
import os
import hashlib
from openai import OpenAI
from tqdm.auto import tqdm


with open("./openai_api_key.txt", "r") as f:
    openai_api_key = f.read().strip()

openai_client = OpenAI(api_key=openai_api_key)

cache_dir = "./llm_cache"

prompt_template = """
Split the provided document into logical sections
that make sense for a Q&A system.

Each section should be self-contained and cover
a specific topic or concept.

<DOCUMENT>
{document}
</DOCUMENT>

Use this format:

## Section Name

Section content with all relevant details

---

## Another Section Name

Another section content

---
""".strip()

######################
### Sliding Window ###
######################

def sliding_window(doc, size=2000, step=1000):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    doc_copy = doc.copy()
    doc_content = doc_copy.pop("content")

    n = len(doc_content)
    chunks = []
    for i in range(0, n, step):
        section = doc_content[i:i+size]
        chunk = {"start": i, "section": section}
        chunk.update(doc_copy)
        chunks.append(chunk)
        if i + size >= n:
            break

    return chunks


def chunk_docs_using_sliding_window(docs, size=2000, step=1000):
    chunks = []
    for doc in tqdm(docs, desc="Chunking docs using sliding window"):
        chunks.extend(sliding_window(doc, size, step))
    return chunks

######################
## Split by Section ##
######################

def split_markdown_by_level(text, level=2):
    """
    Split markdown text by a specific header level.
    
    :param text: Markdown text as a string
    :param level: Header level to split on
    :return: List of sections as strings
    """
    # This regex matches markdown headers
    # For level 2, it matches lines starting with "## "
    header_pattern = r'^(#{' + str(level) + r'} )(.+)$'
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        # We step by 3 because regex.split() with
        # capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i+1]  # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i+2 < len(parts):
            content = parts[i+2].strip()

        if content:
            section = f'{header}\n\n{content}'
        else:
            section = header
        sections.append(section)

    return sections


def chunk_docs_by_section(docs, level=2):
    chunks = []
    for doc in tqdm(docs, desc="Chunking docs by section"):
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        sections = split_markdown_by_level(doc_content, level)

        for section in sections:
            section_copy = doc_copy.copy()
            section_copy["section"] = section
            chunks.extend(section_copy)
    return chunks

######################
#### LLM Chunking ####
######################

def calc_llm_cache_key(prompt):
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    return cache_key


def get_cache_path(prompt):
    cache_key = calc_llm_cache_key(prompt)
    cache_file = os.path.join(cache_dir, f"{cache_key}.txt")
    return cache_file


def read_from_cache(prompt):
    cache_file = get_cache_path(prompt)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()
    return None


def write_to_cache(prompt, text):
    cache_file = get_cache_path(prompt)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(text)

def llm(prompt, model='gpt-4o-mini', force_refresh=False):
    if not force_refresh:
        cached_text = read_from_cache(prompt)
        if cached_text:
            return cached_text


    messages = [
        {"role": "user", "content": prompt}
    ]

    response = openai_client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )

    out_text = response.output_text

    write_to_cache(prompt, out_text)

    return out_text


def intelligent_chunking(text):
    prompt = prompt_template.format(document=text)
    response = llm(prompt)
    sections = response.split("---")
    sections = [s.strip() for s in sections if s.strip()]
    return sections


def chunk_docs_using_llm(docs):
    chunks = []
    for doc in tqdm(docs, desc="Chunking docs using LLM"):
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        sections = intelligent_chunking(doc_content)

        for section in sections:
            section_copy = doc_copy.copy()
            section_copy["section"] = section
            chunks.append(section_copy)
    return chunks


if __name__ == "__main__":
    pass
