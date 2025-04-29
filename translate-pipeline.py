from transformers import pipeline
from transformers import AutoModel,AutoModelForSeq2SeqLM, AutoTokenizer
from sys import argv

from datetime import datetime

#Args
# 1: model path
# 2: input file
# 3: src_lang
# 4: tgt_lang
# 5: number of generated translations

checkpoint = argv[1]
source_file = argv[2]
src_lang = argv[3]
tgt_lang = argv[4]
method = argv[5]
translations_per_source = int(argv[6])

batch_size = 2

def data():
    with open(source_file) as f:
        for line in f:
            yield line.rstrip("\n")


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.src_lang = src_lang

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")
generator = None


if method == "beam_search":
    generator = pipeline(task="translation", model=model, tokenizer=tokenizer,
                    src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size, do_sample=False, num_beams=translations_per_source, num_return_sequences=translations_per_source, max_new_tokens=128, device="cuda")

if method == "diverse_beam_search":
    generator = pipeline(task="translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size, do_sample=False,
    num_beams=translations_per_source, num_beam_groups=translations_per_source, num_return_sequences=translations_per_source, diversity_penalty=0.5, device="cuda")

if method == "top_p":
    generator = pipeline(task="translation", model=model, tokenizer=tokenizer,
                    src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size, do_sample=True, top_p=0.7, num_return_sequences=translations_per_source, max_new_tokens=128, device="cuda")

if method == "top_k":
    generator = pipeline(task="translation", model=model, tokenizer=tokenizer,
                    src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size, do_sample=True, top_k=10, num_return_sequences=translations_per_source, max_new_tokens=128, device="cuda")

if method == "evaluate":
    generator = pipeline(task="translation", model=model, tokenizer=tokenizer,
                    src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size, num_beams=5, device="cuda")


for out in generator(data()):
    if method == "evaluate":
        print(out[0]["translation_text"])
    else:
        for translation in out:
            print(translation["translation_text"])