from ezspeech.utils.common import load_dataset,save_dataset

from ezspeech.modules.data.utils.text import Tokenizer,normalize

tokenizer=Tokenizer("/scratch/midway3/khanhnd/Ezspeech/tokenizer/vi/tokenizer.model")



a=load_dataset("/scratch/midway3/khanhnd/data/metadata/youtube_norm.jsonl")
for i in a:
    i["text"]=normalize(i["text"])

# b=[]
# for i in a:
#     text=i["text"].strip()
#     temp=tokenizer.decode(tokenizer.encode(text)).strip()
#     if temp==text:
#         b.append(i)
    # else:
    #     print(temp)
    #     print(text)
    #     print("______")
# save_dataset(b,"/scratch/midway3/khanhnd/data/metadata/youtube_norm.jsonl")
text_lst=[i["text"] for i in a]

word_lst=[]
for i in text_lst:
    if "smoothini" in i:
        print(i)
    word_lst.extend(i.split() )
word_set=set(word_lst)
with open("/scratch/midway3/khanhnd/Ezspeech/asset/lexicon.txt","w") as f:
    for i in word_set:
        
        hehe=" ".join(tokenizer.decode_list(tokenizer.encode(i)))
        if hehe!="":
            f.write(i+"\t"+str(hehe)+"\n")

