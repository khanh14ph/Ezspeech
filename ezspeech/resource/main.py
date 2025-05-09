from ezspeech.modules.dataset.utils.text import check_end_word
vocab=open("/home4/khanhnd/Ezspeech/ezspeech/resource/vocab_vn.txt").read().splitlines()
new_vocab=[]
for i in vocab:
    if check_end_word(i,vocab)==True:
        new_vocab.append(i+"_")
    else:
        new_vocab.append(i)
with open("/home4/khanhnd/Ezspeech/ezspeech/resource/vocab_vn_suffix.txt","w") as f:
    for i in new_vocab:
        f.write(i+"\n")