from ezspeech.utils.common import load_dataset, save_dataset
a=[(),()]
for i in a:
    temp=load_dataset(i[0])
    for v in temp:
        i["lan_id"]=i[1]
    save_dataset(i[0])