import pandas as pd
from ezspeech.utils.common import load_dataset,save_dataset
lst=[]
df=pd.read_csv("/home3/khanhnd/FINAL_Youtube.csv",sep="\t")
for idx,i in df.iterrows():
    data={"audio_filepath":i["audio_path"],"duration":i["duration"],"text":i["transcription"]}
    lst.append(data)
save_dataset(lst,"/home3/khanhnd/FINAL_Youtube.jsonl")