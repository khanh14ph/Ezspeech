from viphoneme import vi2IPA
import eng_to_ipa as ipa
from tqdm import tqdm
from dp.phonemizer import Phonemizer
phonemizer = Phonemizer.from_checkpoint('/Users/khanh/dev/asr_dev/en_us_cmudict_ipa_forward.pt')
print(vi2IPA("xin chào"))