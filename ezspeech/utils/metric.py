from tqdm import tqdm
from lightspeech.datas.text import build_lexicon
from torchmetrics.functional.text import word_error_rate


def get_oov_wer(hyps, refs):
    wer = word_error_rate(hyps, refs).item() * 100

    lexicon = build_lexicon()

    oov_hyps = []
    oov_refs = []

    for hyp, ref in tqdm(zip(hyps, refs)):
        oov_flag = 0
        for word in ref.split():
            if word not in lexicon.keys():
                oov_flag += 1

        if oov_flag:
            ref_words = ref.split()
            oov_ref = " ".join(
                [word for word in ref_words if word not in lexicon.keys()]
            )
            # print(oov_ref)
            hyp_words = hyp.split()
            oov_hyp = " ".join(
                [word for word in hyp_words if word not in lexicon.keys()]
            )

            oov_hyps.append(oov_hyp)
            oov_refs.append(oov_ref)

    oov_wer = 100 * word_error_rate(oov_hyps, oov_refs).item()
    return wer, oov_wer
