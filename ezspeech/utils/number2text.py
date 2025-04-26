import random
import sys
import math

number = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def create_number(length=2):
    length = int(length)
    upper = min(10**length, 200) + 1
    no = random.randint(0, upper)
    sent = lex_of_number(no, byNo=False)
    return sent


####################################################################
#       lex_of_number and lex_of_number_3
#       obatained from 44:/data/tulm7/ePass/data/number_lex.py
####################################################################


def lex_of_number(no: str, byNo=None, withUnion=None):
    length = len(str(no))
    if length == 0:
        return ""
    if byNo is None:
        byNo = random.random() < 0.85

    if withUnion is None and (not byNo):
        withUnion = random.random() < 0.9

    line = ""
    if (not byNo) and length > 1:
        no = str(no)
        no_group = math.ceil(length / 3)

        split_3 = ["", "nghìn", "triệu", "tỷ"]
        for i in range(no_group):
            s = max(0, length - (i + 1) * 3)
            e = min(s + 3, length - i * 3)
            if int(no[s:e]) < 1:
                continue
            if i >= 4:
                line = lex_of_number(no[:e], byNo, withUnion) + " tỷ " + line
            line = (
                lex_of_number_3(no[s:e], byNo, withUnion)
                + " "
                + split_3[i]
                + " "
                + line
            )
    else:
        for i in str(no):
            line += " " + number[int(i)]
    return line.strip()


def lex_of_number_3(no: str, byNo=None, withUnion=None):
    line = ""
    i = int(no) % 1000
    length = len(str(no))
    if length > 2:
        line += number[i // 100] + " trăm"
        i = i % 100
    if length > 1:
        if i // 10 == 0 and i % 10 != 0:
            if length > 2:
                line += " " + random.choice(["linh", "lẻ"])
            if i % 10 == 4:
                line += " " + random.choice(["tư", "bốn"])
            else:
                line += " " + number[i % 10]
        elif i // 10 == 1:
            line += " mười"
            if i % 10 == 5:
                line += " lăm"
            elif i % 10 != 0:
                line += " " + number[i % 10]
        elif i // 10 > 1:
            line += " " + number[i // 10]
            if i % 10 == 0:
                line += " mươi"
            else:
                if withUnion:
                    line += " mươi"
                if i % 10 == 4:
                    line += random.choice([" tư", " bốn"])
                elif i % 10 == 5:
                    line += " lăm"
                elif i % 10 == 1:
                    line += " mốt"
                else:
                    line += " " + number[i % 10]
    if length == 1:
        line = number[i % 10]
    return line.strip()





if __name__ == "__main__":
    print(number2text("xin chao 23 nhe"))
