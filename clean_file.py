import re

from bs4 import BeautifulSoup
from nltk import sent_tokenize


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_punctuations(text):
    p = re.compile(r'(\r)|(\t)|(\')|(\u00A9)|(["«»#$%&()*+/:;<=>@\[\\\]^_`{|}~])', re.IGNORECASE)
    return re.sub(p, '', text)


def remove_other(text):
    # 2017-01-1818.01.20173040
    text = re.sub('Facebook|Twitter|Google|Pinterest|Vkontakte|Telegram', '', text)
    return re.sub('[0-9]{4}-[0-9]{2}-[0-9]{4}.[0-9]{2}.[0-9]{8}', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_punctuations(text)
    text = remove_other(text)
    return text


file = open('crawler/kaz_nur_kz.txt')
raw = file.read()
# cleaned_text = denoise_text(raw)
cleaned_text = sent_tokenize(raw)
file.close()

new_file = open('cleaned/kaz_nur_kz.txt', 'w')
# new_file.writelines(cleaned_text)
for item in cleaned_text:
    new_file.write("%s\n" % item)
new_file.close()
