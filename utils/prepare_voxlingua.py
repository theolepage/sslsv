import argparse
import os
import pandas as pd
import shutil
from tqdm import tqdm

from prepare_dataset_utils import glob


VOXLINGUA_LABEL_TO_NAME = {
    'ab': 'Abkhazian',
    'af': 'Afrikaans',
    'am': 'Amharic',
    'ar': 'Arabic',
    'as': 'Assamese',
    'az': 'Azerbaijani',
    'ba': 'Bashkir',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'br': 'Breton',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'ceb': 'Cebuano',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'eo': 'Esperanto',
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fo': 'Faroese',
    'fr': 'French',
    'gl': 'Galician',
    'gn': 'Guarani',
    'gu': 'Gujarati',
    'gv': 'Manx',
    'ha': 'Hausa',
    'haw': 'Hawaiian',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'ht': 'Haitian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'ia': 'Interlingua',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'iw': 'Hebrew',
    'ja': 'Japanese',
    'jw': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'km': 'Central Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'la': 'Latin',
    'lb': 'Luxembourgish',
    'ln': 'Lingala',
    'lo': 'Lao',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mg': 'Malagasy',
    'mi': 'Maori',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'mt': 'Maltese',
    'my': 'Burmese',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'nn': 'Norwegian Nynorsk',
    'no': 'Norwegian',
    'oc': 'Occitan',
    'pa': 'Panjabi',
    'pl': 'Polish',
    'ps': 'Pushto',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sa': 'Sanskrit',
    'sco': 'Scots',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sn': 'Shona',
    'so': 'Somali',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'su': 'Sundanese',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'te': 'Telugu',
    'tg': 'Tajik',
    'th': 'Thai',
    'tk': 'Turkmen',
    'tl': 'Tagalog',
    'tr': 'Turkish',
    'tt': 'Tatar',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'war': 'Waray',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zh': 'Mandarin Chinese',
}


def create_smaller_voxlingua107():
    # Sample ~20% of original VoxLingua107

    files = glob('/work2/home/ing2/datasets/voxlingua107/**/*/*.wav')

    d = {}
    new_files = []
    for file in files:
        if file.split('/')[-3] == 'test':
            new_files.append(file)
            continue

        lang = file.split('/')[-2]
        
        if lang not in d: d[lang] = 0
        else: d[lang] += 1

        if d[lang] < 5000:
            new_files.append(file)

    for file in tqdm(new_files):
        new_file = file.replace('voxlingua107', 'voxlingua107_smaller')
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        shutil.copyfile(file, new_file)


def create_voxlingua107_train_csv():
    files = glob('voxlingua107/**/*/*.wav')

    df = pd.DataFrame({
        'File': files,
        'Language': [f.split('/')[-2] for f in files],
        'Set': [f.split('/')[-3] for f in files]
    })

    df.to_csv('voxlingua107_train.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    # create_smaller_voxlingua107()
    create_voxlingua107_train_csv()