import os
from os.path import join

DATA_DIR = '/workdir/data/'
TRAIN_SIMPLIFIED = join(DATA_DIR, 'train_simplified')
TRAIN_TIME = join(DATA_DIR, 'train_time')
TEST_SIMPLIFIED_PATH = join(DATA_DIR, 'test_simplified.csv')
BASE_SIZE_SIMPLIFIED = 256
SAMPLE_SUBMISSION = join(DATA_DIR, 'sample_submission.csv')

CLASSES = sorted([p[:-4] for p in os.listdir(TRAIN_TIME) if p.endswith('.csv')])
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
CLASS_TO_CSV_PATH = {cls: join(TRAIN_TIME, cls+'.csv') for cls in CLASSES}

COUNTRIES = [
    'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AN', 'AO', 'AR', 'AS',
    'AT', 'AU', 'AW', 'AX', 'AZ', 'BA', 'BB', 'BD', 'BE', 'BF', 'BG',
    'BH', 'BI', 'BJ', 'BM', 'BN', 'BO', 'BR', 'BS', 'BT', 'BU', 'BW',
    'BY', 'BZ', 'CA', 'CD', 'CF', 'CG', 'CH', 'CI', 'CL', 'CM', 'CN',
    'CO', 'CR', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM',
    'DO', 'DZ', 'EC', 'EE', 'EG', 'ES', 'ET', 'FI', 'FJ', 'FM', 'FO',
    'FR', 'GA', 'GB', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GL', 'GM',
    'GN', 'GP', 'GR', 'GT', 'GU', 'GY', 'HK', 'HN', 'HR', 'HT', 'HU',
    'ID', 'IE', 'IL', 'IM', 'IN', 'IQ', 'IR', 'IS', 'IT', 'JE', 'JM',
    'JO', 'JP', 'KE', 'KG', 'KH', 'KN', 'KR', 'KW', 'KY', 'KZ', 'LA',
    'LB', 'LC', 'LI', 'LK', 'LR', 'LS', 'LT', 'LU', 'LV', 'LY', 'MA',
    'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MK', 'ML', 'MM', 'MN', 'MO',
    'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ',
    'NC', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PA', 'PE',
    'PF', 'PG', 'PH', 'PK', 'PL', 'PM', 'PR', 'PS', 'PT', 'PW', 'PY',
    'QA', 'RE', 'RO', 'RS', 'RU', 'RW', 'SA', 'SC', 'SE', 'SG', 'SI',
    'SJ', 'SK', 'SM', 'SN', 'SO', 'SR', 'SS', 'ST', 'SV', 'SX', 'SZ',
    'TC', 'TG', 'TH', 'TJ', 'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TW',
    'TZ', 'UA', 'UG', 'US', 'UY', 'UZ', 'VC', 'VE', 'VG', 'VI', 'VN',
    'VU', 'WS', 'YE', 'YT', 'ZA', 'ZM', 'ZW', 'ZZ', 'nan'
]
COUNTRY_TO_IDX = {c: idx for idx, c in enumerate(COUNTRIES)}
IDX_TO_COUNTRY = {idx: c for c, idx in COUNTRY_TO_IDX.items()}
