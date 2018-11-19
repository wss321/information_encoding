# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def dec2bin(x):
    """十进制转二进制"""
    x_int = int(x)
    int_part = bin(x_int).replace('0b', '')
    x -= int(x)
    decimal_part = []
    while x:
        x *= 2
        decimal_part.append('1' if x >= 1. else '0')
        x -= int(x)
    decimal_part = ''.join(decimal_part)
    return {'int_part': int_part, 'decimal_part': decimal_part}


class ShannonEncoder(object):
    """香农编码"""

    def __init__(self, filename='../data/GameOfThrones.txt'):
        self.filename = filename
        self.content = self.__load()
        self.num_and_freq = self.__count_sym_frequency()
        self.frequency = self.num_and_freq[['frequency']]
        self.encode_table = self.__encode()
        self.code_word = self.encode_table['code_word'].to_frame()
        self.source_entropy = self.__entropy()
        self.mean_length = self.__mean_code_length()
        self.encode_efficiency = self.__encode_rate()

    def __load(self):
        """加载文件"""
        with open(self.filename, 'r', encoding='gbk') as f:
            lines = f.readlines()
        return ''.join(lines)

    def __count_sym_num(self):
        """计算每个符号数量"""
        sym = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)] + [' ']
        sym_num = np.zeros(len(sym), dtype=np.int64)
        for letter in self.content:
            if letter in sym:
                sym_num[sym.index(letter)] += 1

        df = pd.DataFrame(sym_num, index=sym)
        df.columns = ['number']
        return df

    def __count_sym_frequency(self):
        """计算符号频率"""
        df = self.__count_sym_num()
        df = df.sort_values(by='number', ascending=False)
        number = df[['number']].values
        number = number / number.sum()
        df['frequency'] = np.squeeze(number)
        return df

    def __encode(self):
        """计算累加概率、码长，并算出码字"""
        # 累加概率
        result = self.frequency.copy()
        zero_freq_item = result[result['frequency'] == 0].index.tolist()
        result.drop(zero_freq_item, inplace=True)
        temp = np.cumsum(result[['frequency']].values)
        cum_sum = np.zeros(temp.shape, dtype=np.float64)
        cum_sum[1:] = temp[:-1]
        result['cum_prob'] = cum_sum
        # 码长
        code_len = np.ceil(- np.log2(result[['frequency']].values)).astype(np.int32).squeeze()
        result['code_len'] = code_len

        # 码字
        cum_prob = np.asarray(result['cum_prob'].values)
        code_word = []
        for idx in range(len(cum_prob)):
            decimal_part = dec2bin(cum_prob[idx])['decimal_part']
            while len(decimal_part) < code_len[idx]:
                decimal_part += '0'
            if len(decimal_part) > code_len[idx]:
                decimal_part = decimal_part[:code_len[idx]]
            code_word.append(decimal_part)
        result['code_word'] = code_word
        return result

    def __mean_code_length(self):
        """计算平均码长"""
        p = self.encode_table.copy()['frequency']
        code_len = self.encode_table.copy()['code_len']
        df = pd.concat([p, code_len], axis=1)
        df['l_i'] = df.apply(lambda x: x['frequency'] * x['code_len'], axis=1)
        return df['l_i'].sum()

    def __entropy(self):
        """计算信息熵"""
        p = self.encode_table.copy()['frequency'].values
        for idx in range(len(p)):
            if p[idx] != 0:
                p[idx] = p[idx] * np.log2(p[idx])
        return - p.sum()

    def __encode_rate(self):
        """计算编码效率"""
        return self.source_entropy / self.mean_length


if __name__ == '__main__':
    encoder = ShannonEncoder(filename='../data/GameOfThrones.txt')
    print(encoder.code_word)

    print(f'信源熵:{encoder.source_entropy:.2f}')
    print(f'平均码长:{encoder.mean_length:.2f}')
    print(f'编码效率:{encoder.encode_efficiency*100:.2f}%')
