from huffman_encoder import HuffmanEncoder
from shannon_encoder import ShannonEncoder
from fano_encoder import FanoEncoder

if __name__ == '__main__':
    filename = '../data/GameOfThrones.txt'
    while True:
        while True:
            encode_type = input('请选择使用编码类型(按下对应的数字):\n1.香农编码\n2.费诺编码\n3.赫夫曼编码\n4.退出\n选择:')
            if encode_type in ['1', '2', '3', '4']:
                break
            else:
                print('输入错误')
        if encode_type == '1':
            encoder = ShannonEncoder(filename)
        elif encode_type == '2':
            encoder = FanoEncoder(filename)
        elif encode_type == '3':
            encoder = HuffmanEncoder(filename)
        elif encode_type == '4':
            exit()
        print(encoder.code_word)
        print(f'信源熵:{encoder.source_entropy:.2f}')
        print(f'平均码长:{encoder.mean_length:.2f}')
        print(f'编码效率:{encoder.encode_efficiency*100:.2f}%')
        print(f'码字长度方差:{encoder.var:.2f}')
        print('\n\n')
