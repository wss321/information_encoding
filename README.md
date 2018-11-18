# information_encoding
信源编码
# 0.环境
python>=3.6<br/>
numpy<br/>
pandas<br/>
# 1. 数据
data/GameOfThrones.txt<br/>
# 2. 编码
1. Huffman 编码 <br/>
运行 code/huffman_encoder.py <br/>
结果：<br/>
| 字符 | 码字 |
| ------------- | -------------: |
|空格 |           00|
|e   |            010|
|t   |          1010|
|a   |          1001|
|o   |          1000|
|h   |          0111|
|n   |          0110|
|r   |         11111|
|s   |         11110|
|i   |         11100|
|d   |         11011|
|l   |         11000|
|w   |        110101|
|u   |        110100|
|m   |        110011|
|g   |        101111|
|y   |        101110|
|f   |        101101|
|c   |       1110111|
|b   |       1110110|
|k   |       1100101|
|p   |       1100100|
|v   |      11101010|
|T   |      10110010|
|I   |     111010111|
|S   |     111010110|
|H   |     111010010|
|A   |     101100010|
|L   |     101100000|
|W   |    1110100111|
|B   |    1110100011|
|M   |    1110100010|
|J   |    1110100001|
|R   |    1110100000|
|N   |    1011001111|
|D   |    1011001101|
|C   |    1011000110|
|Y   |    1011000010|
|q   |   11101001100|
|G   |   10110011101|
|K   |   10110011100|
|O   |   10110011001|
|F   |   10110011000|
|j   |   10110001111|
|P   |   10110000111|
|E   |   10110000110|
|z   |  111010011011|
|x   |  111010011010|
|V   |  101100011101|
|U   | 1011000111001|
|Q   |10110001110001|
|X  |101100011100001|
|Z  |101100011100000|
信源熵:4.22
平均码长:4.26
编码效率:99.03%