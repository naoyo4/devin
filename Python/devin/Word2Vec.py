
import logging
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess # 簡単な前処理用

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [
    "this is the first sentence for word2vec",
    "this is the second sentence",
    "yet another sentence",
    "one more sentence",
    "and the final sentence"
]

processed_sentences = [simple_preprocess(s) for s in sentences]

print("処理後の学習データ:")
print(processed_sentences)

print("\nWord2Vecモデルの学習を開始します...")
model = Word2Vec(sentences=processed_sentences,
                 vector_size=100,  # 100次元のベクトルにする
                 window=5,       # 前後5単語をコンテキストとする
                 min_count=1,    # 1回以上出現した単語はすべて考慮する
                 workers=4,      # 4スレッドで並列処理する
                 sg=0)           # CBOWモデルを使用 (sg=1 でSkip-gram)
print("学習が完了しました。")



vocabulary = model.wv.index_to_key
print(f"\n学習された語彙（一部）: {vocabulary[:10]}...") # 先頭10個表示
print(f"語彙数: {len(vocabulary)}")

try:
    vector_sentence = model.wv['sentence']
    print(f"\n'sentence' のベクトル (最初の10次元): {vector_sentence[:10]}...")
    print(f"'sentence' のベクトルの次元数: {len(vector_sentence)}")
except KeyError:
    print("\n'sentence' は語彙に含まれていません。")

try:
    similar_words = model.wv.most_similar('sentence', topn=5)
    print(f"\n'sentence' に類似している単語 Top 5: {similar_words}")
except KeyError:
    print("\n'sentence' は語彙に含まれていません。")

try:
    similar_words_this = model.wv.most_similar('this', topn=3)
    print(f"\n'this' に類似している単語 Top 3: {similar_words_this}")
except KeyError:
    print("\n'this' は語彙に含まれていません。")

try:
    similarity = model.wv.similarity('first', 'second')
    print(f"\n'first' と 'second' の類似度: {similarity:.4f}")
except KeyError as e:
    print(f"\n単語が見つかりません: {e}")
