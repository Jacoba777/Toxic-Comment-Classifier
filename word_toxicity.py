from typing import Dict, Set, List, Tuple


def main():
    smooth = 10

    with open('dictionaries/counts.txt') as f:
        counts_txt = f.read().split('\n')
    counts: Dict[str, int] = {c[0]: int(c[1]) for c in [line.split(' ') for line in counts_txt]}

    toxic_count_smooth = (counts.get('toxic') + 2*smooth)
    nontoxic_count_smooth = (counts.get('nontoxic') + 2*smooth)

    with open('dictionaries/unigram_nontoxic.txt', encoding="UTF-8") as f:
        nontoxic_freqs_txt = f.read().split('\n')
    nontoxic_freqs: Dict[str, int] = {c[0]: int(c[1]) for c in [line.split(' ') for line in nontoxic_freqs_txt]}

    with open('dictionaries/unigram_toxic.txt', encoding="UTF-8") as f:
        toxic_freqs_txt = f.read().split('\n')
    toxic_freqs: Dict[str, int] = {c[0]: int(c[1]) for c in [line.split(' ') for line in toxic_freqs_txt]}

    words: Dict[str, float] = dict()
    for word in toxic_freqs:
        word_freq_toxic = (toxic_freqs.get(word, 0) + smooth) / toxic_count_smooth
        word_freq_nontoxic = (nontoxic_freqs.get(word, 0) + smooth) / nontoxic_count_smooth
        word_toxicity = word_freq_toxic / word_freq_nontoxic
        words[word] = word_toxicity

    sorted_words_by_toxicity = sorted(words, key=words.get, reverse=True)

    with open('toxicity_results.txt', 'w', encoding='UTF-8') as f:
        for word in sorted_words_by_toxicity:
            f.write(f'{word} {words[word]:.4f}\n')


if __name__ == '__main__':
    main()
