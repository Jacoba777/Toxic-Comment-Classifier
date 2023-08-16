from typing import Dict, Set, List, Tuple


def main():
    smooth = 1
    min_freq = 3

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

    toxicity: Dict[str, float] = dict()
    for word in [w for w in toxic_freqs if (nontoxic_freqs.get(w, 0) + toxic_freqs.get(w, 0)) > min_freq]:
        word_freq_toxic = (toxic_freqs.get(word, 0) + smooth) / toxic_count_smooth
        word_freq_nontoxic = (nontoxic_freqs.get(word, 0) + smooth) / nontoxic_count_smooth
        word_toxicity = word_freq_toxic / word_freq_nontoxic
        toxicity[word] = word_toxicity

    sorted_words_by_toxicity = sorted(toxicity, key=toxicity.get, reverse=True)
    sorted_words_by_toxicity = [w for w in sorted_words_by_toxicity if toxicity[w] > 1]

    with open('toxicity_results.txt', 'w', encoding='UTF-8') as f:
        for word in sorted_words_by_toxicity:
            f.write(f'{word} {toxicity[word]:.4f}\n')


if __name__ == '__main__':
    main()
