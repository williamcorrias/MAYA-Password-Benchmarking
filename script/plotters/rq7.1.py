baseline = {
    'CNN': (0.4838, 15.5416),
    'aP': (1 - 0.9684, 1 - 0.7851),
    'bR': (1 - 0.9434, 1 - 0.1380),
    'auth': (0.5363, 0.6348),
    'IMD': (0.4026, 8.5714),
    'MTopDiv': (18.6986, 244.1587)
}

results = {
    'PassGAN': {
        'CNN': 2.9823,
        'aP': 1 - 0.9318,
        'bR': 1 - 0.9098,
        'auth': 0.5506,
        'IMD': 5.7614,
        'MTopDiv': 22.6687,
    },
    'PLRGAN': {
        'CNN': 1.4374,
        'aP': 1 - 0.9766,
        'bR': 1 - 0.9122,
        'auth': 0.5476,
        'IMD': 0.6648,
        'MTopDiv': 19.9084,
    },
    'PassFlow': {
        'CNN': 9.0067,
        'aP': 1 - 0.8558,
        'bR': 1 - 0.5176,
        'auth': 0.5521,
        'IMD': 16.8088,
        'MTopDiv': 101.3500,
    },
    'PassGPT': {
        'CNN': 0.9298,
        'aP': 1 - 0.9628,
        'bR': 1 - 0.9352,
        'auth': 0.5423,
        'IMD': 0.3297,
        'MTopDiv': 18.9544,
    },
    'VGPT2': {
        'CNN': 4.8891,
        'aP': 1 - 0.8711,
        'bR': 1 - 0.6694,
        'auth': 0.5412,
        'IMD': 11.4420,
        'MTopDiv': 46.0624,
    },
    'FLA': {
        'CNN': 2.4056,
        'aP': 1 - 0.9974,
        'bR': 1 - 0.9538,
        'auth': 0.5672,
        'IMD': 14.4589,
        'MTopDiv': 20.0932,
    },
}


def normalize_values():
    for model in results:
        for metric in results[model]:
            results[model][metric] = ((results[model][metric] - baseline[metric][0])
                                      / (baseline[metric][1] - baseline[metric][0]) * 100)

    return results


def main():
    results = normalize_values()
    print(results)

if __name__ == "__main__":
    main()