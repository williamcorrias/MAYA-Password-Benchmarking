import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def save_figures(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")


def general_structure(fig_size=None, x_caption="", y_caption="", x_lim=None, y_lim=None, title=None, y_log_scale=False,
                      x_log_scale=False, y_ticks=None, x_ticks=None, grid=True, fontsize=12, labelsize=11):

    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, ax = plt.subplots()

    ax.set_xlabel(x_caption, fontsize=fontsize)
    ax.set_ylabel(y_caption, fontsize=fontsize)

    ax.tick_params(axis="both", which='major', labelsize=labelsize)

    if y_log_scale:
        ax.set_yscale('log')

    if x_log_scale:
        plt.xscale('log')

    if x_lim is not None:
        plt.xlim(x_lim)

    if y_lim is not None:
        plt.ylim(y_lim)

    if x_ticks is not None:
        plt.xticks(x_ticks[0], x_ticks[1])

    if y_ticks is not None:
        plt.yticks(y_ticks[0], y_ticks[1])

    if title is not None:
        plt.title(title, fontsize=fontsize)

    if grid:
        ax.grid(True)

    return plt, fig, ax


def plot_distribution(fig_size=None, x_data=None, y_data=None, x_caption="", y_caption="", x_lim=None, y_lim=None, title="",
                      y_log_scale=False, x_log_scale=False, y_ticks=None, x_ticks=None, color='blue', fontsize=12,
                      labelsize=11, dest_path=None, second_data={}):

    plt, fig, ax = general_structure(fig_size=fig_size,
                                     x_caption=x_caption,
                                     y_caption=y_caption,
                                     x_lim=x_lim,
                                     y_lim=y_lim,
                                     title=title,
                                     y_log_scale=y_log_scale,
                                     x_log_scale=x_log_scale,
                                     fontsize=fontsize,
                                     labelsize=labelsize)

    ax.plot(x_data, y_data, color=color)

    if second_data:
        ax2 = ax.twinx()
        ax2.set_ylabel(second_data["label"], fontsize=fontsize)
        ax2.tick_params(axis='y', labelsize=labelsize)
        ax2.set_ylim(0, 10)
        ax2.plot(x_data, second_data["y_data"], color=second_data["color"], label=second_data["label"], linestyle='--')

    fig.tight_layout()
    plt.show()

    if dest_path is not None:
        save_figures(fig, path=dest_path)


def multiline_graph(fig_size=None, x_data=None, y_data=None, y_errors=None, x_caption="", y_caption="", x_lim=None, y_lim=None, title="",
                    y_log_scale=False, x_log_scale=False, y_ticks=None, x_ticks=None, labels=None, color='blue',
                    legend_params={}, fontsize=12, labelsize=11, dest_path=None):

    plt, fig, ax = general_structure(fig_size=fig_size,
                                     x_caption=x_caption,
                                     y_caption=y_caption,
                                     x_lim=x_lim,
                                     y_lim=y_lim,
                                     title=title,
                                     y_log_scale=y_log_scale,
                                     x_log_scale=x_log_scale,
                                     y_ticks=y_ticks,
                                     x_ticks=x_ticks,
                                     fontsize=fontsize,
                                     labelsize=labelsize,)

    for i, data in enumerate(y_data):
        if y_errors is not None:
            ax.errorbar(x_data, data, yerr=y_errors[i], label=labels[i], marker='D',
                        capsize=8, capthick=3, elinewidth=3, fmt='-o', linestyle='-', zorder=15)
        else:
            ax.plot(x_data, data, label=labels[i], marker='o', markersize=4)

    plt.legend(**legend_params)
    fig.tight_layout()
    plt.show()

    if dest_path is not None:
        save_figures(fig, path=dest_path)


def bar_graph(fig_size=None, x_data=None, y_data=None, y_errors=None, x_caption="", y_caption="", x_lim=None, y_lim=None, title="",
                    y_log_scale=False, x_log_scale=False, y_ticks=None, x_ticks=None, labels=None, color='blue',
                    legend_params={}, fontsize=12, labelsize=11, dest_path=None, bar_width=0.75, multiplier=1.15, point_dict=None):

    plt, fig, ax = general_structure(fig_size=fig_size,
                                     x_caption=x_caption,
                                     y_caption=y_caption,
                                     x_lim=x_lim,
                                     y_lim=y_lim,
                                     title=title,
                                     y_log_scale=y_log_scale,
                                     x_log_scale=x_log_scale,
                                     y_ticks=y_ticks,
                                     x_ticks=x_ticks,
                                     fontsize=fontsize,
                                     labelsize=labelsize,
                                     )


    bar_width = bar_width / len(x_data)
    plt.grid(axis='x')

    for i, data in enumerate(y_data):
        x_positions = np.array(x_data) + (i - len(y_data) / 2) * bar_width * multiplier

        bars = ax.bar(x_positions, data, width=bar_width, label=labels[i], edgecolor="black", alpha=0.7)

        if point_dict is not None:
            for j, bar in enumerate(bars):
                percentage = point_dict[str(i)][j]
                ax.plot(bar.get_x() + bar.get_width() / 2, percentage, 'D', color='black', markersize=5)

        if y_errors is not None:
            ax.errorbar(x_positions, data, yerr=y_errors[i], fmt='none', ecolor='black', capsize=1, elinewidth=0.2)

    plt.legend(**legend_params)
    fig.tight_layout()
    plt.show()

    if dest_path is not None:
        save_figures(fig, path=dest_path)


def heatmap_table(data, items, color="Oranges", vmin=None, vmax=None, dest_path=None, cbar_kws=None):
    models_name = {
        'fla': 'FLA',
        'passgan': 'PassGAN',
        'plrgan': 'PLR',
        'passflow': 'PassFlow',
        'passgpt': 'PassGPT',
        'vgpt2': 'VGPT2',
    }

    labels = [models_name.get(item.lower(), item) for item in items]

    matrix = np.zeros((len(items), len(items)))

    lowercase_items = [item.lower() for item in items]

    for key, value in data.items():
        item1, item2 = key.split("-")
        item1, item2 = item1.strip().lower(), item2.strip().lower()
        i, j = lowercase_items.index(item1), lowercase_items.index(item2)
        matrix[i, j] = value
        matrix[j, i] = value

    df = pd.DataFrame(matrix, index=items, columns=items)
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.92)
    sns.heatmap(df, annot=True, fmt=".4f", cmap=color, linewidths=1, yticklabels=labels, xticklabels=labels, cbar=True, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws)
    plt.show()

    if dest_path is not None:
        save_figures(fig, path=dest_path)

def tsne_plot(embeddings, labels, keys, data_paths):
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    num_datasets = len(keys)
    colors = sns.color_palette("husl", num_datasets)

    plt.figure()

    data_paths = [os.path.basename(path) for path in data_paths]

    for dataset_idx, color in zip(keys, colors):
        mask = labels == dataset_idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    label=f"{os.path.basename(data_paths[int(dataset_idx)])}", color=color, s=0.3)

    plt.legend(markerscale=7)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)

    dest_folder = "figures/rq5"
    os.makedirs(dest_folder, exist_ok=True)

    save_path = os.path.join(dest_folder, "-".join(data_paths) + ".png")
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)
    plt.show()
