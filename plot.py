__author__ = 'lqrz'

from par2vec import load_all_models
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from corpora import get_utterances_from_file
from collections import defaultdict

def plotGraph(samples, n_samples, tags, dimensions):

    colours = ['blue', 'red', 'green', 'yellow', 'black']
    n_tags = len(tags)

    if dimensions == '2D':
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        for i in range(n_tags):
            plt.plot(sklearn_transf[i*n_samples:(i+1)*n_samples,0],sklearn_transf[i*n_samples:(i+1)*n_samples,1],\
                 'o', markersize=7, color=colours[i], alpha=0.5, label=tags[i])

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    #     plt.xlim([-4,4])
    #     plt.ylim([-4,4])
        plt.legend()
        plt.title('PCA')

    elif dimensions == '3D':
        sklearn_pca = sklearnPCA(n_components=3)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10

        for i in range(n_tags):
            ax.plot(sklearn_transf[i*n_samples:(i+1)*n_samples,0], sklearn_transf[i*n_samples:(i+1)*n_samples,1],\
                sklearn_transf[i*n_samples:(i+1)*n_samples,2], 'o', markersize=8, color=colours[i], alpha=0.5, label=tags[i])

        plt.title('PCA')
        ax.legend(loc='upper right')

    # plt.savefig("%s.png" % (dimensions), bbox_inches='tight',dpi=200)
    plt.show()
    # plt.close()

    return True


def get_vector_samples(tag, utterances, model, n_samples):

    samples = np.empty((n_samples, model.layer1_size))
    utt_candidates = np.array(utterances[tag])
    idxs = np.random.randint(0, len(utt_candidates), n_samples)

    for i,utt in enumerate(utt_candidates[idxs]):
        samples[i,:] = model.infer_vector(utt)

    return samples


def load_utterances(fname):

    utterances = defaultdict(list)

    for tag, tokens in get_utterances_from_file(fname):
        # remove id from tag
        tag = tag.split("/")[0]
        utterances[tag].append(tokens)

    return utterances


if __name__ == '__main__':

    #TODO: change paths accordingly
    embedding_model_filename = 'data/test'  # path to doc2vec model
    utterance_filename = 'data/swda_utterances.txt'    # path to utterance file

    model, _ = load_all_models(embedding_model_filename)
    utterances = load_utterances(utterance_filename)

    #TODO: select tags to plot (max 5)
    tags_to_plot = ['qw','ft','ar','fa', 't1']

    #TODO: select nr of samples
    n_samples = 500

    samples = np.empty((n_samples*len(tags_to_plot), model.layer1_size))

    for i,tag in enumerate(tags_to_plot):
        tag_samples = get_vector_samples(tag, utterances, model, n_samples)
        samples[i*n_samples:(i+1)*n_samples,:] = tag_samples

    plotGraph(samples, n_samples, tags_to_plot, dimensions='2D')
    plotGraph(samples, n_samples, tags_to_plot, dimensions='3D')