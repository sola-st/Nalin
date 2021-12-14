# Author: Michael Pradel

import argparse
import pandas
import krippendorff
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", help="CSV file with raw survey results (exported from Google Forms)", required=True)
parser.add_argument(
    "--truth", help="CSV file with ground truth (warning vs. original)", required=True)
parser.add_argument(
    "--outdir", help="Output directory", required=True)


def read_data(data_file, truth_file):
    all_data = pandas.read_csv(data_file)
    # drop timestamps and names of participant
    all_data = all_data.drop(all_data.columns[[0, 1]], axis=1)
    all_truth = pandas.read_csv(truth_file, header=None)
    return all_data, all_truth


def compute_interrater_agreement(data):
    all_rater_data = []
    for _, row in data.iterrows():
        all_rater_data.append(row.tolist())
    ira = krippendorff.alpha(all_rater_data, level_of_measurement="ordinal")
    print(f'Inter-rater agreement: {round(ira, 2)}')

    #  for testing only:
    # print("\n================== Random rating data:")
    # all_rater_data = []
    # for rater_idx in range(11):
    #     rater_data = []
    #     for question_idx in range(40):
    #         rater_data.append(random.randint(1, 5))
    #     all_rater_data.append(rater_data)
    # print(all_rater_data)
    # print(krippendorff.alpha(all_rater_data, level_of_measurement="ordinal"))

    # print("\n================== Full-agreement, artificial rating data:")
    # all_rater_data = []
    # question_to_answer = {}
    # for rater_idx in range(11):
    #     rater_data = []
    #     for question_idx in range(40):
    #         if question_idx in question_to_answer:
    #             rater_data.append(question_to_answer[question_idx])
    #         else:
    #             answer = random.randint(1, 5)
    #             rater_data.append(answer)
    #             question_to_answer[question_idx] = answer
    #     all_rater_data.append(rater_data)
    # print(all_rater_data)
    # print(krippendorff.alpha(all_rater_data, level_of_measurement="ordinal"))


def analyze_results(all_data, all_truth, ignore_epsilon=0):
    nb_ignored = 0
    nb_warnings_hard = 0
    nb_warnings_easy = 0
    nb_originals_hard = 0
    nb_originals_easy = 0
    for columnIdx in range(len(all_truth.columns)):
        ratings = all_data[all_data.columns[columnIdx]]
        truth = all_truth[all_truth.columns[columnIdx]]
        truth = truth[0].strip()

        if ratings.mean() <= 3 - ignore_epsilon:  # hard
            if truth == "original":
                nb_originals_hard += 1
            else:
                nb_warnings_hard += 1
        elif ratings.mean() >= 3 + ignore_epsilon:  # easy
            if truth == "original":
                nb_originals_easy += 1
            else:
                nb_warnings_easy += 1
        else:
            nb_ignored += 1

        print(f"\n{ratings}")
        print(f" --> mean: {ratings.mean()}, truth: {truth}")

    print(f"Warnings : {nb_warnings_easy} easy, {nb_warnings_hard} hard -- {round(nb_warnings_hard/(nb_warnings_easy + nb_warnings_hard), 2)} correctly predicted")
    print(f"Originals: {nb_originals_easy} easy, {nb_originals_hard} hard  -- {round(nb_originals_easy/(nb_originals_easy + nb_originals_hard), 2)} correctly predicted")
    print(f"Ignored: {nb_ignored}")


def counter_to_mean(counter):
    sum_of_numbers = sum(number*count for number, count in counter.items())
    count = sum(count for _, count in counter.items())
    return sum_of_numbers / count


def plot(all_data, all_truth, outdir):
    originals_counter = Counter()
    warnings_counter = Counter()
    for columnIdx in range(len(all_truth.columns)):
        ratings = all_data[all_data.columns[columnIdx]]
        truth = all_truth[all_truth.columns[columnIdx]]
        truth = truth[0].strip()

        if truth == "original":
            originals_counter.update(ratings.tolist())
        else:
            warnings_counter.update(ratings.tolist())

    print(f"Ratings for originals: {originals_counter}")
    print(f"Ratings for warnings : {warnings_counter}")

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}

    plt.rc('font', **font)

    ax = plt.subplot()
    xs = range(1, 6)
    ys_originals = [originals_counter[i] for i in xs]
    ys_warnings = [warnings_counter[i] for i in xs]
    ax.bar([x-0.2 for x in xs], ys_originals,
           width=0.2, color='g', align='center')
    ax.bar([x+0.2 for x in xs], ys_warnings,
           width=0.2, color='r', align='center')
    ax.legend(["Pairs without warning", "Pairs with warning"])
    # ax.axvline(counter_to_mean(originals_counter), color='g')
    # ax.axvline(counter_to_mean(warnings_counter), color='r')
    plt.xlabel("Maintainability (1=hard, 5=easy)")
    plt.ylabel("Number of ratings")
    plt.title("Ratings in user study")
    plt.savefig(f"{outdir}/ratings.pdf")
    plt.close()
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)

def plot_as_line(all_data, all_truth, outdir):
    originals_counter = Counter()
    warnings_counter = Counter()
    fontName = "Fira Sans"
    colors = ['#c7011a', '#47bb02']
    sns.set_palette(sns.color_palette(colors))

    plt.rcParams['font.size'] = 15
    plt.rcParams['font.sans-serif'] = [fontName, 'sans-serif']

    for columnIdx in range(len(all_truth.columns)):
        ratings = all_data[all_data.columns[columnIdx]]
        truth = all_truth[all_truth.columns[columnIdx]]
        truth = truth[0].strip()

        if truth == "original":
            originals_counter.update(ratings.tolist())
        else:
            warnings_counter.update(ratings.tolist())

    print(f"Ratings for originals: {originals_counter}")
    print(f"Ratings for warnings : {warnings_counter}")

    font = {'family': 'sans-serif',
            #'weight': 'bold',
            'size': 14}

    plt.rc('font', **font)

    ax = plt.subplot()
    xs = range(1, 6)
    ys_originals = [originals_counter[i] for i in xs]
    ys_warnings = [warnings_counter[i] for i in xs]
    ax_or = sns.lineplot(x=xs, y=ys_originals,marker='o', color = sns.color_palette()[1], markevery=[0,4])
    ax_wrn = sns.lineplot(x=xs, y=ys_warnings,marker='o', color = sns.color_palette()[0], markevery=[0,4])

    ax.set(xlabel="Maintainability (1=hard, 5=easy)", ylabel='Number of ratings')
    label_font_size = 11
    # ax_or.text(1,56,'Hard', fontsize=label_font_size)
    # ax_or.text(4.7,17,'Easy',fontsize=label_font_size)
    #
    # ax_wrn.text(1.1,13, 'Hard', fontsize=label_font_size)
    # ax_wrn.text(4.6,88, 'Easy', fontsize=label_font_size)
    # plt.show()
    ax_or.text(3.4, 80, 'Pairs without warning', color = sns.color_palette()[1],fontsize=label_font_size)
    ax_wrn.text(4, 27, 'Pairs with warning', color = sns.color_palette()[0],fontsize=label_font_size)
    plt.savefig(f'{outdir}/rating_line.pdf', transparent=True,
                            bbox_inches='tight')
    plt.close()

def stats_test(all_data, all_truth):
    originals_rates = []
    warnings_rates = []
    for columnIdx in range(len(all_truth.columns)):
        ratings = all_data[all_data.columns[columnIdx]]
        truth = all_truth[all_truth.columns[columnIdx]]
        truth = truth[0].strip()

        if truth == "original":
            originals_rates.extend(ratings.tolist())
        else:
            warnings_rates.extend(ratings.tolist())

    print(stats.mannwhitneyu(originals_rates, warnings_rates))


if __name__ == "__main__":
    args = parser.parse_args()
    data, truth = read_data(args.data, args.truth)
    compute_interrater_agreement(data)
    analyze_results(data, truth)
    # plot(data, truth, args.outdir)
    plot_as_line(all_data=data, all_truth=truth, outdir=args.outdir)
    stats_test(data, truth)
