import csv
import random
import config

# data_root="/home/jmaronasm/ISIC_challenge_2019/Task_3/"

def find_img_label(row):
    return int(row[7])


def split_dataset(test_class_samples=[1000, 2250, 750, 200, 500, 75, 75, 150]):
    # c0_test=1000, c1_test=2250, c2_test=750, c3_test=200, c4_test=500, c5_test=75, c6_test=75, c7_test=150
    alist = []
    for i in range(8):
        alist.append([])
    with open(config.data_root + "ISIC_2019_Training_GroundTruth.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == 'image':
                continue
            lbl_indx = find_img_label(row)
            alist[lbl_indx].append(row)
    tr_l = []
    tst_l = []
    for i, l in enumerate(alist):
        random.shuffle(l)
        print(len(l))
        tst_l.append(l[:test_class_samples[i]])
        tr_l.append(l[test_class_samples[i]:])

    with open('2k19_submission_validation_partition.csv', 'w', newline='') as testfile, \
            open('2k19_submission_train_partition.csv', 'w', newline='') as trainfile:
        test_writer = csv.writer(testfile, delimiter=',')
        train_writer = csv.writer(trainfile, delimiter=',')
        for l in tst_l:
            for row in l:
                test_writer.writerow(row)
        for l in tr_l:
            for row in l:
                train_writer.writerow(row)

    return


def split_dataset_wvalidation(test_class_samples=[1000, 2250, 750, 200, 500, 75, 75, 150],
                              val_class_samples=[200, 450, 150, 40, 100, 15, 15, 30]):
    # c0_test=1000, c1_test=2250, c2_test=750, c3_test=200, c4_test=500, c5_test=75, c6_test=75, c7_test=150
    alist = []
    for i in range(2):
        alist.append([])
    with open(config.data_root + "train.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == 'image_name':
                continue
            lbl_indx = find_img_label(row)
            alist[lbl_indx].append(row)
    tr_l = []
    tst_l = []
    val_l = []
    for i, l in enumerate(alist):
        random.shuffle(l)
        print(len(l))
        tst_l.append(l[:test_class_samples[i]])
        val_l.append(l[test_class_samples[i]:test_class_samples[i] + val_class_samples[i]])
        tr_l.append(l[test_class_samples[i] + val_class_samples[i]:])

    with open('2k20_test_partition.csv', 'w', newline='') as testfile, \
            open('2k20_train_partition_6.csv', 'w', newline='') as trainfile, \
            open('2k20_validation_partition_6.csv', 'w', newline='') as valfile:
        test_writer = csv.writer(testfile, delimiter=',')
        val_writer = csv.writer(valfile, delimiter=',')
        train_writer = csv.writer(trainfile, delimiter=',')
        for l in tst_l:
            for row in l:
                test_writer.writerow(row)
        for l in val_l:
            for row in l:
                val_writer.writerow(row)
        for l in tr_l:
            for row in l:
                train_writer.writerow(row)
    return


if __name__ == '__main__':
    split_dataset_wvalidation([0, 0], [980, 20])
