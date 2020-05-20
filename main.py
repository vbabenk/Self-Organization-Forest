# Copyright © 2020. All rights reserved.
# Authors: Vitalii Babenko
# Contacts: vbabenko2191@gmail.com

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy


# Getting new Xi and y of first side
def get_X1_y1(X_train, y_train, X_test, y_test, index, threshold, side):
    X1_train = []
    y1_train = []
    for i in range(y_train.shape[0]):
        if side == 1:
            if X_train[i, index] < threshold:
                X1_train.append(X_train[i])
                y1_train.append(y_train[i])
        else:
            if X_train[i, index] >= threshold:
                X1_train.append(X_train[i])
                y1_train.append(y_train[i])
    X1_train = np.asarray(X1_train)
    y1_train = np.asarray(y1_train)
    X1_test = []
    y1_test = []
    for i in range(y_test.shape[0]):
        if side == 1:
            if X_test[i, index] < threshold:
                X1_test.append(X_test[i])
                y1_test.append(y_test[i])
        else:
            if X_test[i, index] >= threshold:
                X1_test.append(X_test[i])
                y1_test.append(y_test[i])
    X1_test = np.asarray(X1_test)
    y1_test = np.asarray(y1_test)
    return X1_train, y1_train, X1_test, y1_test


# Getting new Xi and y of second side
def get_X2_y2(X_train, y_train, X_test, y_test, index, threshold, side):
    X2_train = []
    y2_train = []
    for i in range(y_train.shape[0]):
        if side == 2:
            if X_train[i, index] < threshold:
                X2_train.append(X_train[i])
                y2_train.append(y_train[i])
        else:
            if X_train[i, index] >= threshold:
                X2_train.append(X_train[i])
                y2_train.append(y_train[i])
    X2_train = np.asarray(X2_train)
    y2_train = np.asarray(y2_train)
    X2_test = []
    y2_test = []
    for i in range(y_test.shape[0]):
        if side == 2:
            if X_test[i, index] < threshold:
                X2_test.append(X_test[i])
                y2_test.append(y_test[i])
        else:
            if X_test[i, index] >= threshold:
                X2_test.append(X_test[i])
                y2_test.append(y_test[i])
    X2_test = np.asarray(X2_test)
    y2_test = np.asarray(y2_test)
    return X2_train, y2_train, X2_test, y2_test


# Finding best threshold of single Xi
def find_threshold_of_x(xx, col, y, num_of_pos, num_of_neg):
    threshold_list = []
    TP1_list = []
    TN1_list = []
    TP2_list = []
    TN2_list = []
    value_list1 = []
    value_list2 = []
    if xx.shape[0] > 2:
        for j in range(1, xx.shape[0] - 1):
            TP1 = 0  # number of True Positive
            TN1 = 0  # number of True Negative
            TP2 = 0
            TN2 = 0
            for z in range(col.shape[0]):
                if col[z] < xx[j]:
                    if y[z] == 1:
                        TP1 += 1
                    else:
                        TN2 += 1
                else:
                    if y[z] == 2:
                        TN1 += 1
                    else:
                        TP2 += 1
            threshold_list.append(xx[j])
            TP1_list.append(TP1)
            TN1_list.append(TN1)
            TP2_list.append(TP2)
            TN2_list.append(TN2)
            if (num_of_pos != 0) and (num_of_neg != 0):
                value_list1.append(((TP1 / num_of_pos) + (TN1 / num_of_neg)) / 2)
                value_list2.append(((TP2 / num_of_pos) + (TN2 / num_of_neg)) / 2)
            elif num_of_pos == 0:
                value_list1.append(TN1 / num_of_neg)
                value_list2.append(TN2 / num_of_neg)
            else:
                value_list1.append(TP1 / num_of_pos)
                value_list2.append(TP2 / num_of_pos)
    else:
        TP1 = 0
        TN1 = 0
        TP2 = 0
        TN2 = 0
        for z in range(col.shape[0]):
            if col[z] < xx[1]:
                if y[z] == 1:
                    TP1 += 1
                else:
                    TN2 += 1
            else:
                if y[z] == 2:
                    TN1 += 1
                else:
                    TP2 += 1
        threshold_list.append(xx[1])
        TP1_list.append(TP1)
        TN1_list.append(TN1)
        TP2_list.append(TP2)
        TN2_list.append(TN2)
        if num_of_pos > 0 and num_of_neg > 0:
            value_list1.append(((TP1 / num_of_pos) + (TN1 / num_of_neg)) / 2)
            value_list2.append(((TP2 / num_of_pos) + (TN2 / num_of_neg)) / 2)
        elif num_of_pos == 0:
            value_list1.append(TN1 / num_of_neg)
            value_list2.append(TN2 / num_of_neg)
        else:
            value_list1.append(TP1 / num_of_pos)
            value_list2.append(TP1 / num_of_pos)
    if max(value_list1) > max(value_list2):
        threshold = threshold_list[value_list1.index(max(value_list1))]
        TP = TP1_list[value_list1.index(max(value_list1))]
        TN = TN1_list[value_list1.index(max(value_list1))]
        value = max(value_list1)
        side = 1
    else:
        threshold = threshold_list[value_list2.index(max(value_list2))]
        TP = TP2_list[value_list2.index(max(value_list2))]
        TN = TN2_list[value_list2.index(max(value_list2))]
        value = max(value_list2)
        side = 2
    FP = num_of_pos - TP  # number of False Positive
    FN = num_of_neg - TN  # number of False Negative
    return threshold, value, side, FP, FN


# Finding threshold of each Xi
def find_thresholds(X_train, y_train, X_test, y_test):
    num_of_pos = np.sum(y_train == 1)  # number of positive objects (norma)
    num_of_neg = np.sum(y_train == 2)  # number of negative objects (pathology)
    threshold_list = []
    train_value_list = []
    side_list = []
    FP_list = []
    FN_list = []
    for i in range(X_train.shape[1]):
        col = X_train[:, i]
        xx = copy.deepcopy(col)  # make copy of Xi to not change initial list
        xx.sort()

        # get threshold of Xi, its value on train sample and side of threshold
        threshold, train_value, side, FP, FN = find_threshold_of_x(xx, col, y_train, num_of_pos, num_of_neg)

        threshold_list.append(threshold)
        train_value_list.append(train_value)
        side_list.append(side)
        FP_list.append(FP)
        FN_list.append(FN)
    train_value_list = np.asarray(train_value_list)

    # get value of thresholds on test sample
    num_of_pos = np.sum(y_test == 1)
    num_of_neg = np.sum(y_test == 2)
    test_value_list = []
    if y_test.shape[0] > 0:
        for i in range(X_test.shape[1]):
            col = X_test[:, i]
            TP = 0
            TN = 0
            for j in range(col.shape[0]):
                if side_list[i] == 1:
                    if col[j] < threshold_list[i]:
                        if y_test[j] == 1:
                            TP += 1
                    else:
                        if y_test[j] == 2:
                            TN += 1
                else:
                    if col[j] >= threshold_list[i]:
                        if y_test[j] == 1:
                            TP += 1
                    else:
                        if y_test[j] == 2:
                            TN += 1
            if num_of_pos > 0 and num_of_neg > 0:
                test_value_list.append(((TP / num_of_pos) + (TN / num_of_neg)) / 2)
            elif num_of_pos == 0:
                test_value_list.append(TN / num_of_neg)
            else:
                test_value_list.append(TP / num_of_pos)
        test_value_list = np.asarray(test_value_list)
    else:
        test_value_list = np.ones(X_train.shape[1])
    return threshold_list, side_list, FP_list, FN_list, train_value_list, test_value_list


# Getting value of each feature on next level
def get_value_on_next_level(Xtrain, ytrain, Xtest, ytest, test_weight):
    threshold_list, side_list, FP_list, FN_list, train_value_list, test_value_list = find_thresholds(X_train=Xtrain,
                                                                                                     y_train=ytrain,
                                                                                                     X_test=Xtest,
                                                                                                     y_test=ytest)
    complex_value_list = (1 - test_weight) * train_value_list + test_weight * test_value_list
    df = pd.DataFrame(
        {'train_value': train_value_list,
         'test_value': test_value_list,
         'complex_value': complex_value_list})
    df = df.sort_values(['complex_value', 'test_value', 'train_value'], ascending=[False, False, False])
    return df['complex_value'].values[0]


# Getting new nodes of tree
def get_new_nodes(Xtrain, ytrain, Xtest, ytest, col_names, test_weight, Xtrain_list, ytrain_list, Xtest_list,
                  ytest_list, lnl, pll, psl, til, leaf_number, tree_index, level_number, previous_leaf, previous_side,
                  mti, F):
    threshold_list, side_list, FP_list, FN_list, train_value_list, test_value_list = find_thresholds(X_train=Xtrain,
                                                                                                     y_train=ytrain,
                                                                                                     X_test=Xtest,
                                                                                                     y_test=ytest)
    if max(train_value_list) < 1.0:
        value_list = []
        for index in range(len(col_names)):
            if FP_list[index] > 0:
                X1_train, y1_train, X1_test, y1_test = get_X1_y1(Xtrain, ytrain, Xtest, ytest, index,
                                                                 threshold_list[index], side_list[index])
                if y1_train.shape[0] > 1:
                    first_value = get_value_on_next_level(X1_train, y1_train, X1_test, y1_test, test_weight)
                else:
                    first_value = 0.0
            else:
                first_value = 1.0
            if FN_list[index] > 0:
                X2_train, y2_train, X2_test, y2_test = get_X2_y2(Xtrain, ytrain, Xtest, ytest, index,
                                                                 threshold_list[index], side_list[index])
                if y2_train.shape[0] > 1:
                    second_value = get_value_on_next_level(X2_train, y2_train, X2_test, y2_test, test_weight)
                else:
                    second_value = 0.0
            else:
                second_value = 1.0
            value_list.append((first_value + second_value) / 2)

        # find indexes of F best features
        df = pd.DataFrame({'value': value_list})
        df = df.sort_values(['value'], ascending=[False])
    else:
        complex_value_list = (1 - test_weight) * train_value_list + test_weight * test_value_list
        df = pd.DataFrame(
            {'train_value': train_value_list,
             'test_value': test_value_list,
             'complex_value': complex_value_list})
        df = df.sort_values(['complex_value', 'test_value', 'train_value'], ascending=[False, False, False])
    index_list = df.index.tolist()[:F]
    node_list = []
    temp = 0
    for index in index_list:
        node = []
        node.append(col_names[index])  # best feature
        node.append(side_list[index])  # side of threshold
        node.append(threshold_list[index])  # threshold value
        node.append(train_value_list[index])  # train value
        node.append(test_value_list[index])  # test value
        node.append(FP_list[index])  # number of False Positive
        if FP_list[index] > 0:
            X1_train, y1_train, X1_test, y1_test = get_X1_y1(Xtrain, ytrain, Xtest, ytest, index,
                                                             threshold_list[index], side_list[index])
            if y1_train.shape[0] > 1:
                Xtrain_list.append(X1_train)
                ytrain_list.append(y1_train)
                Xtest_list.append(X1_test)
                ytest_list.append(y1_test)
                lnl.append(level_number)
                pll.append(leaf_number)
                psl.append(1)
                if temp == 0:
                    til.append(tree_index)
                else:
                    til.append(mti)

        node.append(FN_list[index])  # number of False Negative
        if FN_list[index] > 0:
            X2_train, y2_train, X2_test, y2_test = get_X2_y2(Xtrain, ytrain, Xtest, ytest, index,
                                                             threshold_list[index], side_list[index])
            if y2_train.shape[0] > 1:
                Xtrain_list.append(X2_train)
                ytrain_list.append(y2_train)
                Xtest_list.append(X2_test)
                ytest_list.append(y2_test)
                lnl.append(level_number)
                pll.append(leaf_number)
                psl.append(2)
                if temp == 0:
                    til.append(tree_index)
                else:
                    til.append(mti)

        node.append(leaf_number)  # current leaf number
        node.append(level_number)  # current level number
        node.append(previous_leaf)  # previous leaf number
        node.append(previous_side)  # previous side
        node_list.append(node)
        mti += 1
        temp += 1
    return node_list, Xtrain_list, ytrain_list, Xtest_list, ytest_list, lnl, pll, psl


# Finding best tree
def get_forest(test_weight, X_train, y_train, X_test, y_test, col_names, F):
    tree_list = []
    leaf_number = 1  # first leaf number
    level_number = 1  # first level number
    tree_index = 0  # first tree index
    Xtrain_list = []
    ytrain_list = []
    Xtest_list = []
    ytest_list = []
    lnl = []  # level number list
    pll = []  # previous leaf list
    psl = []  # previous side list
    til = []  # tree index list
    F_counter = 0
    node_list, Xtrain_list, ytrain_list, Xtest_list, ytest_list, lnl, pll, psl = get_new_nodes(X_train, y_train,
                                                                                               X_test, y_test,
                                                                                               col_names,
                                                                                               test_weight,
                                                                                               Xtrain_list,
                                                                                               ytrain_list,
                                                                                               Xtest_list,
                                                                                               ytest_list, lnl, pll,
                                                                                               psl, til,
                                                                                               leaf_number,
                                                                                               tree_index,
                                                                                               level_number, 0, 0,
                                                                                               0, F)
    for new_node in node_list:
        tree_list.append(new_node)
    count_list = np.zeros(F)
    i = 0
    while i < len(pll):
        Xtrain = Xtrain_list[i]
        ytrain = ytrain_list[i]
        Xtest = Xtest_list[i]
        ytest = ytest_list[i]
        leaf_number += 1
        level_number = lnl[i] + 1
        previous_leaf = pll[i]
        previous_side = psl[i]
        tree_index = til[i]
        mti = max(til)  # max tree index
        node_list, Xtrain_list, ytrain_list, Xtest_list, ytest_list, lnl, pll, psl = get_new_nodes(Xtrain, ytrain,
                                                                                                   Xtest, ytest,
                                                                                                   col_names,
                                                                                                   test_weight,
                                                                                                   Xtrain_list,
                                                                                                   ytrain_list,
                                                                                                   Xtest_list,
                                                                                                   ytest_list, lnl,
                                                                                                   pll, psl, til,
                                                                                                   leaf_number,
                                                                                                   tree_index,
                                                                                                   level_number,
                                                                                                   previous_leaf,
                                                                                                   previous_side,
                                                                                                   mti, F)
        temp_tree = copy.deepcopy(tree_list[til[i]])
        temp = 0
        for new_node in node_list:
            tree = []
            if sum(count_list) < F ** 2:
                if count_list[til[i]] < F:
                    tree.append(temp_tree)
                    count_list[til[i]] += 1
                else:
                    for node in temp_tree:
                        tree.append(node)
            else:
                for node in temp_tree:
                    tree.append(node)
            tree.append(new_node)
            if temp == 0:
                tree_list[til[i]] = tree
            else:
                tree_list.append(tree)
            temp += 1
        i += 1
    return tree_list


# Calculate exam value for forest
def get_exam_value(tree_list, exam_data, y_exam):
    num_of_pos = np.sum(y_exam == 1)
    num_of_neg = np.sum(y_exam == 2)
    count = 0
    TP = 0
    TN = 0
    for i in range(exam_data.shape[0]):
        obj = exam_data.loc[i]
        ypl = []  # y_pred list
        for tree in tree_list:
            tree_df = pd.DataFrame(tree)
            tree = tree_df.values
            level = 1
            index = 0
            flag = False
            y_pred = 0
            while flag != True:
                node = tree[index]
                if node[1] == 1:
                    if obj[node[0]] < node[2]:
                        y_pred = 1
                    else:
                        y_pred = 2
                else:
                    if obj[node[0]] < node[2]:
                        y_pred = 2
                    else:
                        y_pred = 1
                if np.where((tree[:, 9] == level) & (tree[:, 10] == y_pred))[0].size > 0:
                    index = np.where((tree[:, 9] == level) & (tree[:, 10] == y_pred))[0][0]
                    level = tree[index, 7]
                else:
                    flag = True
            ypl.append(y_pred)
        ypl = np.asarray(ypl)
        if np.sum(ypl == 1) > np.sum(ypl == 2):
            y_pred = 1
        else:
            y_pred = 2
        if y_exam[i] == y_pred:
            count += 1
            if y_exam[i] == 1:
                TP += 1
            else:
                TN += 1
    return count / y_exam.shape[0], TP / num_of_pos, TN / num_of_neg


# weight_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
weight_list = [0.9]
sensor_list = ['convex', 'linear', 'reinforced', 'xmixed', 'ymixed']
F_list = [3, 6, 7, 7, 7]
for i in range(len(sensor_list)):
    sensor_type = sensor_list[i]
    print('Sensor type: ', sensor_type)

    F = F_list[i]

    name_of_train = sensor_type + '(train).xlsx'  # train + test sample
    name_of_exam = sensor_type + '(exam).xlsx'  # exam sample
    name_of_validation = sensor_type + '(validation).xlsx'  # validation sample

    train_data = pd.read_excel(name_of_train)
    exam_data = pd.read_excel(name_of_exam)
    validate_data = pd.read_excel(name_of_validation)

    col_names = list(train_data.columns[:-1])

    y = train_data['class'].values
    X = train_data.drop(['class'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    y_exam = exam_data['class'].values

    y_val = validate_data['class'].values

    best_value = 0
    for test_weight in weight_list:
        forest = get_forest(test_weight=test_weight, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                            col_names=col_names, F=F)
        tree_columns = ['feature', 'side', 'threshold', 'train_value', 'test_value', 'FP', 'FN', 'leaf_number',
                        'level_number', 'previous_leaf', 'previous_side']
        leaf_list = []
        level_list = []
        for tree in forest:
            tree_df = pd.DataFrame(tree, columns=tree_columns)
            leaf_list.append(max(tree_df['leaf_number']))
            level_list.append(max(tree_df['level_number']))
        criterion = pd.DataFrame(
            {'number_of_leafs': leaf_list,
             'number_of_levels': level_list})
        criterion = criterion.sort_values(['number_of_leafs', 'number_of_levels'])
        for t in range(1, 21):
            t_best = criterion.index.tolist()[:t]
            new_forest = []
            for z in t_best:
                new_forest.append(forest[z])
            accuracy, sensitivity, specificity = get_exam_value(new_forest, exam_data, y_exam)
            F_value = 0.5 * sensitivity + 0.5 * specificity
            if F_value > best_value:
                best_value = F_value
                best_forest = copy.deepcopy(new_forest)
                best_weight = test_weight
                optimal_t = t
                top_accuracy = accuracy
                top_sensitivity = sensitivity
                top_specificity = specificity
    print('Exam result:')
    print(' - Best weight: ', best_weight)
    print(' - Optimal t: ', optimal_t)
    print(' - Top accuracy: ', top_accuracy)
    print(' - Top sensitivity: ', top_sensitivity)
    print(' - Top specificty: ', top_specificity)
    accuracy, sensitivity, specificity = get_exam_value(best_forest, validate_data, y_val)
    print('Validation result:')
    print(' - Accuracy: ', accuracy)
    print(' - Sensitivity: ', sensitivity)
    print(' - Specificity: ', specificity)
    print()
