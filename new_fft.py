import collections
import pandas as pd
from helpers import get_performance, get_score, subtotal, get_recall, get_auc
from sklearn.metrics import auc
from DE import DE


PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
MATRIX = "\t".join(["\tTP", "FP", "TN", "FN"])
PERFORMANCE = " \t".join(["\tCLF", "PRE ", "REC", "SPE", "FPR", "NPV", "ACC", "F_1"])


class FFT(object):

    def __init__(self, max_level=4, max_depth=4, medianTop=0):
        self.max_depth = max_depth
        self.median_top = medianTop
        cnt = 2 ** (max_level - 1)
        self.tree_cnt = cnt
        self.tree_depths = [0] * cnt
        self.best = -1
        self.target = "" # "bug"
        self.ignore = {} # moved to main. # {"name", "version", 'name.1', 'prediction'}
        self.criteria = "Dist2Heaven"
        self.data_name = ''
        self.train, self.test = None, None
        self.structures = None
        self.computed_cache = {}
        self.selected = [{} for _ in range(cnt)]
        self.tree_scores = [None] * cnt
        self.dist2heavens = [None] * cnt
        self.node_descriptions = [None] * cnt
        self.performance_on_train = [collections.defaultdict(dict) for _ in xrange(cnt)]
        self.performance_on_test = [None] * cnt
        self.predictions = [None] * cnt
        self.loc_aucs = [None] * cnt
        self.print_enabled = True
        self.results = {}

    "Get the loc_auc for a specific tree"
    def get_tree_loc_auc(self, data, i):
        if "loc" not in data:
            return 8
        # self.predict will add/modify the 'prediction' column to the data
        self.predict(data, i)
        sorted_data = data.sort_values(by=["prediction", "loc"], ascending=[False, True])
        return get_auc(sorted_data)

    "Build all possible tress."

    def build_trees(self):
        self.structures = self.get_all_structure()
        data = self.train
        for i in range(self.tree_cnt):
            self.grow(data, i, 0, [0, 0, 0, 0])
            self.loc_aucs[i] = self.get_tree_loc_auc(data, i)


    "Evaluate all tress built on TEST data."
    def eval_trees(self):
        for i in range(self.tree_cnt):
            # Get performance on TEST data.
            self.eval_tree(i)

    "Evaluate the performance of the given tree on the TEST data."
    def eval_tree(self, t_id):
        if self.performance_on_test[t_id]:
            return
        depth = self.tree_depths[t_id]
        self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        TP, FP, TN, FN = 0, 0, 0, 0
        data = self.test
        for level in range(depth + 1):
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
            print(metrics)
            tp, fp, tn, fn = self.update_metrics(level, depth, decision, metrics)
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
            if len(undecided) == 0:
                break
            data = undecided
        pre, rec, spec, fpr, npv, acc, f1 = get_performance([TP, FP, TN, FN])
        self.performance_on_test[t_id] = [TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1]

    "Find the best tree based on the score in TRAIN data."

    def find_best_tree(self):
        if self.tree_scores and self.tree_scores[0]:
            return
        if not self.performance_on_train or not self.performance_on_train[0]:
            self.grow()
        if self.print_enabled:
            print "\t----- PERFORMANCES FOR ALL FFTs on Training Data -----"
            print PERFORMANCE + " \t" + self.criteria
        best = [-1, float('inf')]
        for i in range(self.tree_cnt):
            all_metrics = self.performance_on_train[i][self.tree_depths[i]]
            if self.criteria == "LOC_AUC":
                score = self.loc_aucs[i]
            else:
                score = get_score(self.criteria, all_metrics[:4])
            self.tree_scores[i] = score
            self.dist2heavens[i] = get_score("Dist2Heaven", all_metrics[:4])
            if score < best[-1]:
                best = [i, score]
            if self.print_enabled:
                print "\t" + "\t".join(
                    ["FFT(" + str(i) + ")"] + \
                    [str(x).ljust(5, "0") for x in all_metrics[4:] + \
                     [score if self.criteria == "Dist2Heaven" else -score]])
        if self.print_enabled:
            print "\tThe best tree found on training data is: FFT(" + str(best[0]) + ")"
        self.best = best[0]
        return best[0]

    "Given how the decision is made, get the description for the node."

    def describe_decision(self, t_id, level, metrics, reversed=False):
        cue, direction, threshold, decision = self.selected[t_id][level]
        tp, fp, tn, fn = metrics
        results = ["\'Good\'", "\'Bug!\'"]
        mappings = {">":"<=", "<":">="}
        direction = mappings[direction] if reversed else direction+" "
        description = ("\t| " * (level + 1) + \
                       " ".join([cue, direction, str(threshold)]) + \
                       "\t--> " + results[1 - decision if reversed else decision]).ljust(30, " ")
        pos = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        neg = "\tCorrect Rej: " + str(tn) + ", Miss: " + str(fn)
        if not reversed:
            description += pos if decision == 1 else neg
        else:
            description += neg if decision == 1 else pos
        return description

    "Given how the decision is made, get the performance for this decision."
    def eval_decision(self, data, cue, direction, threshold, decision):
        try:
            if direction == ">":
                pos, neg = data.loc[data[cue] > threshold], data.loc[data[cue] <= threshold]
            else:
                pos, neg = data.loc[data[cue] < threshold], data.loc[data[cue] >= threshold]
        except:
            return 1, 2, 3
        if decision == 1:
            decided = pos
            undecided = neg
        else:
            pos, neg = neg, pos
            decided = neg
            undecided = pos
        # get auc for loc.
        if "loc" in pos and "loc" in neg:
            sorted_data = pd.concat([df.sort_values(by=["loc"], ascending=True) for df in [pos, neg]])
            loc_auc = get_auc(sorted_data)
        else:
            loc_auc = 0
        tp = pos.loc[pos[self.target] == 1]
        fp = pos.loc[pos[self.target] == 0]
        tn = neg.loc[neg[self.target] == 0]
        fn = neg.loc[neg[self.target] == 1]
        # pre, rec, spec, fpr, npv, acc, f1 = get_performance([tp, fp, tn, fn])
        # return undecided, [tp, fp, tn, fn, pre, rec, spec, fpr, npv, acc, f1]
        return undecided, map(len, [tp, fp, tn, fn]), loc_auc


    "Given data and the specific tree id, add a 'prediction' column to the dataframe."
    def predict(self, data, t_id=-1):
        # predictions = pd.Series([None] * len(data))
        if t_id == -1:
            t_id = self.best
        original = data
        original['prediction'] = pd.Series([None] * len(data))
        depth = self.tree_depths[t_id]
        for level in range(depth + 1):
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
            decided_idx = [i for i in data.index if i not in undecided.index]
            # original['prediction'][decided_idx] = decision
            original.loc[decided_idx, 'prediction'] = decision
            data = undecided
        original.loc[data.index, 'prediction'] = 1 if decision == 0 else 0
        # original['prediction'][undecided.index] = 1 if decision == 0 else 0
        if None in original['prediction']:
            print "ERROR!"
        self.predictions[t_id] = original['prediction'].values
        return self.predictions[t_id]


    "Grow the t_id_th tree for the level with the given data. Also save its performance on the TRAIN data"

    def grow(self, data, t_id, level, cur_performance):
        """
        :param data: current data for future tree growth
        :param t_id: tree id
        :param level: level id
        :return: None
        """
        if level >= self.max_depth:
            return
        if len(data) == 0:
            print "?????????????????????? Early Ends ???????????????????????"
            return
        # print "level, ", level
        self.tree_depths[t_id] = level
        decision = self.structures[t_id][level]
        structure = tuple(self.structures[t_id][:level + 1])
        cur_selected = self.computed_cache.get(structure, None)
        TP, FP, TN, FN = cur_performance
        if not cur_selected:
            for cue in list(data):
                if cue in self.ignore or cue == self.target:
                    continue
                if(self.median_top == 1 and level == 0) or (self.median_top == 0):
                    threshold = data[cue].median()
                else:
                    threshold = data[cue]
                for direction in "><":
                    undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
                    tp, fp, tn, fn = self.update_metrics(level, self.max_depth, decision, metrics)
                    # if the decision lead to no data, punish the score
                    if sum([tp, fp, tn, fn]) == 0:
                        score = float('inf')
                    elif self.criteria == "LOC_AUC":
                        score = loc_auc
                    else:
                        score = get_score(self.criteria, [TP + tp, FP + fp, TN + tn, FN + fn])
                    # score = get_score(self.criteria, metrics)
                    # if not cur_selected or metrics[goal] > self.performance_on_train[t_id][level][cur_selected][goal]:
                    if not cur_selected or score < cur_selected['score']:
                        cur_selected = {'rule': (cue, direction, threshold, decision), \
                                        'undecided': undecided, \
                                        'metrics': [TP + tp, FP + fp, TN + tn, FN + fn], \
                                        # 'metrics': metrics,
                                        'score': score}
                        x = 1
            self.computed_cache[structure] = cur_selected
        self.selected[t_id][level] = cur_selected['rule']
        self.performance_on_train[t_id][level] = cur_selected['metrics'] + get_performance(cur_selected['metrics'])
        self.grow(cur_selected['undecided'], t_id, level + 1, cur_selected['metrics'])

    # "describe a tree on the TEST data"
    #
    # def describe_tree(self, t_id):
    #     data = self.test
    #     depth = self.tree_depths[t_id]
    #     for level in range(depth + 1):
    #         cue, direction, threshold, decision = self.selected[t_id][level]
    #         undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
    #         tp, fp, tn, fn = self.update_metrics(level, depth, decision, metrics)
    #         description = self.describe_decision(t_id, level, metrics)
    #         self.node_descriptions[t_id][level] += [description]
    #         if len(undecided) == 0:
    #             break
    #         data = undecided
    #     description = self.describe_decision(t_id, level, metrics, reversed=True)
    #     self.node_descriptions[t_id][level] += [description]

    "Given tree id, print the specific tree and its performances on the test data."

    def print_tree(self, t_id):
        data = self.test
        depth = self.tree_depths[t_id]
        if not self.node_descriptions[t_id]:
            self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        for i in range(depth + 1):
            if self.node_descriptions[t_id][i]:
                print self.node_descriptions[t_id][i][0]
            else:
                cue, direction, threshold, decision = self.selected[t_id][i]
                undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
                description = self.describe_decision(t_id, i, metrics)
                self.node_descriptions[t_id][i] += [description]
                print description
                if len(undecided) == 0:
                    break
                data = undecided
        description = self.describe_decision(t_id, i, metrics, reversed=True)
        self.node_descriptions[t_id][i] += [description]
        dist2heaven = get_score("Dist2Heaven", self.performance_on_test[t_id][:4])
        loc_auc = -self.get_tree_loc_auc(self.test, t_id)

        self.results[t_id] = {"d2h": dist2heaven, "auc": loc_auc}
        if self.print_enabled:
            print self.node_descriptions[t_id][i][1]
            print "\t----- CONFUSION MATRIX -----"
            print MATRIX
            print "\t" + "\t".join(map(str, self.performance_on_test[t_id][:4]))

            print "\t----- PERFORMANCES ON TEST DATA -----"
            print PERFORMANCE + " \tD2H"+ " \tLOC"
            print "\t" + "\t".join(
                ["FFT(" + str(self.best) + ")"] + \
                [str(x).ljust(5, "0") for x in self.performance_on_test[t_id][4:11] + [dist2heaven, loc_auc]])
                # map(str, ["FFT(" + str(self.best) + ")"] + self.performance_on_test[t_id][4:] + [dist2heaven]))

    "Get all possible tree structure"

    def get_all_structure(self):
        def dfs(cur, n):
            if len(cur) == n:
                ans.append(cur)
                return
            dfs(cur + [1], n)
            dfs(cur + [0], n)

        if self.max_depth < 0:
            return []
        ans = []
        dfs([], self.max_depth)
        return ans

    "Update the metrics(TP, FP, TN, FN) based on the decision."

    def update_metrics(self, level, depth, decision, metrics):
        tp, fp, tn, fn = metrics
        if level < depth:  # Except the last level, only part of the data(pos or neg) is decided.
            if decision == 1:
                tn, fn = 0, 0
            else:
                tp, fp = 0, 0
        return tp, fp, tn, fn
