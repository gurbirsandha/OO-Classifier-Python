# Copyright 2020 Paul Lu
import sys
import copy     # for deepcopy()

Debug = False   # Sometimes, print for debugging
InputFilename = "file.input.txt"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]


def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return(self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return(s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        # FIXME:  Call superclass, here and for all classes
        self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    # FIXME:  Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords Hardcoded (%d): " % ln, end='')
        print(self.get_target_words())
        return

    def print_run_info(self):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():    # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w            # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)


# Define new class which is a subclass of ClassifyByTarget
class ClassifyByTopN(ClassifyByTarget):
    # Initialize this class and let it inherit all the attributes of
    # ClassifyByTarget
    def __init__(self, lw=[]):
        self.type = str(self.__class__)
        super().__init__(lw)
        return

    # In a given training set (tset), this method will find the top num
    # most frequent words whose label matches the given label (label)
    def target_top_n(self, tset, num=5, label=''):
        # initialize dictionary to store count of each word
        counts = {}
        # loop through the whole training set
        for ti in tset.get_instances():
            # if a given training instance has the desired label, we are
            # interested in all the words in that list
            if ti.inst["label"] == label:
                # iterate over the training instances and count how many times
                # each unique word occurs
                for word in ti.inst["words"]:
                    # if word is new, set its count to 1
                    if word not in counts:
                        counts[word] = 1
                    # if word already exists in dictionary, increment its
                    # count by 1
                    else:
                        counts[word] += 1
        # initialize list to store count values
        count_list = []
        # loop through dictionary keys
        for x in counts:
            # add *unique* count values to count_list
            if counts[x] not in count_list:
                count_list.append(counts[x])
        # create a sub list containing the top num count values from the
        # dictionary
        temp = sorted(count_list, reverse = True)[:num]
        # initialize list to contain top num most frequent words, including
        # ties
        top_words = []
        # iterate over dictionary keys
        for y in counts:
            # check if the count value of each unique word is in the list of
            # the num highest counts. If so, append that word to top_words
            if counts[y] in temp:
                top_words.append(y)
        # update target words so that they only consist of the top num most
        # common words
        self.set_target_words(top_words)
        return


class TrainingInstance(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):             # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=True
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                # FIXME: For testing only.  Compare to previous version.
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)

    def lowercase(self):
        # using list comprehension, turn each word in the training instance
        # to a fully lowercase word
        self.inst["words"] = [x.lower() for x in self.inst["words"]]
        return(self)

    def remove_symbols(self):
        # initialize list that will contain the amended strings
        new_words = []
        # This for loop iterates over each training instance
        for x in self.inst["words"]:
            # This for loop iterates over each string in each training
            # instance
            for y in x:
                # if the character is neither a digit nor a letter, remove it
                if not (y.isdigit() or y.isalpha()):
                    x = x.replace(y, "")
            # add updated string to new_words
            new_words.append(x)
        # update the training instance with the new list of modified words
        self.inst["words"] = new_words
        return(self)

    def remove_numbers(self):
        # initialize list that will contain the amended strings
        new_words = []
        # This for loop iterates over the training instance
        for x in self.inst["words"]:
        # If the string is entirely numeric, it will not be changed
            if not x.isdigit():
                # This for loop iterates over each string that is not entirely
                # numeric
                for y in x:
                    # Remove the digits in each string
                    if y.isdigit():
                        x = x.replace(y, "")
            # Add strings to new_words
            new_words.append(x)
        # update the training instance with the new list of modified words
        self.inst["words"] = new_words
        return(self)

    def remove_stops(self):
        # A given list containing all the stopwords
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be","been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an","the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now"]
        # initialize list that will contain the desired strings
        new_words = []
        # This for loop iterates over the training instance
        for x in self.inst["words"]:
            # Keep all the strings in the training instance that are not
            # also in stopwords
            if x not in stopwords:
                # Add them to new_words
                new_words.append(x)
        # update training instance with new list of modified words
        self.inst["words"] = new_words
        return(self)

    def preprocess_words(self, mode=''):
        # depending on the value of mode, invoke the appropriate methods to
        # transform the training instance in the desired way
        if mode == "keep-digits":
            self.lowercase().remove_symbols().remove_stops()
        elif mode == "keep-stops":
            self.lowercase().remove_symbols().remove_numbers()
        elif mode == "keep-symbols":
            self.lowercase().remove_numbers().remove_stops()
        else:
            self.lowercase().remove_symbols().remove_numbers().remove_stops()
        return(self.inst["words"])

class TrainingSet(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set
        self.inObjHash = []     # Parsed lines, in dictionary/hash
        return

    def get_instances(self):
        return(self.inObjHash)      # FIXME Should protect this more

    def get_lines(self):
        return(self.inObjList)      # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")     # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            # Check for comments
            if line[0] == '%':  # Comments must start with %
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self, mode=''):
        # this method invokes the method preprocess_words from class
        # TrainingInstance and invokes it on the whole training set,
        # according to the specified mode. This preprocesses all training
        # instances in a given training set 
        for x in self.inObjHash:
            x.preprocess_words(mode=mode)
        return

    def return_nfolds(self, num=3):
        # initialize list that will contain the folds created
        fold_list = []
        # this for loop iterates once for each fold desired, creating a
        # deepcopy of the original training set, but emtpying the inObjList and
        # inObjHash attributes of each one
        for n in range(num):
            t_s = copy.deepcopy(self)
            t_s.inObjList = []
            t_s.inObjHash = []
            fold_list.append(t_s)
        # flag for while loop
        flag = 1
        # initialize counter for indexing purposes
        counter = 0
        # this while loop executes until it reaches the end of the training
        # set
        while flag:
            # this for loop divides the original training set into num folds
            # in a "round robin" fashion
            for x in range(num):
                # check if end of training set is reached. if so, break out
                # of for loop
                if counter >= (len(self.inObjList)-1):
                    break
                fold_list[x].inObjList.append(self.inObjList[counter])
                fold_list[x].inObjHash.append(self.inObjHash[counter])
                # increment indexing value
                counter+=1
            # exit while loop when end of training set is reached
            flag = 0
        # return list of folds, each fold being an object of class trainin
        # set  
        return(fold_list)

    def copy(self):
        # using deepcopy, create a copy of the original training set 
        t_s = copy.deepcopy(self)
        # return it
        return(t_s)

    def add_fold(self, tset):
        # create copy of given training set (fold) using deepcopy
        temp = copy.deepcopy(tset)
        # iterate for the length inObjList (same as length of inObjHash)
        for x in range(len(temp.inObjList)):
            # append each training instance of tset to the training set
            self.inObjList.append(temp.inObjList[x])
            self.inObjHash.append(temp.inObjHash[x])
        return


def basemain():
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    print(run1)     # Just to show __str__
    lr = [run1]
    print(lr)       # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, '#weather')

    return


if __name__ == "__main__":
    basemain()