#################################
# Enes Turan Ozcan - 2015400003 #
# Sercan Ersoy     - 2015400039 #   
# Sadi Uysal       - 2015400162 #
#################################

import sys
import numpy as np
import os
from pronto import Ontology
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch
from scipy import spatial
import json


# File paths need to be set here.
# They are not passed via argument to the program.

_dev_path = "BioNLP-OST-2019_BB-norm_dev/"
_train_path = "BioNLP-OST-2019_BB-norm_train/"
_test_path = "BioNLP-OST-2019_BB-norm_test/"
_obo_base = Ontology("OntoBiotope_BioNLP-OST-2019.obo")

_train_set_output_folder = "train_results/"
_dev_set_output_folder = "dev_results/"
_test_set_output_folder = "test_results/"

# -2 at the end excludes LICENCE and README files.
_dev_files = sorted(list(os.walk(_dev_path))[0][2])[:-2]
_train_files = sorted(list(os.walk(_train_path))[0][2])[:-2]
_test_files = sorted(list(os.walk(_test_path))[0][2])[:-2]


###########################################################################
#                           DATA STRUCTURES                               #
###########################################################################


class TrainData:
    def __init__(self, a1, a2, txt, file_name):
        """
            a1 format after parse:
            
            [
                ('T1', 'Title', 'The differential diagnosis ...'),
                ('T2', 'Paragraph', 'The distinction between benign ...'),
                ...
                ('T4', 'Habitat', 'gastric mucosa'),
                ('T5', 'Habitat', 'gastric mucosal-associated lymphoid tissue'),
                ('T6', 'Habitat', 'gastric mucosal'),
                ...
            ]
            
        """
        self.a1 = self.parse_a1(a1)

        """ 
            a2 format after parse:
            
            [
                ('N1', 'OntoBiotope', 'T3', 'OBT:001792'),
                ('N2', 'OntoBiotope', 'T4', 'OBT:001577'),
                ('N3', 'OntoBiotope', 'T5', 'OBT:000334'),
                ...
                ('N8', 'NCBI_Taxonomy', 'T10', '210')
            ]
        """
        self.a2 = self.parse_a2(a2)

        """
            plain text format of abstract.
        """
        self.txt = txt

        """
            Identifier for the test document. i.e: 
            
            if name is BB-norm-F-25036636-004,
            
            corresponding files are:
            
            BB-norm-F-25036636-004.a1
            BB-norm-F-25036636-004.a2
            BB-norm-F-25036636-004.txt
        """
        self.file_name = file_name

        self.predicted_a2 = None

    @staticmethod
    def parse_a1(file_text):
        data = []
        for line in file_text:
            words = line.split()
            atr_0 = words[0]
            atr_1 = words[1]
            for i in range(3, len(words)):
                if ';' in words[i]:
                    # The logic here ignores offsets in the original set.
                    # It's not needed for now since only exact matching is
                    # performing. Must be included in the future if semantic
                    # data is to be extracted, etc.
                    continue
                atr_2 = " ".join(words[i+1:])
                break
            data.append((atr_0, atr_1, atr_2))
        return data

    @staticmethod
    def parse_a2(file_text):
        data = []
        for line in file_text:
            words = line.split()
            mark = words[0]
            base_type = words[1]
            annotation = words[2][words[2].find(":")+1:]
            referent = words[3][words[3].find(":")+1:]
            data.append((mark, base_type, annotation, referent))
        return data

    def set_predicted_a2(self, res_a2):
        self.predicted_a2 = res_a2

    def save_predicted_a2(self, folder):
        if self.predicted_a2 is None:
            raise Exception("a2 file is not yet set!")
        with open(folder + self.file_name + '.a2', 'w+') as out:
            out.write('\n'.join(self.predicted_a2))


class TestData:
    def __init__(self, a1, txt, file_name):
        self.a1 = TrainData.parse_a1(a1)
        self.txt = txt
        self.file_name = file_name
        self.result_a2 = None

    def set_result_a2(self, res_a2):
        self.result_a2 = res_a2

    def save_a2_file(self, folder):
        if self.result_a2 is None:
            raise Exception("a2 file is not yet set!")
        with open(folder + self.file_name + '.a2', 'w+') as out:
            out.write('\n'.join(self.result_a2))


###########################################################################
#                           LOADING DATASET                               #
###########################################################################

def load_train_data(is_dev: bool):
    files, path = (_dev_files, _dev_path) if is_dev else (_train_files, _train_path)
    res = []
    for i in range(0, len(files), 3):
        a1, a2, txt = files[i], files[i + 1], files[i + 2]
        assert a1[:-3] == a2[:-3] == txt[:-4]  # Assert three of them belongs to the same data.
        name = a1[:-3]
        a1 = open(path + a1, "r").readlines()
        a2 = open(path + a2, "r").readlines()
        txt = open(path + txt, "r").read()
        res.append(TrainData(a1, a2, txt, name))
    return res


def load_test_data():
    res = []
    for i in range(0, len(_test_files), 2):
        a1,txt = _test_files[i], _test_files[i+1]
        assert a1[:-3] == txt[:-4]  # Assert two of them belongs to the same data.
        name = a1[:-3]
        a1 = open(_test_path + a1, "r").readlines()
        txt = open(_test_path + txt, "r").read()
        res.append(TestData(a1, txt, name))
    return res


###########################################################################
#                           EXACT MATCHING                                #
###########################################################################

def find_by_exact_match(name):
    # TODO: Find a better lookup solution. May be built in feature of pronto
    for k, v in _obo_base._terms.items():
        if name == v.name:
            return k
    return None


def get_a2_by_exact_match(a1_content, matches_only=True):
    res = []
    for i in a1_content[2:]:
        if i[1] != 'Habitat':
            if not matches_only:
                res.append("--non habitat--")
        else:
            cnt = int(i[0][1:])
            id_ = find_by_exact_match(i[2])
            if matches_only and not id_:
                continue
            res.append("N{}\tOntoBiotope Annotation:T{} Referent:{}".format(cnt, cnt, id_))
    return res


###########################################################################
#                     EXACT SYNONYM MATCHING                              #
###########################################################################

def find_by_exact_synonym_match(name):
    """
    Example usage:
    
    ###################################################
    [Term]
    id: OBT:003432
    name: streched curd cheese
    synonym: "pasta filata" EXACT [TyDI:55043]
    synonym: "plastic curd cheese" EXACT [TyDI:55042]
    synonym: "pulled-curd cheese" EXACT [TyDI:55041]
    is_a: OBT:003381 ! fermented cheese
    ###################################################
    
    In exact matching, "streched curd cheese" term will return OBT:003432
    In synonym exact matching, "pasta filata" term will also return OBT:003432.
    
    Note: name matching is ignored in this function. Check exact match
        before synonym exact match.
    """
    for k, v in _obo_base._terms.items():
        if len(v.synonyms) > 0:
            res = [s.description for s in v.synonyms]
            if name in res:
                return k
    return None


def get_a2_by_exact_synonym_match(a1_content, matches_only=True):
    res = []
    for i in a1_content[2:]:
        if i[1] != 'Habitat':
            if not matches_only:
                res.append("--non habitat--")
        else:
            cnt = int(i[0][1:])
            id_ = find_by_exact_synonym_match(i[2])
            if matches_only and not id_:
                continue
            res.append("N{}\tOntoBiotope Annotation:T{} Referent:{}".format(cnt, cnt, id_))
    return res


###########################################################################
#                         WEIGHTED SIMILARITY                             #
###########################################################################

def find_by_weighted_score(name, score_func, weighted_selection, cos_threshold, weights):
    """
    Returns id of the term having highest similarity score.
    score_func is the function for the calculation of the similarity score.
    Weighted selection and cos selection is equal to threshold or equal to False.
    """
    if weighted_selection:
        scores = {k: score_func(name, v, weights) for k, v in _obo_base.items()}
        id = max(scores, key=lambda k: scores[k])  # find maximum element according to weighted score
        if scores[id] < weighted_selection:
            # if it's below threshold, look for cosine sim greater than cos_threshold
            temp = []
            for k, v in ID_Vector_Dict_Name.items():
                score = cos_similarity_score(encoder(name),
                                             np.array([v, ID_Vector_Dict_OntologyClasses[k]]).sum(axis=0))
                if score > cos_threshold:
                    temp.append((k, score))
            if len(temp):
                temp.sort(key=lambda tup: tup[1], reverse=True)
                return temp[0][0]  # return max of them
            else:
                return None
        else:
            # if weighted score greater than threshold than return id
            return id
    else:
        # if not weighted selection, apply this
        scores = {k: score_func(name, v.name) for k, v in _obo_base.items()}
    return max(scores, key=lambda k: scores[k])


def calc_jaccard_sim(str1, str2):
    """
    Calculates Jaccard similarity of two strings.
    """
    a = set(str1.split())
    b = set(str2.split())
    c = a & b
    return float(len(c)) / (len(a) + len(b) - len(c))


def cos_similarity_score(v1, v2):
    """
    Calculates cosine similarity of two string vectors.
    """
    similarity = 1 - spatial.distance.cosine(v1, v2)
    return similarity


def weighted_score(name, candidate, weights):
    """
    Calculates weighted score of given name and candidate according to given weights.
    Weights for [exact match,synonym jacard similarity,jacard similarity]
    """
    score = 0
    if len(candidate.synonyms) > 0:
        if name == candidate:
            score += weights[0]
        res = [s.description for s in candidate.synonyms]
        for synonym in res:
            score += (weights[1]/len(res))*calc_jaccard_sim(name, synonym)
    else:
        score += (weights[0]+weights[1])
    score += calc_jaccard_sim(name, candidate.name)*weights[2]
    return score

def get_a2_by_similarity_score_2(name,cos_threshold):
    """
    NEW HOPES
    """
    keys=[]
    weights=[2,10,50]
    weighted_threshold=10
    for k,v in _obo_base._terms.items():
      res = [s.description for s in v.synonyms]
      if name==v.name:
        return k #if exact match return 
      else:
        if name in res:
          return k #if synonym exact match return
      jacard_syn=0
      for syn in res:
        jacard_syn+=calc_jaccard_sim(name,syn)  #sum all jacard sim pairs (name, synonym)
    #if any of them has at least 1 common word
      if calc_jaccard_sim(name,v.name)>0 or jacard_syn>0:  
        keys.append(k) #save key 
    temp=[]
    #get key with the max cosine sim
    for k in keys:  
      score_name=cos_similarity_score(encoder(name),ID_Vector_Dict_Name[k])
      if score_name>cos_threshold: #if score>threshold than save it to temp
        temp.append((k,score_name))
    if len(temp):
      temp.sort(key=lambda tup: tup[1], reverse=True)
      return temp[0][0]
    for k,v in _obo_base._terms.items():
        if weighted_score(name,v,weights)>weighted_threshold:
            return k
    #get key with the max cosine sim
    for k,v in ID_Vector_Dict_Name.items():  
      score_name=cos_similarity_score(encoder(name),v)
      if score_name>cos_threshold: #if score>threshold than save it to temp
        temp.append((k,score_name))
    if len(temp):
      temp.sort(key=lambda tup: tup[1], reverse=True)
      return temp[0][0]
    
    return None


def get_a2_by_similarity_score(a1_content, score_func, weighted_selection, cos_threshold, weights):
    """
    Calculates similarity score results for given a1_content.
    """
    res = []
    for i in a1_content:
        if i[1] == 'Habitat':
            cnt = int(i[0][1:])
            id_ = find_by_weighted_score(i[2], score_func, weighted_selection, cos_threshold, weights)
            if not id_:
                continue
            res.append("N{}\tOntoBiotope Annotation:T{} Referent:{}".format(cnt, cnt, id_))
    return res


###########################################################################
#                    EVALUATION & TRAIN HELPERS                           #
###########################################################################

def compare_train_result(results, ref_a1, ref_a2, verbose_fails=False):
    """
      Given a result set, compares the set with ground truth
      discarding non habitat ones.

      /// results :

        list of result strings. Ex:

        [
          'N5\tOntoBiotope Annotation:T5 Referent:OBT:001480',
          'N6\tOntoBiotope Annotation:T6 Referent:OBT:001987'
        ]

        TODO: Store results in a certain format or object other than
         string to make sure all comparisons can be evaluated in a single place
         with the same result object.

      /// ref_a1  :

        a1 field of the object <TrainData>

      /// ref_a2  :

        a2 field of the object <TrainData>
    """
    success = fail = ignored = 0
    total = len(list(filter(lambda x: x[1] == 'Habitat', ref_a1)))
    for r in results:
        splitted = r.split()
        annotation = splitted[2]  # Annotation:T6
        t_id = annotation[annotation.find(":")+1:]  # T6, id for the prediction
        predict = splitted[3][splitted[3].find(":")+1:]  # OBT:001480
        actual = list(filter(lambda x: x[2] == t_id, ref_a2))
        assert len(actual) != 0
        if len(actual) > 1 and verbose_fails:
            print("WARN: Found more than one actual value for annotation {}."
                  " Using one of them randomly. Actual values: ".format(t_id), actual)
        if actual[0][1] != 'OntoBiotope':
            ignored += 1
            continue  # ignore different obo base.
        elif actual[0][3] == predict:
            success += 1
        else:
            fail += 1
            if verbose_fails:
                print("Fail: ", t_id, " -- Predicted: {}, Actual: {}".format(predict, actual[0][3]))
    return success, fail, ignored, total


def get_ontology_terms(ontology_term):
    """
    :param ontology_term:  term of which is_a relations are to be extracted.
    :returns:              is_a relations of the term in BFS order (until the root)
    -----------------------------------------------------------------------
    TODO: Add depth support if necessary.
    Example: Given the Ontology base as follows:

    [Term]
    id: OBT:000007
    name: experimental medium
    is_a: OBT:000001 ! microbial habitat

    [Term]
    id: OBT:000001
    name: microbial habitat
    is_a: OBT:000000 ! root for extraction

    [Term]
    id: OBT:000000
    name: root for extraction

    get_ontology_terms('OBT:000007') returns:

    [('OBT:000001', 'microbial habitat'), ('OBT:000000', 'root for extraction')]

    """
    res = iter(_obo_base[ontology_term].superclasses())
    next(res)  # exclude self.
    return [(p.id, p.name) for p in list(res)]


def encoder(str):
    # Vectorize given string according to pre-trained embedding model and returns it
    input_ids = torch.tensor(tokenizer.encode(str)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0].data[0]  # The last hidden-state is the first element of the output tuple
    vector = last_hidden_states[-1, :]
    return vector


def obo_items_map():
    # encodes every item in ontology base with their name and ontology super classes of it
    # maps those encodings to corresponding dictionaries
    count = 0
    for k, v in _obo_base._terms.items():
        count += 1
        vector = encoder(v.name)
        ID_Vector_Dict_Name[k] = vector.tolist()  # map id to vector at from-Name Dict
        # get ontology super classes for another vector representation
        ontology_classes = get_ontology_terms(k)
        temp_vector = vector
        temp_count = 1
        for (class_id, class_name) in ontology_classes:
            temp_vector += encoder(class_name)
            temp_count += 1
        # map id to vector at from-OntologyClass Dict
        ID_Vector_Dict_OntologyClasses[k] = (temp_vector / temp_count).tolist()
        if count % 50 == 0:
            print("Item :" + str(count) + "--DONE")


def get_a2_find_habitat_representations_from_train_data(a1_content, a2_content):
    # encode training data for each ontology term, and get a representative vector from training data.
    # map those encodings to ID_Vector_Dict_TrainingData dictionary.
    for i in a1_content[2:]:
        if i[1] == 'Habitat':
            link = i[0]  # ex:T6
            name = i[2]  # ex:gastric mucosa
            a2_row = list(filter(lambda x: x[2] == link, a2_content))
            if a2_row[0][1] != 'OntoBiotope':
                continue  # ignore different obo base.
            else:
                referrant_id = a2_row[0][3]
                vector = encoder(name)  # get the name vector of the training namedEntity for this id
                oldVectorSet = ID_Vector_Dict_TrainingData.get(referrant_id)
                if oldVectorSet is None:
                    ID_Vector_Dict_TrainingData[referrant_id] = (1, vector)
                else:
                    new_count = oldVectorSet[0] + 1
                    new_vector = oldVectorSet[1] + vector
                    ID_Vector_Dict_TrainingData[referrant_id] = (new_count, new_vector)


def fill_dictionaries():
    # ---Creating and filling Dictionaries
    obo_items_map()
    # Filling other dictionaries
    count = 0
    for d in _train_data:
        count += 1
        get_a2_find_habitat_representations_from_train_data(a1_content=d.a1, a2_content=d.a2)
    # Normalize sum of training vectors
    for key in ID_Vector_Dict_TrainingData.keys():
        value = ID_Vector_Dict_TrainingData[key]
        ID_Vector_Dict_TrainingData[key] = (value[1] / value[0]).tolist()


def save_dictionary(dict_name, dict_itself):
    # saves given dictionary to json with given name
    """for k in dictItself.keys():
      dictItself[k]=dictItself[k].tolist()
    """
    with open(dict_name+'.json', 'w') as file:
        json.dump(dict_itself, file, sort_keys=True, indent=4)


def load_dictionary(dict_name):
    # loads dictionary from json with given name
    with open(dict_name+'.json', 'r') as file:
        return json.load(file)


def execute(params):
    # executes model with the given parameters and returns detailed result
    (i, j, k, z, cos_threshold, data_part) = params  # 2,10,50,10,0.4

    # Weights for [exact match,synonym exact match, jaccard similarity]
    weighted_score_weights=[i, j, k]
    w = []

    # part_count must be change among parallel trainings.
    part_count = 1
    part_size = int(len(_dev_data) / part_count) + 1

    data_temp = _dev_data[data_part*part_size:(data_part+1)*part_size]
    for d in data_temp:
        weighted_match_res = get_a2_by_similarity_score(d.a1, weighted_score, weighted_selection=z,
            cos_threshold=cos_threshold, weights=weighted_score_weights) # weighted selection is threshold or False
        weighted = compare_train_result(weighted_match_res, d.a1, d.a2)
        w.append(weighted)
    fail = sum([x[1] for x in w])
    success = sum([x[0] for x in w])
    ignored = sum([x[2] for x in w])
    if success+fail == 0:
        s_rate = 0
        s_rate_w_ignored = 0
    else:
        s_rate = success/(success+fail)
        s_rate_w_ignored = success/(success+fail+ignored)
    return (s_rate_w_ignored," Success_rate: "+str(s_rate), " Success Rate with Ignored Cases: "+str(s_rate_w_ignored),
            " Weights and threshold: "+str([i, j, k, z]), " Data Part "+str(data_part), "CONFIG: "+str(params))

###########################################################################
#                           PROGRAM ENTRY                                 #
###########################################################################


_dev_data = load_train_data(True)
_train_data = load_train_data(False)
_test_data = load_test_data()


# Mapping from vectors to id's
ID_Vector_Dict_Name = {}  # ID-vector dict for represantions from HabitatItem_name
ID_Vector_Dict_TrainingData = {}  # ID-vector dict for represantions from trainingData
ID_Vector_Dict_OntologyClasses = {}  # ID-vector dict for represantions from ontology super classes

# pre-trained tokenizer biobert from huggingface more info: https://github.com/huggingface/transformers#installation
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

model = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")  # model


# Save and load dictionaries if necessary

# save_dictionary("ID_Vector_Dict_Name",ID_Vector_Dict_Name)
# save_dictionary("ID_Vector_Dict_TrainingData",ID_Vector_Dict_TrainingData)
# save_dictionary("ID_Vector_Dict_OntologyClasses",ID_Vector_Dict_OntologyClasses)

ID_Vector_Dict_Name = load_dictionary("ID_Vector_Dict_Name")  # ID-vector dict for representations from HabitatItem_name

# ID-vector dict for representations from ontology super classes
ID_Vector_Dict_OntologyClasses = load_dictionary("ID_Vector_Dict_OntologyClasses")

# ID-vector dict for representations from trainingData
# ID_Vector_Dict_TrainingData=load_dictionary("IR_Final/ID_Vector_Dict_TrainingData")


###########################################################################
#                           TESTING PHASE                                 #
###########################################################################

def three_phase_test(a1_content, matches_only=True):
    res = []
    for i in a1_content[2:]:
        if i[1] != 'Habitat':
            if not matches_only:
                res.append("--non habitat--")
        else:
            cnt = int(i[0][1:])
            id_ = get_a2_by_similarity_score_2(i[2],cos_threshold=0.1)
            if not id_:
                continue
            res.append("N{}\tOntoBiotope Annotation:T{} Referent:{}".format(cnt, cnt, id_))
    return res


# Train dataset
for tr in _train_data:
    prediction = three_phase_test(tr.a1)
    if prediction is None:
        raise Exception("None")
    tr.set_predicted_a2(prediction)
    tr.save_predicted_a2(_train_set_output_folder)

# Dev dataset
for d in _dev_data:
    prediction = three_phase_test(d.a1)
    if prediction is None:
        raise Exception("None")
    d.set_predicted_a2(prediction)
    d.save_predicted_a2(_dev_set_output_folder)
"""
# Test Dataset
for t in _test_data:
    prediction = three_phase_test(t.a1)
    if prediction is None:
        raise Exception("None")
    t.set_result_a2(prediction)
    t.save_a2_file(_test_set_output_folder)
"""



