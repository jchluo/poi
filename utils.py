# -*- coding: utf-8 -*-

import time
import cPickle
import logging

log = logging.getLogger(__name__)

class Filename(object):
    def __init__(self, dataset, parent="."):
        self._dataset = dataset
        self.parent = parent
        self.dataset = "%s/datasets/%s/data.txt" % (parent, dataset)
        self.train = "%s/datasets/%s/train.txt" % (parent, dataset)
        self.test = "%s/datasets/%s/test.txt" % (parent, dataset)
        self.locations = "%s/datasets/%s/locations.txt" % (parent, dataset)

    def log(self, model_name):
        return "%s/log/%s-%s.log" % (self.parent, self._dataset, model_name)

    
def save_model(model, filename):
    f = open(filename, "w") 
    cPickle.dump(model, f)
    f.close()


def read_model(filename):
    f = open(filename, "r")
    model = cPickle.load(f)
    f.close()
    return model


def poi_locations(filename):
    locations = {}
    with open(filename) as in_file:
        for line in in_file:
            params = line.strip().split('\t')
            item = int(params[1])
            lat, lon = params[2].split(",")
            lat = float(lat)
            lon = float(lon)
            locations[item] = (lat, lon)
    return locations


def setup_log(filename=None,screen=True):
    #sformat = "%(asctime)s %(filename)s[line:%(lineno)d]"\
    #            " %(levelname)s %(message)s"
    sformat = "%(asctime)s %(filename)s %(levelname)s %(message)s"
    if filename is not None:
        logging.basicConfig(level=logging.DEBUG,
                        format=sformat,
                        filename=filename,
                        filemode="a")
    if screen:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(sformat))
        logging.getLogger('').addHandler(console)
    log.info("new session")

